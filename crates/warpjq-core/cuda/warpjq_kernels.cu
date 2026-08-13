// warpjq CUDA kernels.
//
// Design notes that are not obvious from the code:
//
// * One thread per line, not one warp per line. Log lines are 100-800 bytes,
//   so a warp-cooperative structural index spends most of its time with 31
//   lanes idle while lane 0 walks the structural bitmap. Thread-per-line keeps
//   every lane busy and, more importantly, is a direct transliteration of the
//   CPU scanner in src/json.rs, which is what lets the differential tests
//   assert byte equality instead of "close enough". The warp-cooperative
//   version is a measured optimisation, not a starting point. See
//   docs/ARCHITECTURE.md.
//
// * The kernel is allowed to give up. Any line it cannot decide *exactly* --
//   a number outside the provably-correctly-rounded fast path, nesting deeper
//   than the depth stack, a string that needs materialising to render, is
//   marked WARPJQ_LINE_FALLBACK and finished on the CPU, in order. A GPU JSON
//   parser that is subtly wrong on 0.001% of lines is worse than useless, so
//   the design makes "I am not sure" a first-class outcome.
//
// * Output order is input order, always. Selection uses a stable stream
//   compaction over ascending line indices and the emit kernel writes into
//   offsets from a prefix sum, so rows land where they started.

#include "warpjq_kernels.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdio>
#include <cstring>
#include <new>

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

// Depth of the explicit container stack. One bit per level, so 64 is free.
// Deeper input falls back rather than being silently truncated.
#define WARPJQ_MAX_DEPTH 64
// Boolean stack for the condition RPN. The parser cannot produce anything
// close to this from the v0.1 grammar.
#define WARPJQ_MAX_COND_STACK 32
#define WARPJQ_MAX_SLOTS 16
#define WARPJQ_GROUP_TABLE_BITS 16
#define WARPJQ_GROUP_TABLE_SIZE (1u << WARPJQ_GROUP_TABLE_BITS)
#define WARPJQ_GROUP_MAX_PROBE 64
#define WARPJQ_GROUP_KEY_MAX 65535
#define WARPJQ_BLOCK 256

#define WARPJQ_EMPTY_SLOT 0xFFFFFFFFFFFFFFFFull

// JSON value kinds. Must match json::Kind in Rust.
enum {
  JK_MISSING = 0,
  JK_NULL = 1,
  JK_BOOL = 2,
  JK_NUM = 3,
  JK_STR = 4,
  JK_ARR = 5,
  JK_OBJ = 6
};

// Negative return codes shared by the scanning helpers.
#define R_INVALID (-1)
#define R_FALLBACK (-2)
#define R_TYPE_ERROR (-3)

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

#define CUDA_TRY(expr, op)                                                     \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      set_err(err, err_len, op, cudaGetErrorString(_e));                       \
      return WARPJQ_ERR_CUDA;                                                  \
    }                                                                          \
  } while (0)

static void set_err(char *err, size_t err_len, const char *op,
                    const char *detail) {
  if (!err || err_len == 0) return;
  snprintf(err, err_len, "%s: %s", op, detail);
}

__device__ __forceinline__ bool d_is_ws(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

__device__ __forceinline__ bool d_is_digit(char c) {
  return c >= '0' && c <= '9';
}

__device__ __forceinline__ int d_skip_ws(const char *b, int i, int n) {
  while (i < n && d_is_ws(b[i])) i++;
  return i;
}

__device__ __forceinline__ int d_hex(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

// Past the closing quote of the string starting at b[i] == '"'.
__device__ int d_skip_string(const char *b, int i, int n) {
  int j = i + 1;
  while (j < n) {
    char c = b[j];
    if (c == '"') return j + 1;
    if (c == '\\') {
      if (j + 1 >= n) return R_INVALID;
      char e = b[j + 1];
      if (e == 'u') {
        if (j + 6 > n) return R_INVALID;
        for (int k = 2; k < 6; k++)
          if (d_hex(b[j + k]) < 0) return R_INVALID;
        j += 6;
      } else if (e == '"' || e == '\\' || e == '/' || e == 'b' || e == 'f' ||
                 e == 'n' || e == 'r' || e == 't') {
        j += 2;
      } else {
        return R_INVALID;
      }
    } else if ((unsigned char)c < 0x20) {
      // Raw control characters are illegal inside a JSON string.
      return R_INVALID;
    } else {
      j++;
    }
  }
  return R_INVALID;
}

__device__ int d_skip_number(const char *b, int i, int n) {
  int j = i;
  if (j < n && b[j] == '-') j++;
  int int_start = j;
  while (j < n && d_is_digit(b[j])) j++;
  if (j == int_start) return R_INVALID;
  if (b[int_start] == '0' && j - int_start > 1) return R_INVALID;
  if (j < n && b[j] == '.') {
    j++;
    int f = j;
    while (j < n && d_is_digit(b[j])) j++;
    if (j == f) return R_INVALID;
  }
  if (j < n && (b[j] == 'e' || b[j] == 'E')) {
    j++;
    if (j < n && (b[j] == '+' || b[j] == '-')) j++;
    int e = j;
    while (j < n && d_is_digit(b[j])) j++;
    if (j == e) return R_INVALID;
  }
  return j;
}

__device__ __forceinline__ int d_expect_lit(const char *b, int i, int n,
                                            const char *lit, int len) {
  if (i + len > n) return R_INVALID;
  for (int k = 0; k < len; k++)
    if (b[i + k] != lit[k]) return R_INVALID;
  return i + len;
}

// Iterative whole-value skip. Recursion would blow the per-thread stack on
// deeply nested input and, worse, do so nondeterministically.
__device__ int d_skip_value(const char *b, int i, int n) {
  unsigned long long is_arr = 0;  // bit d set => container at depth d is an array
  int depth = 0;
  enum { WANT_VALUE, WANT_KEY, AFTER_VALUE };
  int state = WANT_VALUE;

  for (;;) {
    if (state == WANT_VALUE) {
      i = d_skip_ws(b, i, n);
      if (i >= n) return R_INVALID;
      char c = b[i];
      if (c == '{') {
        i = d_skip_ws(b, i + 1, n);
        if (i < n && b[i] == '}') {
          i++;
          state = AFTER_VALUE;
        } else {
          if (depth >= WARPJQ_MAX_DEPTH) return R_FALLBACK;
          is_arr &= ~(1ull << depth);
          depth++;
          state = WANT_KEY;
        }
      } else if (c == '[') {
        i = d_skip_ws(b, i + 1, n);
        if (i < n && b[i] == ']') {
          i++;
          state = AFTER_VALUE;
        } else {
          if (depth >= WARPJQ_MAX_DEPTH) return R_FALLBACK;
          is_arr |= (1ull << depth);
          depth++;
          state = WANT_VALUE;
        }
      } else if (c == '"') {
        i = d_skip_string(b, i, n);
        if (i < 0) return i;
        state = AFTER_VALUE;
      } else if (c == 't') {
        i = d_expect_lit(b, i, n, "true", 4);
        if (i < 0) return i;
        state = AFTER_VALUE;
      } else if (c == 'f') {
        i = d_expect_lit(b, i, n, "false", 5);
        if (i < 0) return i;
        state = AFTER_VALUE;
      } else if (c == 'n') {
        i = d_expect_lit(b, i, n, "null", 4);
        if (i < 0) return i;
        state = AFTER_VALUE;
      } else if (c == '-' || d_is_digit(c)) {
        i = d_skip_number(b, i, n);
        if (i < 0) return i;
        state = AFTER_VALUE;
      } else {
        return R_INVALID;
      }
    } else if (state == WANT_KEY) {
      i = d_skip_ws(b, i, n);
      if (i >= n || b[i] != '"') return R_INVALID;
      i = d_skip_string(b, i, n);
      if (i < 0) return i;
      i = d_skip_ws(b, i, n);
      if (i >= n || b[i] != ':') return R_INVALID;
      i++;
      state = WANT_VALUE;
    } else {  // AFTER_VALUE
      if (depth == 0) return i;
      i = d_skip_ws(b, i, n);
      if (i >= n) return R_INVALID;
      char c = b[i];
      bool arr = (is_arr >> (depth - 1)) & 1ull;
      if (c == ',') {
        i++;
        state = arr ? WANT_VALUE : WANT_KEY;
      } else if (c == '}' && !arr) {
        i++;
        depth--;
      } else if (c == ']' && arr) {
        i++;
        depth--;
      } else {
        return R_INVALID;
      }
    }
  }
}

// Whole-line validation: exactly one value, then only whitespace.
__device__ int d_validate(const char *b, int n) {
  int end = d_skip_value(b, 0, n);
  if (end < 0) return end;
  end = d_skip_ws(b, end, n);
  return (end == n) ? 0 : R_INVALID;
}

// ---------------------------------------------------------------------------
// Decoded-string iteration
// ---------------------------------------------------------------------------

// StrIter::next() has to distinguish "the string ended" from "the string is
// malformed". These MUST NOT collide with each other: using R_INVALID (-1) for
// both made every successful comparison look like a decode failure the moment
// it reached the terminator, which quietly sent all of group_by to the CPU.
#define STR_END (-1)
#define STR_ERR (-2)

// Walks a JSON string body yielding *decoded* UTF-8 bytes, so an escaped
// string can be compared against a plain literal without materialising it.
struct StrIter {
  const char *p;
  int i, n;
  unsigned char pend[4];
  int plen, ppos;

  __device__ void init(const char *body, int len) {
    p = body;
    i = 0;
    n = len;
    plen = 0;
    ppos = 0;
  }

  __device__ void emit_cp(unsigned int cp) {
    plen = 0;
    ppos = 0;
    if (cp < 0x80) {
      pend[plen++] = (unsigned char)cp;
    } else if (cp < 0x800) {
      pend[plen++] = (unsigned char)(0xC0 | (cp >> 6));
      pend[plen++] = (unsigned char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      pend[plen++] = (unsigned char)(0xE0 | (cp >> 12));
      pend[plen++] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
      pend[plen++] = (unsigned char)(0x80 | (cp & 0x3F));
    } else {
      pend[plen++] = (unsigned char)(0xF0 | (cp >> 18));
      pend[plen++] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
      pend[plen++] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
      pend[plen++] = (unsigned char)(0x80 | (cp & 0x3F));
    }
  }

  // >=0: next decoded byte. STR_END: end of string. STR_ERR: malformed.
  __device__ int next() {
    if (ppos < plen) return pend[ppos++];
    if (i >= n) return STR_END;
    char c = p[i];
    if (c != '\\') {
      i++;
      return (unsigned char)c;
    }
    if (i + 1 >= n) return STR_ERR;
    char e = p[i + 1];
    i += 2;
    switch (e) {
      case '"': return '"';
      case '\\': return '\\';
      case '/': return '/';
      case 'b': return 0x08;
      case 'f': return 0x0c;
      case 'n': return '\n';
      case 'r': return '\r';
      case 't': return '\t';
      case 'u': break;
      default: return STR_ERR;
    }
    if (i + 4 > n) return STR_ERR;
    unsigned int cp = 0;
    for (int k = 0; k < 4; k++) {
      int h = d_hex(p[i + k]);
      if (h < 0) return STR_ERR;
      cp = (cp << 4) | (unsigned)h;
    }
    i += 4;
    // Surrogate pair, matching the CPU decoder byte for byte.
    if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= n && p[i] == '\\' &&
        p[i + 1] == 'u') {
      unsigned int lo = 0;
      bool ok = true;
      for (int k = 0; k < 4; k++) {
        int h = d_hex(p[i + 2 + k]);
        if (h < 0) { ok = false; break; }
        lo = (lo << 4) | (unsigned)h;
      }
      if (ok && lo >= 0xDC00 && lo < 0xE000) {
        i += 6;
        cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
      } else {
        cp = 0xFFFD;
      }
    } else if (cp >= 0xD800 && cp < 0xE000) {
      cp = 0xFFFD;
    }
    emit_cp(cp);
    return pend[ppos++];
  }
};

// Lexicographic compare of a JSON string body against raw bytes.
// Returns -1/0/1, or R_INVALID.
__device__ int d_str_cmp(const char *body, int len, const unsigned char *want,
                         int wlen) {
  // Fast path: no escapes means a plain memcmp.
  bool esc = false;
  for (int k = 0; k < len; k++) {
    if (body[k] == '\\') { esc = true; break; }
  }
  if (!esc) {
    int m = len < wlen ? len : wlen;
    for (int k = 0; k < m; k++) {
      unsigned char a = (unsigned char)body[k];
      if (a != want[k]) return a < want[k] ? -1 : 1;
    }
    return len == wlen ? 0 : (len < wlen ? -1 : 1);
  }
  StrIter it;
  it.init(body, len);
  int k = 0;
  for (;;) {
    int a = it.next();
    if (a == STR_ERR) return R_INVALID;
    if (a == STR_END) return k == wlen ? 0 : -1;
    if (k >= wlen) return 1;
    if (a != want[k]) return a < want[k] ? -1 : 1;
    k++;
  }
}

__device__ __forceinline__ bool d_key_eq(const char *body, int len,
                                         const unsigned char *want, int wlen,
                                         bool *bad) {
  int c = d_str_cmp(body, len, want, wlen);
  if (c == R_INVALID) { *bad = true; return false; }
  return c == 0;
}

// FNV-1a over the *decoded* bytes, so `"ab"` and `"ab"` hash alike.
// Mirrors query::key_hash on the Rust side.
__device__ unsigned int d_str_hash(const char *body, int len, bool *bad) {
  unsigned int h = 0x811c9dc5u;
  bool esc = false;
  for (int k = 0; k < len; k++) {
    if (body[k] == '\\') { esc = true; break; }
  }
  if (!esc) {
    for (int k = 0; k < len; k++) {
      h ^= (unsigned char)body[k];
      h *= 0x01000193u;
    }
    return h;
  }
  StrIter it;
  it.init(body, len);
  for (;;) {
    int a = it.next();
    if (a == STR_ERR) { *bad = true; return 0; }
    if (a == STR_END) return h;
    h ^= (unsigned)a;
    h *= 0x01000193u;
  }
}

// ---------------------------------------------------------------------------
// Number parsing
// ---------------------------------------------------------------------------

// Correctly-rounded fast path only.
//
// A mantissa below 2^53 times an exactly-representable power of ten is exact
// in one double multiply; outside that window naive accumulation is off by an
// ULP often enough to matter, so those lines go to the CPU instead of being
// quietly wrong. In practice logs are integers and two-decimal fixed point,
// so the fast path takes essentially everything.
__device__ int d_parse_double(const char *b, int n, double *out) {
  int i = 0;
  bool neg = false;
  if (i < n && b[i] == '-') { neg = true; i++; }
  unsigned long long mant = 0;
  int digits = 0;
  while (i < n && d_is_digit(b[i])) {
    if (digits < 19) { mant = mant * 10ull + (unsigned)(b[i] - '0'); digits++; }
    else return R_FALLBACK;  // too many significant digits to be sure
    i++;
  }
  int exp10 = 0;
  if (i < n && b[i] == '.') {
    i++;
    while (i < n && d_is_digit(b[i])) {
      if (digits < 19) {
        mant = mant * 10ull + (unsigned)(b[i] - '0');
        digits++;
        exp10--;
      } else {
        return R_FALLBACK;
      }
      i++;
    }
  }
  if (i < n && (b[i] == 'e' || b[i] == 'E')) {
    i++;
    bool eneg = false;
    if (i < n && (b[i] == '+' || b[i] == '-')) { eneg = b[i] == '-'; i++; }
    int e = 0;
    while (i < n && d_is_digit(b[i])) {
      e = e * 10 + (b[i] - '0');
      if (e > 400) return R_FALLBACK;
      i++;
    }
    exp10 += eneg ? -e : e;
  }
  if (i != n) return R_INVALID;

  if (mant >= (1ull << 53)) return R_FALLBACK;
  if (exp10 < -22 || exp10 > 22) return R_FALLBACK;

  static const double POW10[23] = {
      1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
      1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};

  double v = (double)mant;
  if (exp10 > 0) v *= POW10[exp10];
  else if (exp10 < 0) v /= POW10[-exp10];
  *out = neg ? -v : v;
  return 0;
}

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

struct DevSlot {
  int off;   // relative to the line start; -1 when missing
  int len;
  int kind;
};

__device__ __forceinline__ int d_kind_of(char c) {
  switch (c) {
    case '{': return JK_OBJ;
    case '[': return JK_ARR;
    case '"': return JK_STR;
    case 't':
    case 'f': return JK_BOOL;
    case 'n': return JK_NULL;
    default: return JK_NUM;
  }
}

// Finds `want` in the object at b[i]. Last duplicate wins, which is jq's rule
// and the reason this cannot early-exit on the first match.
__device__ int d_object_get(const char *b, int i, int n,
                            const unsigned char *want, int wlen,
                            unsigned int want_hash, int *vs, int *ve) {
  int j = d_skip_ws(b, i + 1, n);
  if (j < n && b[j] == '}') return 0;  // not found
  int found_s = -1, found_e = -1;
  for (;;) {
    j = d_skip_ws(b, j, n);
    if (j >= n || b[j] != '"') return R_INVALID;
    int key_start = j;
    j = d_skip_string(b, j, n);
    if (j < 0) return j;
    const char *kbody = b + key_start + 1;
    int klen = j - key_start - 2;

    j = d_skip_ws(b, j, n);
    if (j >= n || b[j] != ':') return R_INVALID;
    int val_start = d_skip_ws(b, j + 1, n);
    int val_end = d_skip_value(b, val_start, n);
    if (val_end < 0) return val_end;

    // The length check kills almost every candidate for free; the hash kills
    // the same-length ones before touching the bytes again.
    if (klen == wlen) {
      bool bad = false;
      unsigned int h = d_str_hash(kbody, klen, &bad);
      if (bad) return R_FALLBACK;
      if (h == want_hash && d_key_eq(kbody, klen, want, wlen, &bad)) {
        found_s = val_start;
        found_e = val_end;
      }
      if (bad) return R_FALLBACK;
    } else {
      // A key containing escapes can decode to a different length, so the
      // cheap length check is only valid when there are none.
      bool esc = false;
      for (int k = 0; k < klen; k++)
        if (kbody[k] == '\\') { esc = true; break; }
      if (esc) {
        bool bad = false;
        if (d_key_eq(kbody, klen, want, wlen, &bad)) {
          found_s = val_start;
          found_e = val_end;
        }
        if (bad) return R_FALLBACK;
      }
    }

    j = d_skip_ws(b, val_end, n);
    if (j >= n) return R_INVALID;
    if (b[j] == ',') { j++; continue; }
    if (b[j] == '}') break;
    return R_INVALID;
  }
  if (found_s < 0) return 0;
  *vs = found_s;
  *ve = found_e;
  return 1;
}

__device__ int d_array_get(const char *b, int i, int n, unsigned int idx,
                           int *vs, int *ve) {
  int j = d_skip_ws(b, i + 1, n);
  if (j < n && b[j] == ']') return 0;
  unsigned int k = 0;
  for (;;) {
    int s = d_skip_ws(b, j, n);
    int e = d_skip_value(b, s, n);
    if (e < 0) return e;
    if (k == idx) { *vs = s; *ve = e; return 1; }
    k++;
    j = d_skip_ws(b, e, n);
    if (j >= n) return R_INVALID;
    if (b[j] == ',') { j++; continue; }
    if (b[j] == ']') return 0;
    return R_INVALID;
  }
}

// 0 on success, or one of R_INVALID / R_FALLBACK / R_TYPE_ERROR.
__device__ int d_lookup(const char *line, int n, const warpjq_step *steps,
                        int nsteps, const unsigned char *blob, DevSlot *out) {
  int start = d_skip_ws(line, 0, n);
  if (start >= n) return R_INVALID;
  int kind = d_kind_of(line[start]);
  int end = n;

  if (nsteps == 0) {
    end = d_skip_value(line, start, n);
    if (end < 0) return end;
    out->off = start;
    out->len = end - start;
    out->kind = kind;
    return 0;
  }

  for (int s = 0; s < nsteps; s++) {
    if (kind == JK_MISSING || kind == JK_NULL) {
      out->off = -1;
      out->len = 0;
      out->kind = JK_MISSING;
      return 0;
    }
    const warpjq_step st = steps[s];
    int vs = 0, ve = 0, r;
    if (st.kind == WARPJQ_STEP_KEY) {
      if (kind != JK_OBJ) return R_TYPE_ERROR;
      r = d_object_get(line, start, n, blob + st.key_off, (int)st.key_len,
                       st.key_hash, &vs, &ve);
    } else {
      if (kind != JK_ARR) return R_TYPE_ERROR;
      r = d_array_get(line, start, n, st.index, &vs, &ve);
    }
    if (r < 0) return r;
    if (r == 0) {
      out->off = -1;
      out->len = 0;
      out->kind = JK_MISSING;
      return 0;
    }
    start = vs;
    end = ve;
    kind = d_kind_of(line[vs]);
  }

  out->off = start;
  out->len = end - start;
  out->kind = kind;
  return 0;
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

__device__ __forceinline__ int d_rank(const char *line, const DevSlot &s) {
  switch (s.kind) {
    case JK_MISSING:
    case JK_NULL: return 0;
    case JK_BOOL: return line[s.off] == 't' ? 2 : 1;
    case JK_NUM: return 3;
    case JK_STR: return 4;
    case JK_ARR: return 5;
    default: return 6;
  }
}

__device__ __forceinline__ int d_lit_rank(unsigned int lit_kind) {
  switch (lit_kind) {
    case WARPJQ_LIT_NULL: return 0;
    case WARPJQ_LIT_FALSE: return 1;
    case WARPJQ_LIT_TRUE: return 2;
    case WARPJQ_LIT_NUM: return 3;
    default: return 4;
  }
}

// -1/0/1, or a negative R_* code.
__device__ int d_compare(const char *line, const DevSlot &s,
                         const warpjq_cmp &c, const unsigned char *blob) {
  int sr = d_rank(line, s);
  int lr = d_lit_rank(c.lit_kind);
  if (sr != lr) return sr < lr ? -1 : 1;

  if (c.lit_kind == WARPJQ_LIT_NULL || c.lit_kind == WARPJQ_LIT_TRUE ||
      c.lit_kind == WARPJQ_LIT_FALSE)
    return 0;

  if (c.lit_kind == WARPJQ_LIT_NUM) {
    double v;
    int r = d_parse_double(line + s.off, s.len, &v);
    if (r < 0) return r;
    if (v < c.lit_num) return -1;
    if (v > c.lit_num) return 1;
    return 0;
  }
  // String against string: bodies exclude the quotes.
  int r = d_str_cmp(line + s.off + 1, s.len - 2, blob + c.lit_off,
                    (int)c.lit_len);
  if (r == R_INVALID) return R_INVALID;
  return r;
}

__device__ __forceinline__ bool d_op_accepts(unsigned int op, int ord) {
  switch (op) {
    case WARPJQ_OP_EQ: return ord == 0;
    case WARPJQ_OP_NE: return ord != 0;
    case WARPJQ_OP_LT: return ord < 0;
    case WARPJQ_OP_LE: return ord <= 0;
    case WARPJQ_OP_GT: return ord > 0;
    default: return ord >= 0;
  }
}

__device__ __forceinline__ bool d_truthy(const char *line, const DevSlot &s) {
  if (s.kind == JK_MISSING || s.kind == JK_NULL) return false;
  if (s.kind == JK_BOOL) return line[s.off] == 't';
  return true;
}

// ---------------------------------------------------------------------------
// Program constants in device memory
// ---------------------------------------------------------------------------

struct DevProgram {
  const warpjq_step *steps;
  const warpjq_path *paths;
  const warpjq_cmp *cmps;
  const warpjq_cond_op *cond_rpn;
  const unsigned char *blob;
  const unsigned int *needed_paths;
  const unsigned int *slot_of_path;
  const unsigned int *field_paths;
  const unsigned int *prefix_off;
  const unsigned int *prefix_len;
  unsigned int n_cond;
  unsigned int n_needed;
  unsigned int output_kind;
  unsigned int output_path;
  unsigned int agg_kind;
  unsigned int agg_path;
  unsigned int group_path;
  unsigned int has_filter;
  unsigned int n_fields;
  unsigned int suffix_off;
  unsigned int suffix_len;
  unsigned int csv_mode;
};

// Resolves every path the query needs into `slots`, indexed by slot number.
__device__ int d_extract(const char *line, int n, const DevProgram &p,
                         DevSlot *slots) {
  for (unsigned int k = 0; k < p.n_needed; k++) {
    unsigned int pid = p.needed_paths[k];
    warpjq_path pp = p.paths[pid];
    DevSlot s;
    int r = d_lookup(line, n, p.steps + pp.step_off, (int)pp.step_count, p.blob,
                     &s);
    if (r < 0) return r;
    slots[p.slot_of_path[pid]] = s;
  }
  return 0;
}

__device__ int d_eval_cond(const char *line, const DevProgram &p,
                           const DevSlot *slots, bool *out) {
  bool stack[WARPJQ_MAX_COND_STACK];
  int sp = 0;
  for (unsigned int k = 0; k < p.n_cond; k++) {
    warpjq_cond_op op = p.cond_rpn[k];
    switch (op.op) {
      case WARPJQ_COND_CMP: {
        warpjq_cmp c = p.cmps[op.arg];
        const DevSlot &s = slots[p.slot_of_path[c.path]];
        int ord = d_compare(line, s, c, p.blob);
        if (ord < -1) return ord;  // R_* code
        if (sp >= WARPJQ_MAX_COND_STACK) return R_FALLBACK;
        stack[sp++] = d_op_accepts(c.op, ord);
        break;
      }
      case WARPJQ_COND_TRUTHY: {
        const DevSlot &s = slots[p.slot_of_path[op.arg]];
        if (sp >= WARPJQ_MAX_COND_STACK) return R_FALLBACK;
        stack[sp++] = d_truthy(line, s);
        break;
      }
      case WARPJQ_COND_NOT:
        if (sp < 1) return R_FALLBACK;
        stack[sp - 1] = !stack[sp - 1];
        break;
      case WARPJQ_COND_AND:
        if (sp < 2) return R_FALLBACK;
        stack[sp - 2] = stack[sp - 2] && stack[sp - 1];
        sp--;
        break;
      default:  // OR
        if (sp < 2) return R_FALLBACK;
        stack[sp - 2] = stack[sp - 2] || stack[sp - 1];
        sp--;
        break;
    }
  }
  *out = sp > 0 ? stack[0] : true;
  return 0;
}

// ---------------------------------------------------------------------------
// Line splitting
// ---------------------------------------------------------------------------

struct IsNewline {
  const char *data;
  __device__ __forceinline__ bool operator()(const int &i) const {
    return data[i] == '\n';
  }
};

// Turns newline positions into (offset, length) pairs, trimming a trailing
// '\r' so CRLF input behaves, and flagging blank lines so they neither produce
// output nor count as parse failures.
__global__ void k_build_lines(const char *data, long long n,
                              const int *nl_pos, long long n_nl,
                              long long n_lines, unsigned int *line_off,
                              unsigned int *line_len, unsigned char *status) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n_lines) return;
  long long start = (i == 0) ? 0 : (long long)nl_pos[i - 1] + 1;
  long long end = (i < n_nl) ? (long long)nl_pos[i] : n;
  if (end > start && data[end - 1] == '\r') end--;

  bool blank = true;
  for (long long k = start; k < end; k++) {
    if (!d_is_ws(data[k])) { blank = false; break; }
  }
  line_off[i] = (unsigned int)start;
  line_len[i] = (unsigned int)(end - start);
  status[i] = blank ? WARPJQ_LINE_BLANK : WARPJQ_LINE_OK;
}

// ---------------------------------------------------------------------------
// Group-by hash table
// ---------------------------------------------------------------------------
//
// Open addressing keyed on (kind, decoded key bytes). The key location is
// packed into the single 64-bit word that atomicCAS publishes, so there is no
// window where another thread can see a claimed slot whose key is not yet
// written. That is the classic bug in "CAS a flag then fill the payload" tables.

__device__ __forceinline__ unsigned long long d_pack_key(unsigned int off,
                                                         unsigned int len,
                                                         unsigned int kind) {
  return ((unsigned long long)off << 24) | ((unsigned long long)len << 4) |
         (unsigned long long)(kind & 0xF);
}
__device__ __forceinline__ unsigned int d_key_off(unsigned long long w) {
  return (unsigned int)(w >> 24);
}
__device__ __forceinline__ unsigned int d_key_len(unsigned long long w) {
  return (unsigned int)((w >> 4) & 0xFFFFF);
}
__device__ __forceinline__ unsigned int d_key_kind(unsigned long long w) {
  return (unsigned int)(w & 0xF);
}

// Equality over the decoded key, so `"ab"` and `"ab"` land in one group.
__device__ bool d_group_key_eq(const char *data, unsigned long long a,
                               unsigned long long b, bool *bad) {
  if (d_key_kind(a) != d_key_kind(b)) return false;
  unsigned int ao = d_key_off(a), al = d_key_len(a);
  unsigned int bo = d_key_off(b), bl = d_key_len(b);
  bool esc = false;
  for (unsigned int k = 0; k < al && !esc; k++) esc = data[ao + k] == '\\';
  for (unsigned int k = 0; k < bl && !esc; k++) esc = data[bo + k] == '\\';
  // Overwhelmingly the common case: neither key contains an escape, so the
  // decoded forms are the raw bytes and this is a plain memcmp.
  if (d_key_kind(a) != JK_STR || !esc) {
    if (al != bl) return false;
    for (unsigned int k = 0; k < al; k++)
      if (data[ao + k] != data[bo + k]) return false;
    return true;
  }
  StrIter x, y;
  x.init(data + ao, (int)al);
  y.init(data + bo, (int)bl);
  for (;;) {
    int u = x.next(), v = y.next();
    if (u == STR_ERR || v == STR_ERR) { *bad = true; return false; }
    if (u != v) return false;
    if (u == STR_END) return true;
  }
}

__device__ __forceinline__ double d_ordered_from_bits(unsigned long long u) {
  unsigned long long bits = (u & (1ull << 63)) ? (u ^ (1ull << 63)) : ~u;
  return __longlong_as_double((long long)bits);
}

__device__ void d_atomic_min_double(double *addr, double val) {
  unsigned long long *p = (unsigned long long *)addr;
  unsigned long long old = *p, assumed;
  do {
    assumed = old;
    double cur = __longlong_as_double((long long)assumed);
    if (!(val < cur)) return;
    old = atomicCAS(p, assumed, (unsigned long long)__double_as_longlong(val));
  } while (assumed != old);
}

__device__ void d_atomic_max_double(double *addr, double val) {
  unsigned long long *p = (unsigned long long *)addr;
  unsigned long long old = *p, assumed;
  do {
    assumed = old;
    double cur = __longlong_as_double((long long)assumed);
    if (!(val > cur)) return;
    old = atomicCAS(p, assumed, (unsigned long long)__double_as_longlong(val));
  } while (assumed != old);
}

struct GroupTable {
  unsigned long long *keys;  // WARPJQ_EMPTY_SLOT when free
  unsigned long long *count;
  unsigned long long *numeric;
  double *sum;
  double *minv;
  double *maxv;
  unsigned int *overflow;
};

__global__ void k_group_reset(GroupTable t, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  t.keys[i] = WARPJQ_EMPTY_SLOT;
  t.count[i] = 0;
  t.numeric[i] = 0;
  t.sum[i] = 0.0;
  t.minv[i] = INFINITY;
  t.maxv[i] = -INFINITY;
  if (i == 0) *t.overflow = 0;
}

// Returns the slot index, or 0xFFFFFFFF if the table is too full / the key is
// undecidable. Never returns a wrong slot.
__device__ unsigned int d_group_insert(const char *data, GroupTable t,
                                       unsigned long long packed,
                                       unsigned int hash, bool *bad) {
  unsigned int mask = WARPJQ_GROUP_TABLE_SIZE - 1;
  unsigned int slot = hash & mask;
  for (int probe = 0; probe < WARPJQ_GROUP_MAX_PROBE; probe++) {
    unsigned long long cur = t.keys[slot];
    if (cur == WARPJQ_EMPTY_SLOT) {
      unsigned long long old =
          atomicCAS(&t.keys[slot], WARPJQ_EMPTY_SLOT, packed);
      if (old == WARPJQ_EMPTY_SLOT) return slot;
      cur = old;
    }
    if (d_group_key_eq(data, cur, packed, bad)) return slot;
    if (*bad) return 0xFFFFFFFFu;
    slot = (slot + 1) & mask;
  }
  return 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// Evaluation kernel
// ---------------------------------------------------------------------------

struct ChunkCounters {
  unsigned long long n_blank;
  unsigned long long n_invalid;
  unsigned long long n_type_error;
  unsigned long long n_fallback;
  unsigned int fallback_count;
  unsigned int overflow;
  /* Lowest line index that failed to parse, so --strict can name the actual
   * offending line instead of the chunk it happened to fall in. Initialised
   * to 0xFFFFFFFF (memset 0xFF), not 0, or atomicMin would never move. */
  unsigned int first_invalid;
  /* A value of the wrong type reached a numeric aggregate somewhere in this
   * chunk. The CPU engine tracks the same thing per AggState. */
  unsigned int saw_non_numeric;
};

__global__ void k_eval(const char *data, const unsigned int *line_off,
                       const unsigned int *line_len, long long n_lines,
                       DevProgram p, unsigned char *status, unsigned char *pass,
                       double *agg_value, unsigned int *group_slot,
                       GroupTable table, ChunkCounters *ctr,
                       unsigned int *fallback_idx, unsigned int *fallback_off,
                       unsigned int *fallback_len, unsigned int fallback_cap) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n_lines) return;
  if (status[i] == WARPJQ_LINE_BLANK) {
    pass[i] = 0;
    atomicAdd(&ctr->n_blank, 1ull);
    return;
  }

  const char *line = data + line_off[i];
  int n = (int)line_len[i];
  pass[i] = 0;

  int rc = d_validate(line, n);
  if (rc == R_INVALID) {
    status[i] = WARPJQ_LINE_INVALID;
    atomicAdd(&ctr->n_invalid, 1ull);
    atomicMin(&ctr->first_invalid, (unsigned int)i);
    return;
  }
  if (rc == R_FALLBACK) goto fallback;

  {
    DevSlot slots[WARPJQ_MAX_SLOTS];
#pragma unroll 1
    for (int k = 0; k < WARPJQ_MAX_SLOTS; k++) {
      slots[k].off = -1;
      slots[k].len = 0;
      slots[k].kind = JK_MISSING;
    }

    rc = d_extract(line, n, p, slots);
    if (rc == R_INVALID) {
      status[i] = WARPJQ_LINE_INVALID;
      atomicAdd(&ctr->n_invalid, 1ull);
      atomicMin(&ctr->first_invalid, (unsigned int)i);
      return;
    }
    if (rc == R_TYPE_ERROR) {
      status[i] = WARPJQ_LINE_TYPE_ERROR;
      atomicAdd(&ctr->n_type_error, 1ull);
      return;
    }
    if (rc == R_FALLBACK) goto fallback;

    bool keep = true;
    if (p.has_filter) {
      rc = d_eval_cond(line, p, slots, &keep);
      if (rc == R_INVALID) {
        status[i] = WARPJQ_LINE_INVALID;
        atomicAdd(&ctr->n_invalid, 1ull);
        atomicMin(&ctr->first_invalid, (unsigned int)i);
        return;
      }
      if (rc == R_FALLBACK) goto fallback;
    }
    if (!keep) return;

    // A projection that must be rendered has to be renderable. Anything
    // needing materialisation the kernel cannot do exactly goes to the CPU.
    if (p.output_kind == WARPJQ_OUT_PROJECT || p.output_kind == WARPJQ_OUT_PATH) {
      unsigned int nf =
          (p.output_kind == WARPJQ_OUT_PROJECT) ? p.n_fields : 1u;
      for (unsigned int f = 0; f < nf; f++) {
        unsigned int pid = (p.output_kind == WARPJQ_OUT_PROJECT)
                               ? p.field_paths[f]
                               : p.output_path;
        const DevSlot &s = slots[p.slot_of_path[pid]];
        if (p.csv_mode && s.kind == JK_STR) {
          // CSV needs the decoded text; an escaped string would have to be
          // unescaped into the output, which is the CPU's job for now.
          for (int k = 1; k < s.len - 1; k++)
            if (line[s.off + k] == '\\') goto fallback;
        }
      }
    }

    if (p.output_kind == WARPJQ_OUT_AGG) {
      double v = 0.0;
      bool has_v = false;
      bool non_numeric = false;
      if (p.agg_path != 0xFFFFFFFFu) {
        const DevSlot &s = slots[p.slot_of_path[p.agg_path]];
        if (s.kind == JK_NUM) {
          int r = d_parse_double(line + s.off, s.len, &v);
          if (r == R_FALLBACK) goto fallback;
          if (r < 0) {
            status[i] = WARPJQ_LINE_INVALID;
            atomicAdd(&ctr->n_invalid, 1ull);
            atomicMin(&ctr->first_invalid, (unsigned int)i);
            return;
          }
          has_v = true;
        } else if (s.kind != JK_MISSING && s.kind != JK_NULL) {
          non_numeric = true;
        }
      }
      agg_value[i] = has_v ? v : NAN;
      // Mirrors AggState::saw_non_numeric on the CPU side: the value is
      // ignored by the aggregate, but the fact that one was seen is reported.
      if (non_numeric) atomicExch(&ctr->saw_non_numeric, 1u);

      if (p.group_path != 0xFFFFFFFFu) {
        const DevSlot &gs = slots[p.slot_of_path[p.group_path]];
        unsigned int koff, klen;
        unsigned int kind = gs.kind;
        if (gs.kind == JK_MISSING) {
          // Group missing and explicit null together, as jq does.
          koff = line_off[i];
          klen = 0;
          kind = JK_NULL;
        } else if (gs.kind == JK_STR) {
          koff = line_off[i] + gs.off + 1;
          klen = gs.len - 2;
        } else {
          koff = line_off[i] + gs.off;
          klen = gs.len;
        }
        if (klen > WARPJQ_GROUP_KEY_MAX) goto fallback;

        bool bad = false;
        unsigned int h;
        if (kind == JK_STR) {
          h = d_str_hash(data + koff, (int)klen, &bad);
        } else {
          h = 0x811c9dc5u;
          for (unsigned int k = 0; k < klen; k++) {
            h ^= (unsigned char)data[koff + k];
            h *= 0x01000193u;
          }
        }
        h ^= kind * 0x9E3779B9u;
        if (bad) goto fallback;

        unsigned int slot =
            d_group_insert(data, table, d_pack_key(koff, klen, kind), h, &bad);
        if (bad) goto fallback;
        if (slot == 0xFFFFFFFFu) {
          atomicExch(table.overflow, 1u);
          return;
        }
        group_slot[i] = slot;
        atomicAdd(&table.count[slot], 1ull);
        if (has_v) {
          atomicAdd(&table.numeric[slot], 1ull);
          atomicAdd(&table.sum[slot], v);
          d_atomic_min_double(&table.minv[slot], v);
          d_atomic_max_double(&table.maxv[slot], v);
        }
      }
    }

    pass[i] = 1;
    return;
  }

fallback:
  status[i] = WARPJQ_LINE_FALLBACK;
  atomicAdd(&ctr->n_fallback, 1ull);
  {
    unsigned int k = atomicAdd(&ctr->fallback_count, 1u);
    if (k < fallback_cap) {
      fallback_idx[k] = (unsigned int)i;
      fallback_off[k] = line_off[i];
      fallback_len[k] = line_len[i];
    } else {
      atomicExch(&ctr->overflow, 1u);
    }
  }
}

// ---------------------------------------------------------------------------
// Ungrouped aggregation
// ---------------------------------------------------------------------------

__global__ void k_agg_reduce(const unsigned char *pass, const double *value,
                             long long n_lines, unsigned int has_value,
                             warpjq_agg_partial *partials) {
  typedef cub::BlockReduce<double, WARPJQ_BLOCK> BR;
  typedef cub::BlockReduce<unsigned long long, WARPJQ_BLOCK> BRU;
  __shared__ union {
    typename BR::TempStorage d;
    typename BRU::TempStorage u;
  } tmp;

  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  bool ok = (i < n_lines) && pass[i];
  double v = 0.0;
  bool numeric = false;
  if (ok && has_value) {
    double raw = value[i];
    if (!isnan(raw)) { v = raw; numeric = true; }
  }

  unsigned long long c = ok ? 1ull : 0ull;
  unsigned long long nm = numeric ? 1ull : 0ull;

  unsigned long long sum_c = BRU(tmp.u).Sum(c);
  __syncthreads();
  unsigned long long sum_n = BRU(tmp.u).Sum(nm);
  __syncthreads();
  double sum_v = BR(tmp.d).Sum(numeric ? v : 0.0);
  __syncthreads();
  double min_v = BR(tmp.d).Reduce(numeric ? v : INFINITY, cub::Min());
  __syncthreads();
  double max_v = BR(tmp.d).Reduce(numeric ? v : -INFINITY, cub::Max());

  if (threadIdx.x == 0) {
    warpjq_agg_partial p;
    p.count = sum_c;
    p.numeric = sum_n;
    p.sum = sum_v;
    p.min = min_v;
    p.max = max_v;
    p.saw_non_numeric = 0;
    p._pad = 0;
    partials[blockIdx.x] = p;
  }
}

__global__ void k_agg_final(const warpjq_agg_partial *partials, int n,
                            warpjq_agg_partial *out) {
  // One block, serial over blocks. `n` is (lines / 256), so even a 256 MB
  // chunk of tiny lines lands in the low millions, cheap next to the parse.
  __shared__ unsigned long long s_count, s_num;
  __shared__ double s_sum, s_min, s_max;
  if (threadIdx.x == 0) {
    s_count = 0;
    s_num = 0;
    s_sum = 0.0;
    s_min = INFINITY;
    s_max = -INFINITY;
  }
  __syncthreads();

  unsigned long long lc = 0, ln = 0;
  double ls = 0.0, lmin = INFINITY, lmax = -INFINITY;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    warpjq_agg_partial p = partials[i];
    lc += p.count;
    ln += p.numeric;
    ls += p.sum;
    if (p.min < lmin) lmin = p.min;
    if (p.max > lmax) lmax = p.max;
  }
  atomicAdd(&s_count, lc);
  atomicAdd(&s_num, ln);
  atomicAdd(&s_sum, ls);
  d_atomic_min_double(&s_min, lmin);
  d_atomic_max_double(&s_max, lmax);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->count = s_count;
    out->numeric = s_num;
    out->sum = s_sum;
    out->min = s_min;
    out->max = s_max;
    out->saw_non_numeric = 0;
    out->_pad = 0;
  }
}

// ---------------------------------------------------------------------------
// Output assembly
// ---------------------------------------------------------------------------

// CSV cell length for a value, matching output::write_csv_cell.
__device__ int d_csv_len(const char *line, const DevSlot &s) {
  if (s.kind == JK_MISSING || s.kind == JK_NULL) return 0;
  const char *p;
  int len;
  if (s.kind == JK_STR) {
    p = line + s.off + 1;
    len = s.len - 2;
  } else {
    p = line + s.off;
    len = s.len;
  }
  int quotes = 0;
  bool need = false;
  for (int k = 0; k < len; k++) {
    char c = p[k];
    if (c == ',' || c == '\n' || c == '\r') need = true;
    if (c == '"') { need = true; quotes++; }
  }
  return need ? len + quotes + 2 : len;
}

__device__ int d_csv_emit(char *dst, const char *line, const DevSlot &s) {
  if (s.kind == JK_MISSING || s.kind == JK_NULL) return 0;
  const char *p;
  int len;
  if (s.kind == JK_STR) {
    p = line + s.off + 1;
    len = s.len - 2;
  } else {
    p = line + s.off;
    len = s.len;
  }
  bool need = false;
  for (int k = 0; k < len; k++) {
    char c = p[k];
    if (c == ',' || c == '\n' || c == '\r' || c == '"') { need = true; break; }
  }
  int w = 0;
  if (!need) {
    for (int k = 0; k < len; k++) dst[w++] = p[k];
    return w;
  }
  dst[w++] = '"';
  for (int k = 0; k < len; k++) {
    if (p[k] == '"') dst[w++] = '"';
    dst[w++] = p[k];
  }
  dst[w++] = '"';
  return w;
}

// Row length for each selected line. Re-resolves the output paths rather than
// keeping a slot table for every line in the chunk: the extra parse touches
// only surviving lines, while a slot table would cost memory proportional to
// the whole chunk.
__global__ void k_row_len(const char *data, const unsigned int *line_off,
                          const unsigned int *line_len,
                          const unsigned int *sel_idx, const long long *n_sel,
                          DevProgram p, uint64_t *row_len) {
  long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (k >= *n_sel) return;
  unsigned int i = sel_idx[k];
  const char *line = data + line_off[i];
  int n = (int)line_len[i];

  uint64_t len = 0;
  if (p.output_kind == WARPJQ_OUT_PASSTHROUGH) {
    len = (uint64_t)n + 1;
  } else if (p.output_kind == WARPJQ_OUT_PATH) {
    warpjq_path pp = p.paths[p.output_path];
    DevSlot s;
    if (d_lookup(line, n, p.steps + pp.step_off, (int)pp.step_count, p.blob,
                 &s) < 0) {
      row_len[k] = 0;
      return;
    }
    if (p.csv_mode) {
      len = (uint64_t)d_csv_len(line, s) + 1;
    } else {
      len = (uint64_t)(s.kind == JK_MISSING ? 4 : s.len) + 1;
    }
  } else if (p.output_kind == WARPJQ_OUT_PROJECT) {
    for (unsigned int f = 0; f < p.n_fields; f++) {
      warpjq_path pp = p.paths[p.field_paths[f]];
      DevSlot s;
      if (d_lookup(line, n, p.steps + pp.step_off, (int)pp.step_count, p.blob,
                   &s) < 0) {
        row_len[k] = 0;
        return;
      }
      len += p.prefix_len[f];
      if (p.csv_mode) {
        len += d_csv_len(line, s);
      } else {
        len += (s.kind == JK_MISSING) ? 4 : s.len;
      }
    }
    len += p.suffix_len + 1;
  }
  row_len[k] = len;
}

__global__ void k_emit(const char *data, const unsigned int *line_off,
                       const unsigned int *line_len,
                       const unsigned int *sel_idx, const long long *n_sel,
                       DevProgram p, const uint64_t *row_off,
                       const uint64_t *row_len,
                       unsigned long long out_cap, ChunkCounters *ctr,
                       char *out) {
  long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (k >= *n_sel) return;
  // A projection can expand a line by an unbounded factor. A handful of
  // named fields over short lines beats the 1.5x slack in out_cap easily. The
  // running total is only known on the device at this point, so the check has
  // to happen here: performing it on the host after this kernel returns would
  // be checking for a buffer overrun that already happened.
  if (row_off[k] + row_len[k] > out_cap) {
    atomicExch(&ctr->overflow, 1u);
    return;
  }
  unsigned int i = sel_idx[k];
  const char *line = data + line_off[i];
  int n = (int)line_len[i];
  char *dst = out + row_off[k];
  int w = 0;

  if (p.output_kind == WARPJQ_OUT_PASSTHROUGH) {
    for (int t = 0; t < n; t++) dst[w++] = line[t];
  } else if (p.output_kind == WARPJQ_OUT_PATH) {
    warpjq_path pp = p.paths[p.output_path];
    DevSlot s;
    if (d_lookup(line, n, p.steps + pp.step_off, (int)pp.step_count, p.blob,
                 &s) < 0)
      return;
    if (p.csv_mode) {
      w += d_csv_emit(dst + w, line, s);
    } else if (s.kind == JK_MISSING) {
      dst[w++] = 'n'; dst[w++] = 'u'; dst[w++] = 'l'; dst[w++] = 'l';
    } else {
      for (int t = 0; t < s.len; t++) dst[w++] = line[s.off + t];
    }
  } else if (p.output_kind == WARPJQ_OUT_PROJECT) {
    for (unsigned int f = 0; f < p.n_fields; f++) {
      for (unsigned int t = 0; t < p.prefix_len[f]; t++)
        dst[w++] = (char)p.blob[p.prefix_off[f] + t];
      warpjq_path pp = p.paths[p.field_paths[f]];
      DevSlot s;
      if (d_lookup(line, n, p.steps + pp.step_off, (int)pp.step_count, p.blob,
                   &s) < 0)
        return;
      if (p.csv_mode) {
        w += d_csv_emit(dst + w, line, s);
      } else if (s.kind == JK_MISSING) {
        dst[w++] = 'n'; dst[w++] = 'u'; dst[w++] = 'l'; dst[w++] = 'l';
      } else {
        for (int t = 0; t < s.len; t++) dst[w++] = line[s.off + t];
      }
    }
    for (unsigned int t = 0; t < p.suffix_len; t++)
      dst[w++] = (char)p.blob[p.suffix_off + t];
  }
  dst[w++] = '\n';
}

__global__ void k_group_compact(GroupTable t, unsigned int n,
                                warpjq_group *out, unsigned int *n_out) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long key = t.keys[i];
  if (key == WARPJQ_EMPTY_SLOT) return;
  unsigned int k = atomicAdd(n_out, 1u);
  warpjq_group g;
  g.key_off = d_key_off(key);
  g.key_len = d_key_len(key);
  g.key_kind = d_key_kind(key);
  g._pad = 0;
  g.count = t.count[i];
  g.numeric = t.numeric[i];
  g.sum = t.sum[i];
  g.min = t.minv[i];
  g.max = t.maxv[i];
  out[k] = g;
}

// Sets the last prefix-sum entry so the host can read the total without a
// separate reduction.
__global__ void k_finish_offsets(const uint64_t *row_len,
                                 uint64_t *row_off,
                                 const long long *n_sel) {
  long long n = *n_sel;
  row_off[n] = (n == 0) ? 0ull : row_off[n - 1] + row_len[n - 1];
}

__global__ void k_set_flags(const unsigned char *pass, long long n,
                            unsigned char *flags) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) flags[i] = pass[i];
}

// ===========================================================================
// Host side
// ===========================================================================

namespace {

struct DeviceProgramStorage {
  warpjq_step *steps = nullptr;
  warpjq_path *paths = nullptr;
  warpjq_cmp *cmps = nullptr;
  warpjq_cond_op *cond = nullptr;
  unsigned char *blob = nullptr;
  unsigned int *needed = nullptr;
  unsigned int *slot_of_path = nullptr;
  unsigned int *field_paths = nullptr;
  unsigned int *prefix_off = nullptr;
  unsigned int *prefix_len = nullptr;
};

struct Slot {
  cudaStream_t stream = nullptr;
  cudaEvent_t done = nullptr;

  unsigned char *h_pinned = nullptr;  // staging for the chunk bytes
  char *d_data = nullptr;

  int *d_nl_pos = nullptr;
  long long *d_n_nl = nullptr;
  unsigned int *d_line_off = nullptr;
  unsigned int *d_line_len = nullptr;
  unsigned char *d_status = nullptr;
  unsigned char *d_pass = nullptr;
  unsigned char *d_flags = nullptr;
  double *d_agg_value = nullptr;
  unsigned int *d_group_slot = nullptr;

  unsigned int *d_sel_idx = nullptr;
  long long *d_n_sel = nullptr;
  uint64_t *d_row_len = nullptr;
  uint64_t *d_row_off = nullptr;
  char *d_out = nullptr;

  unsigned int *d_fallback_idx = nullptr;
  unsigned int *d_fallback_off = nullptr;
  unsigned int *d_fallback_len = nullptr;

  ChunkCounters *d_ctr = nullptr;
  warpjq_agg_partial *d_partials = nullptr;
  warpjq_agg_partial *d_agg = nullptr;
  warpjq_group *d_groups = nullptr;
  unsigned int *d_n_groups = nullptr;

  void *d_cub = nullptr;
  size_t cub_bytes = 0;

  // Pinned landing zones for results.
  unsigned int *h_sel_idx = nullptr;
  uint64_t *h_row_off = nullptr;
  unsigned char *h_out = nullptr;
  unsigned int *h_fallback_idx = nullptr;
  unsigned int *h_fallback_off = nullptr;
  unsigned int *h_fallback_len = nullptr;
  ChunkCounters *h_ctr = nullptr;
  warpjq_agg_partial *h_agg = nullptr;
  warpjq_group *h_groups = nullptr;
  unsigned int *h_n_groups = nullptr;
  long long *h_n_sel = nullptr;

  long long n_lines = 0;
  long long n_bytes = 0;

  GroupTable table{};
};

}  // namespace

struct warpjq_ctx {
  DevProgram prog{};
  DeviceProgramStorage store{};
  Slot *slots = nullptr;
  unsigned int n_slots = 0;
  unsigned long long chunk_cap = 0;
  long long max_lines = 0;
  unsigned int fallback_cap = 0;
  unsigned long long out_cap = 0;
  bool needs_groups = false;
  bool needs_output = false;
};

// A line cannot be shorter than `{}\n`, but sizing for that is 85 MB of index
// arrays per 256 MB chunk. Assume a realistic floor instead and fail over to
// the CPU for a chunk that violates it, rather than allocating for the worst
// case that never happens.
#define WARPJQ_MIN_LINE_BYTES 24

static warpjq_status alloc_all(warpjq_ctx *ctx, char *err, size_t err_len) {
  size_t err_len_ = err_len;
  (void)err_len_;
  const long long ml = ctx->max_lines;
  const unsigned int fc = ctx->fallback_cap;

  for (unsigned int s = 0; s < ctx->n_slots; s++) {
    Slot &sl = ctx->slots[s];
    CUDA_TRY(cudaStreamCreateWithFlags(&sl.stream, cudaStreamNonBlocking),
             "cudaStreamCreate");
    CUDA_TRY(cudaEventCreateWithFlags(&sl.done, cudaEventDisableTiming),
             "cudaEventCreate");

    CUDA_TRY(cudaHostAlloc(&sl.h_pinned, ctx->chunk_cap, cudaHostAllocDefault),
             "cudaHostAlloc(staging)");
    CUDA_TRY(cudaMalloc(&sl.d_data, ctx->chunk_cap), "cudaMalloc(data)");

    CUDA_TRY(cudaMalloc(&sl.d_nl_pos, ml * sizeof(int)), "cudaMalloc(nl_pos)");
    CUDA_TRY(cudaMalloc(&sl.d_n_nl, sizeof(long long)), "cudaMalloc(n_nl)");
    CUDA_TRY(cudaMalloc(&sl.d_line_off, ml * sizeof(unsigned int)),
             "cudaMalloc(line_off)");
    CUDA_TRY(cudaMalloc(&sl.d_line_len, ml * sizeof(unsigned int)),
             "cudaMalloc(line_len)");
    CUDA_TRY(cudaMalloc(&sl.d_status, ml), "cudaMalloc(status)");
    CUDA_TRY(cudaMalloc(&sl.d_pass, ml), "cudaMalloc(pass)");
    CUDA_TRY(cudaMalloc(&sl.d_flags, ml), "cudaMalloc(flags)");
    CUDA_TRY(cudaMalloc(&sl.d_ctr, sizeof(ChunkCounters)), "cudaMalloc(ctr)");
    CUDA_TRY(cudaMalloc(&sl.d_fallback_idx, fc * sizeof(unsigned int)),
             "cudaMalloc(fb_idx)");
    CUDA_TRY(cudaMalloc(&sl.d_fallback_off, fc * sizeof(unsigned int)),
             "cudaMalloc(fb_off)");
    CUDA_TRY(cudaMalloc(&sl.d_fallback_len, fc * sizeof(unsigned int)),
             "cudaMalloc(fb_len)");

    CUDA_TRY(cudaMalloc(&sl.d_agg_value, ml * sizeof(double)),
             "cudaMalloc(agg_value)");
    long long nblocks = (ml + WARPJQ_BLOCK - 1) / WARPJQ_BLOCK;
    CUDA_TRY(cudaMalloc(&sl.d_partials, nblocks * sizeof(warpjq_agg_partial)),
             "cudaMalloc(partials)");
    CUDA_TRY(cudaMalloc(&sl.d_agg, sizeof(warpjq_agg_partial)),
             "cudaMalloc(agg)");

    if (ctx->needs_groups) {
      CUDA_TRY(cudaMalloc(&sl.d_group_slot, ml * sizeof(unsigned int)),
               "cudaMalloc(group_slot)");
      size_t nt = WARPJQ_GROUP_TABLE_SIZE;
      CUDA_TRY(cudaMalloc(&sl.table.keys, nt * sizeof(unsigned long long)),
               "cudaMalloc(gt.keys)");
      CUDA_TRY(cudaMalloc(&sl.table.count, nt * sizeof(unsigned long long)),
               "cudaMalloc(gt.count)");
      CUDA_TRY(cudaMalloc(&sl.table.numeric, nt * sizeof(unsigned long long)),
               "cudaMalloc(gt.numeric)");
      CUDA_TRY(cudaMalloc(&sl.table.sum, nt * sizeof(double)),
               "cudaMalloc(gt.sum)");
      CUDA_TRY(cudaMalloc(&sl.table.minv, nt * sizeof(double)),
               "cudaMalloc(gt.min)");
      CUDA_TRY(cudaMalloc(&sl.table.maxv, nt * sizeof(double)),
               "cudaMalloc(gt.max)");
      CUDA_TRY(cudaMalloc(&sl.table.overflow, sizeof(unsigned int)),
               "cudaMalloc(gt.overflow)");
      CUDA_TRY(cudaMalloc(&sl.d_groups, nt * sizeof(warpjq_group)),
               "cudaMalloc(groups)");
      CUDA_TRY(cudaMalloc(&sl.d_n_groups, sizeof(unsigned int)),
               "cudaMalloc(n_groups)");
      CUDA_TRY(cudaHostAlloc(&sl.h_groups, nt * sizeof(warpjq_group),
                             cudaHostAllocDefault),
               "cudaHostAlloc(groups)");
      CUDA_TRY(cudaHostAlloc(&sl.h_n_groups, sizeof(unsigned int),
                             cudaHostAllocDefault),
               "cudaHostAlloc(n_groups)");
    }

    if (ctx->needs_output) {
      CUDA_TRY(cudaMalloc(&sl.d_sel_idx, ml * sizeof(unsigned int)),
               "cudaMalloc(sel_idx)");
      CUDA_TRY(cudaMalloc(&sl.d_row_len, ml * sizeof(uint64_t)),
               "cudaMalloc(row_len)");
      CUDA_TRY(cudaMalloc(&sl.d_row_off, (ml + 1) * sizeof(uint64_t)),
               "cudaMalloc(row_off)");
      CUDA_TRY(cudaMalloc(&sl.d_out, ctx->out_cap), "cudaMalloc(out)");
      CUDA_TRY(cudaHostAlloc(&sl.h_sel_idx, ml * sizeof(unsigned int),
                             cudaHostAllocDefault),
               "cudaHostAlloc(sel_idx)");
      CUDA_TRY(cudaHostAlloc(&sl.h_row_off, (ml + 1) * sizeof(uint64_t),
                             cudaHostAllocDefault),
               "cudaHostAlloc(row_off)");
      CUDA_TRY(cudaHostAlloc(&sl.h_out, ctx->out_cap, cudaHostAllocDefault),
               "cudaHostAlloc(out)");
    }
    CUDA_TRY(cudaMalloc(&sl.d_n_sel, sizeof(long long)), "cudaMalloc(n_sel)");

    CUDA_TRY(cudaHostAlloc(&sl.h_ctr, sizeof(ChunkCounters), cudaHostAllocDefault),
             "cudaHostAlloc(ctr)");
    CUDA_TRY(cudaHostAlloc(&sl.h_agg, sizeof(warpjq_agg_partial),
                           cudaHostAllocDefault),
             "cudaHostAlloc(agg)");
    CUDA_TRY(cudaHostAlloc(&sl.h_n_sel, sizeof(long long), cudaHostAllocDefault),
             "cudaHostAlloc(n_sel)");
    CUDA_TRY(cudaHostAlloc(&sl.h_fallback_idx, fc * sizeof(unsigned int),
                           cudaHostAllocDefault),
             "cudaHostAlloc(fb_idx)");
    CUDA_TRY(cudaHostAlloc(&sl.h_fallback_off, fc * sizeof(unsigned int),
                           cudaHostAllocDefault),
             "cudaHostAlloc(fb_off)");
    CUDA_TRY(cudaHostAlloc(&sl.h_fallback_len, fc * sizeof(unsigned int),
                           cudaHostAllocDefault),
             "cudaHostAlloc(fb_len)");

    // Size the CUB scratch once, for the largest of the three uses.
    //
    // These must be sized with the *real* maximum item counts. CUB's temp
    // storage grows with the number of tiles, so sizing the newline scan with
    // a placeholder count produces a buffer that works on small inputs and
    // fails with "invalid argument" once a chunk is big enough to need more
    // tiles, so it passes every quick test and breaks on real data.
    size_t b1 = 0, b2 = 0;
    IsNewline pred{sl.d_data};
    thrust::counting_iterator<int> counter(0);
    cudaError_t e = cub::DeviceSelect::If(nullptr, b1, counter, sl.d_nl_pos,
                                          sl.d_n_nl, (int)ctx->chunk_cap, pred);
    if (e != cudaSuccess) {
      set_err(err, err_len, "cub::DeviceSelect::If(size)",
              cudaGetErrorString(e));
      return WARPJQ_ERR_CUDA;
    }
    e = cub::DeviceScan::ExclusiveSum(nullptr, b2, sl.d_row_len, sl.d_row_off,
                                      (int)(ml > 0 ? ml : 1));
    // `ml` is the true per-chunk line ceiling, so this one is already sized
    // for the worst case.
    if (e != cudaSuccess) {
      set_err(err, err_len, "cub::DeviceScan(size)", cudaGetErrorString(e));
      return WARPJQ_ERR_CUDA;
    }
    size_t b3 = 0;
    e = cub::DeviceSelect::Flagged(nullptr, b3, counter, sl.d_flags,
                                   sl.d_sel_idx, sl.d_n_sel, (int)(ml > 0 ? ml : 1));
    if (e != cudaSuccess) {
      set_err(err, err_len, "cub::DeviceSelect::Flagged(size)",
              cudaGetErrorString(e));
      return WARPJQ_ERR_CUDA;
    }
    sl.cub_bytes = b1 > b2 ? b1 : b2;
    if (b3 > sl.cub_bytes) sl.cub_bytes = b3;
    CUDA_TRY(cudaMalloc(&sl.d_cub, sl.cub_bytes), "cudaMalloc(cub)");
  }
  return WARPJQ_OK;
}

static void free_slot(Slot &sl) {
  if (sl.stream) cudaStreamDestroy(sl.stream);
  if (sl.done) cudaEventDestroy(sl.done);
  cudaFreeHost(sl.h_pinned);
  cudaFree(sl.d_data);
  cudaFree(sl.d_nl_pos);
  cudaFree(sl.d_n_nl);
  cudaFree(sl.d_line_off);
  cudaFree(sl.d_line_len);
  cudaFree(sl.d_status);
  cudaFree(sl.d_pass);
  cudaFree(sl.d_flags);
  cudaFree(sl.d_agg_value);
  cudaFree(sl.d_group_slot);
  cudaFree(sl.d_sel_idx);
  cudaFree(sl.d_n_sel);
  cudaFree(sl.d_row_len);
  cudaFree(sl.d_row_off);
  cudaFree(sl.d_out);
  cudaFree(sl.d_fallback_idx);
  cudaFree(sl.d_fallback_off);
  cudaFree(sl.d_fallback_len);
  cudaFree(sl.d_ctr);
  cudaFree(sl.d_partials);
  cudaFree(sl.d_agg);
  cudaFree(sl.d_groups);
  cudaFree(sl.d_n_groups);
  cudaFree(sl.d_cub);
  cudaFree(sl.table.keys);
  cudaFree(sl.table.count);
  cudaFree(sl.table.numeric);
  cudaFree(sl.table.sum);
  cudaFree(sl.table.minv);
  cudaFree(sl.table.maxv);
  cudaFree(sl.table.overflow);
  cudaFreeHost(sl.h_sel_idx);
  cudaFreeHost(sl.h_row_off);
  cudaFreeHost(sl.h_out);
  cudaFreeHost(sl.h_fallback_idx);
  cudaFreeHost(sl.h_fallback_off);
  cudaFreeHost(sl.h_fallback_len);
  cudaFreeHost(sl.h_ctr);
  cudaFreeHost(sl.h_agg);
  cudaFreeHost(sl.h_groups);
  cudaFreeHost(sl.h_n_groups);
  cudaFreeHost(sl.h_n_sel);
}

template <typename T>
static cudaError_t upload(T **dst, const T *src, size_t n) {
  if (n == 0) {
    *dst = nullptr;
    return cudaSuccess;
  }
  cudaError_t e = cudaMalloc(dst, n * sizeof(T));
  if (e != cudaSuccess) return e;
  return cudaMemcpy(*dst, src, n * sizeof(T), cudaMemcpyHostToDevice);
}

extern "C" {

warpjq_status warpjq_probe(char *err, size_t err_len) {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess) {
    set_err(err, err_len, "no usable CUDA device", cudaGetErrorString(e));
    return WARPJQ_ERR_NO_DEVICE;
  }
  if (n == 0) {
    set_err(err, err_len, "no usable CUDA device", "device count is zero");
    return WARPJQ_ERR_NO_DEVICE;
  }
  cudaDeviceProp prop;
  e = cudaGetDeviceProperties(&prop, 0);
  if (e != cudaSuccess) {
    set_err(err, err_len, "cudaGetDeviceProperties", cudaGetErrorString(e));
    return WARPJQ_ERR_NO_DEVICE;
  }
  // Compaction and the atomics here need Pascal or newer.
  if (prop.major < 6) {
    set_err(err, err_len, "GPU is too old",
            "warpjq needs compute capability 6.0 or newer");
    return WARPJQ_ERR_NO_DEVICE;
  }
  return WARPJQ_OK;
}

warpjq_status warpjq_device_name(char *buf, size_t len) {
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return WARPJQ_ERR_NO_DEVICE;
  snprintf(buf, len, "%s (sm_%d%d, %zu MiB)", prop.name, prop.major, prop.minor,
           (size_t)(prop.totalGlobalMem >> 20));
  return WARPJQ_OK;
}

warpjq_status warpjq_abi_check(const uint64_t *sizes, size_t n) {
  const uint64_t mine[] = {
      (uint64_t)sizeof(warpjq_step),        (uint64_t)sizeof(warpjq_path),
      (uint64_t)sizeof(warpjq_cmp),         (uint64_t)sizeof(warpjq_cond_op),
      (uint64_t)sizeof(warpjq_agg_partial), (uint64_t)sizeof(warpjq_group),
  };
  if (n != sizeof(mine) / sizeof(mine[0])) return WARPJQ_ERR_ABI;
  for (size_t i = 0; i < n; i++)
    if (sizes[i] != mine[i]) return WARPJQ_ERR_ABI;
  return WARPJQ_OK;
}

warpjq_status warpjq_ctx_create(const warpjq_program *prog,
                                uint64_t max_chunk_bytes, uint32_t n_slots,
                                warpjq_ctx **out, char *err, size_t err_len) {
  if (!prog || !out || n_slots == 0 || max_chunk_bytes == 0)
    return WARPJQ_ERR_INVALID_ARG;
  // d_pack_key folds a byte offset into the chunk into 40 bits alongside a
  // 20-bit length. Both hold comfortably at any sane chunk size, but the
  // packing would silently alias group keys if that ever stopped being true,
  // so make it a loud precondition rather than a latent one.
  if (max_chunk_bytes >= (1ull << 40)) {
    set_err(err, err_len, "chunk is too large",
            "the group-key packing holds offsets below 2^40 bytes");
    return WARPJQ_ERR_INVALID_ARG;
  }
  if (prog->n_needed > WARPJQ_MAX_SLOTS) {
    set_err(err, err_len, "query is too wide",
            "more distinct field paths than the kernel slot table holds");
    return WARPJQ_ERR_INVALID_ARG;
  }

  warpjq_ctx *ctx = new (std::nothrow) warpjq_ctx();
  if (!ctx) return WARPJQ_ERR_OOM;
  ctx->n_slots = n_slots;
  ctx->chunk_cap = max_chunk_bytes;
  ctx->max_lines = (long long)(max_chunk_bytes / WARPJQ_MIN_LINE_BYTES) + 2;
  ctx->fallback_cap = (unsigned int)(ctx->max_lines / 16 + 1024);
  ctx->needs_groups = prog->group_path != 0xFFFFFFFFu;
  ctx->needs_output = prog->output_kind != WARPJQ_OUT_AGG;
  // Worst case a projection is longer than the line it came from, so leave
  // slack rather than silently truncating rows.
  ctx->out_cap = max_chunk_bytes + (max_chunk_bytes / 2) + (1u << 20);

  DeviceProgramStorage &st = ctx->store;
  cudaError_t e;
#define UP(field, src, count)                                                  \
  e = upload(&st.field, src, count);                                           \
  if (e != cudaSuccess) {                                                      \
    set_err(err, err_len, "uploading the compiled query",                      \
            cudaGetErrorString(e));                                            \
    warpjq_ctx_destroy(ctx);                                                   \
    return WARPJQ_ERR_CUDA;                                                    \
  }
  UP(steps, prog->steps, prog->n_steps);
  UP(paths, prog->paths, prog->n_paths);
  UP(cmps, prog->cmps, prog->n_cmps);
  UP(cond, prog->cond_rpn, prog->n_cond);
  UP(blob, prog->blob, prog->n_blob);
  UP(needed, prog->needed_paths, prog->n_needed);
  UP(slot_of_path, prog->slot_of_path, prog->n_paths);
  UP(field_paths, prog->field_paths, prog->n_fields);
  UP(prefix_off, prog->prefix_off, prog->n_fields);
  UP(prefix_len, prog->prefix_len, prog->n_fields);
#undef UP

  DevProgram &dp = ctx->prog;
  dp.steps = st.steps;
  dp.paths = st.paths;
  dp.cmps = st.cmps;
  dp.cond_rpn = st.cond;
  dp.blob = st.blob;
  dp.needed_paths = st.needed;
  dp.slot_of_path = st.slot_of_path;
  dp.field_paths = st.field_paths;
  dp.prefix_off = st.prefix_off;
  dp.prefix_len = st.prefix_len;
  dp.n_cond = prog->n_cond;
  dp.n_needed = prog->n_needed;
  dp.output_kind = prog->output_kind;
  dp.output_path = prog->output_path;
  dp.agg_kind = prog->agg_kind;
  dp.agg_path = prog->agg_path;
  dp.group_path = prog->group_path;
  dp.has_filter = prog->has_filter;
  dp.n_fields = prog->n_fields;
  dp.suffix_off = prog->suffix_off;
  dp.suffix_len = prog->suffix_len;
  dp.csv_mode = prog->csv_mode;

  ctx->slots = new (std::nothrow) Slot[n_slots];
  if (!ctx->slots) {
    warpjq_ctx_destroy(ctx);
    return WARPJQ_ERR_OOM;
  }

  warpjq_status s = alloc_all(ctx, err, err_len);
  if (s != WARPJQ_OK) {
    warpjq_ctx_destroy(ctx);
    return s;
  }
  *out = ctx;
  return WARPJQ_OK;
}

void warpjq_ctx_destroy(warpjq_ctx *ctx) {
  if (!ctx) return;
  if (ctx->slots) {
    for (unsigned int i = 0; i < ctx->n_slots; i++) free_slot(ctx->slots[i]);
    delete[] ctx->slots;
  }
  cudaFree(ctx->store.steps);
  cudaFree(ctx->store.paths);
  cudaFree(ctx->store.cmps);
  cudaFree(ctx->store.cond);
  cudaFree(ctx->store.blob);
  cudaFree(ctx->store.needed);
  cudaFree(ctx->store.slot_of_path);
  cudaFree(ctx->store.field_paths);
  cudaFree(ctx->store.prefix_off);
  cudaFree(ctx->store.prefix_len);
  delete ctx;
}

uint8_t *warpjq_slot_buffer(warpjq_ctx *ctx, uint32_t slot) {
  if (!ctx || slot >= ctx->n_slots) return nullptr;
  return ctx->slots[slot].h_pinned;
}

uint64_t warpjq_slot_capacity(const warpjq_ctx *ctx) {
  return ctx ? ctx->chunk_cap : 0;
}

warpjq_status warpjq_submit(warpjq_ctx *ctx, uint32_t slot, uint64_t n_bytes,
                            char *err, size_t err_len) {
  if (!ctx || slot >= ctx->n_slots) return WARPJQ_ERR_INVALID_ARG;
  if (n_bytes > ctx->chunk_cap) return WARPJQ_ERR_INVALID_ARG;
  Slot &sl = ctx->slots[slot];
  sl.n_bytes = (long long)n_bytes;
  cudaStream_t st = sl.stream;

  if (n_bytes == 0) {
    sl.n_lines = 0;
    CUDA_TRY(cudaMemsetAsync(sl.d_ctr, 0, sizeof(ChunkCounters), st),
             "memset(ctr)");
    CUDA_TRY(cudaEventRecord(sl.done, st), "eventRecord");
    return WARPJQ_OK;
  }

  CUDA_TRY(cudaMemcpyAsync(sl.d_data, sl.h_pinned, n_bytes,
                           cudaMemcpyHostToDevice, st),
           "H2D chunk");
  CUDA_TRY(cudaMemsetAsync(sl.d_ctr, 0, sizeof(ChunkCounters), st),
           "memset(ctr)");
  // atomicMin needs a maximal starting value, which a zeroing memset cannot
  // give it.
  CUDA_TRY(cudaMemsetAsync(&sl.d_ctr->first_invalid, 0xFF,
                           sizeof(unsigned int), st),
           "memset(first_invalid)");

  // 1. Newline positions.
  IsNewline pred{sl.d_data};
  thrust::counting_iterator<int> counter(0);
  size_t cub_bytes = sl.cub_bytes;
  CUDA_TRY(cub::DeviceSelect::If(sl.d_cub, cub_bytes, counter, sl.d_nl_pos,
                                 sl.d_n_nl, (int)n_bytes, pred, st),
           "cub::DeviceSelect::If(newlines)");

  // The line count depends on whether the chunk ends with a newline. The
  // chunker guarantees it does except for the last chunk of a file, so read
  // the final byte from the host copy rather than syncing the device.
  bool ends_nl = sl.h_pinned[n_bytes - 1] == '\n';
  // n_nl is on the device; the host bound is what the kernels are sized for.
  // k_build_lines clamps using n_nl, so an over-estimate here is harmless.
  long long max_lines_here =
      (long long)(n_bytes / WARPJQ_MIN_LINE_BYTES) + 2;
  if (max_lines_here > ctx->max_lines) max_lines_here = ctx->max_lines;

  // Copy the newline count into the slot's pinned cell so k_build_lines and
  // the launch bounds agree without a full device sync.
  CUDA_TRY(cudaMemcpyAsync(sl.h_n_sel, sl.d_n_nl, sizeof(long long),
                           cudaMemcpyDeviceToHost, st),
           "D2H(n_nl)");
  CUDA_TRY(cudaStreamSynchronize(st), "sync(n_nl)");
  long long n_nl = *sl.h_n_sel;
  long long n_lines = n_nl + (ends_nl ? 0 : 1);
  if (n_lines > ctx->max_lines) {
    // Far more, far shorter lines than the buffers were sized for. Say so and
    // let the host redo the chunk on the CPU rather than truncating.
    sl.n_lines = -1;
    CUDA_TRY(cudaEventRecord(sl.done, st), "eventRecord");
    return WARPJQ_OK;
  }
  sl.n_lines = n_lines;
  if (n_lines == 0) {
    CUDA_TRY(cudaEventRecord(sl.done, st), "eventRecord");
    return WARPJQ_OK;
  }

  const int B = WARPJQ_BLOCK;
  long long grid = (n_lines + B - 1) / B;

  k_build_lines<<<(unsigned)grid, B, 0, st>>>(sl.d_data, (long long)n_bytes,
                                              sl.d_nl_pos, n_nl, n_lines,
                                              sl.d_line_off, sl.d_line_len,
                                              sl.d_status);

  if (ctx->needs_groups) {
    unsigned int gt = WARPJQ_GROUP_TABLE_SIZE;
    k_group_reset<<<(gt + B - 1) / B, B, 0, st>>>(sl.table, gt);
    CUDA_TRY(cudaMemsetAsync(sl.d_n_groups, 0, sizeof(unsigned int), st),
             "memset(n_groups)");
  }

  GroupTable table = ctx->needs_groups ? sl.table : GroupTable{};
  k_eval<<<(unsigned)grid, B, 0, st>>>(
      sl.d_data, sl.d_line_off, sl.d_line_len, n_lines, ctx->prog, sl.d_status,
      sl.d_pass, sl.d_agg_value, sl.d_group_slot, table, sl.d_ctr,
      sl.d_fallback_idx, sl.d_fallback_off, sl.d_fallback_len,
      ctx->fallback_cap);

  if (ctx->prog.output_kind == WARPJQ_OUT_AGG) {
    unsigned int has_value = ctx->prog.agg_path != 0xFFFFFFFFu;
    k_agg_reduce<<<(unsigned)grid, B, 0, st>>>(sl.d_pass, sl.d_agg_value,
                                               n_lines, has_value,
                                               sl.d_partials);
    k_agg_final<<<1, 256, 0, st>>>(sl.d_partials, (int)grid, sl.d_agg);
    if (ctx->needs_groups) {
      unsigned int gt = WARPJQ_GROUP_TABLE_SIZE;
      k_group_compact<<<(gt + B - 1) / B, B, 0, st>>>(sl.table, gt, sl.d_groups,
                                                      sl.d_n_groups);
      CUDA_TRY(cudaMemcpyAsync(sl.h_n_groups, sl.d_n_groups,
                               sizeof(unsigned int), cudaMemcpyDeviceToHost, st),
               "D2H(n_groups)");
      // The compacted array is copied in warpjq_wait, sized by the count
      // above. Copying the whole 65536-entry table here would move ~3.6 MB
      // per chunk to collect the handful of groups that actually exist.
    }
    CUDA_TRY(cudaMemcpyAsync(sl.h_agg, sl.d_agg, sizeof(warpjq_agg_partial),
                             cudaMemcpyDeviceToHost, st),
             "D2H(agg)");
  } else {
    k_set_flags<<<(unsigned)grid, B, 0, st>>>(sl.d_pass, n_lines, sl.d_flags);
    cub_bytes = sl.cub_bytes;
    CUDA_TRY(cub::DeviceSelect::Flagged(sl.d_cub, cub_bytes, counter,
                                        sl.d_flags, sl.d_sel_idx, sl.d_n_sel,
                                        (int)n_lines, st),
             "cub::DeviceSelect::Flagged");
    CUDA_TRY(cudaMemsetAsync(sl.d_row_len, 0,
                             (size_t)n_lines * sizeof(unsigned long long), st),
             "memset(row_len)");
    k_row_len<<<(unsigned)grid, B, 0, st>>>(sl.d_data, sl.d_line_off,
                                            sl.d_line_len, sl.d_sel_idx,
                                            sl.d_n_sel, ctx->prog, sl.d_row_len);
    cub_bytes = sl.cub_bytes;
    CUDA_TRY(cub::DeviceScan::ExclusiveSum(sl.d_cub, cub_bytes, sl.d_row_len,
                                           sl.d_row_off, (int)n_lines, st),
             "cub::DeviceScan::ExclusiveSum");
    k_finish_offsets<<<1, 1, 0, st>>>(sl.d_row_len, sl.d_row_off, sl.d_n_sel);
    k_emit<<<(unsigned)grid, B, 0, st>>>(
        sl.d_data, sl.d_line_off, sl.d_line_len, sl.d_sel_idx, sl.d_n_sel,
        ctx->prog, sl.d_row_off, sl.d_row_len, ctx->out_cap, sl.d_ctr,
        sl.d_out);
    CUDA_TRY(cudaMemcpyAsync(sl.h_n_sel, sl.d_n_sel, sizeof(long long),
                             cudaMemcpyDeviceToHost, st),
             "D2H(n_sel)");
  }

  CUDA_TRY(cudaMemcpyAsync(sl.h_ctr, sl.d_ctr, sizeof(ChunkCounters),
                           cudaMemcpyDeviceToHost, st),
           "D2H(counters)");
  CUDA_TRY(cudaEventRecord(sl.done, st), "eventRecord");
  return WARPJQ_OK;
}

warpjq_status warpjq_wait(warpjq_ctx *ctx, uint32_t slot,
                          warpjq_chunk_result *out, char *err, size_t err_len) {
  if (!ctx || slot >= ctx->n_slots || !out) return WARPJQ_ERR_INVALID_ARG;
  Slot &sl = ctx->slots[slot];
  CUDA_TRY(cudaStreamSynchronize(sl.stream), "stream sync");
  cudaError_t last = cudaGetLastError();
  if (last != cudaSuccess) {
    set_err(err, err_len, "kernel", cudaGetErrorString(last));
    return WARPJQ_ERR_CUDA;
  }

  memset(out, 0, sizeof(*out));

  if (sl.n_lines < 0) {
    // More lines than the index buffers hold.
    out->chunk_overflow = 1;
    return WARPJQ_OK;
  }
  out->n_lines = (uint64_t)sl.n_lines;
  if (sl.n_lines == 0) return WARPJQ_OK;

  ChunkCounters c = *sl.h_ctr;
  out->n_blank = c.n_blank;
  out->n_invalid = c.n_invalid;
  out->n_type_error = c.n_type_error;
  out->n_fallback = c.n_fallback;
  out->chunk_overflow = c.overflow;
  out->first_invalid = c.first_invalid;
  out->agg.saw_non_numeric = c.saw_non_numeric;

  if (c.overflow) {
    // Either the fallback list or the output buffer did not fit. Nothing was
    // written past the end of a buffer (the kernels check before storing)
    // but this chunk's results are incomplete, so the host redoes it on the
    // CPU rather than emitting a partial answer.
    return WARPJQ_OK;
  }

  unsigned int nfb = c.fallback_count;
  if (nfb > ctx->fallback_cap) nfb = ctx->fallback_cap;
  if (nfb > 0) {
    CUDA_TRY(cudaMemcpy(sl.h_fallback_idx, sl.d_fallback_idx,
                        nfb * sizeof(unsigned int), cudaMemcpyDeviceToHost),
             "D2H(fallback idx)");
    CUDA_TRY(cudaMemcpy(sl.h_fallback_off, sl.d_fallback_off,
                        nfb * sizeof(unsigned int), cudaMemcpyDeviceToHost),
             "D2H(fallback off)");
    CUDA_TRY(cudaMemcpy(sl.h_fallback_len, sl.d_fallback_len,
                        nfb * sizeof(unsigned int), cudaMemcpyDeviceToHost),
             "D2H(fallback len)");
    // The kernel appends in completion order; the host merge needs ascending
    // line indices. Sorting a handful of entries on the host is cheaper than
    // a device sort, and fallbacks are rare by construction.
    for (unsigned int i = 1; i < nfb; i++) {
      unsigned int ki = sl.h_fallback_idx[i], ko = sl.h_fallback_off[i],
                   kl = sl.h_fallback_len[i];
      int j = (int)i - 1;
      while (j >= 0 && sl.h_fallback_idx[j] > ki) {
        sl.h_fallback_idx[j + 1] = sl.h_fallback_idx[j];
        sl.h_fallback_off[j + 1] = sl.h_fallback_off[j];
        sl.h_fallback_len[j + 1] = sl.h_fallback_len[j];
        j--;
      }
      sl.h_fallback_idx[j + 1] = ki;
      sl.h_fallback_off[j + 1] = ko;
      sl.h_fallback_len[j + 1] = kl;
    }
  }
  out->fallback_idx = sl.h_fallback_idx;
  out->fallback_off = sl.h_fallback_off;
  out->fallback_len = sl.h_fallback_len;
  out->n_fallback = nfb;

  if (ctx->prog.output_kind == WARPJQ_OUT_AGG) {
    out->agg = *sl.h_agg;
    if (ctx->needs_groups) {
      unsigned int ng = *sl.h_n_groups;
      if (ng > WARPJQ_GROUP_TABLE_SIZE) ng = WARPJQ_GROUP_TABLE_SIZE;
      if (ng > 0) {
        CUDA_TRY(cudaMemcpy(sl.h_groups, sl.d_groups,
                            (size_t)ng * sizeof(warpjq_group),
                            cudaMemcpyDeviceToHost),
                 "D2H(groups)");
      }
      out->groups = sl.h_groups;
      out->n_groups = ng;
      unsigned int ov = 0;
      CUDA_TRY(cudaMemcpy(&ov, sl.table.overflow, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost),
               "D2H(group overflow)");
      out->group_overflow = ov;
    }
    return WARPJQ_OK;
  }

  long long n_sel = *sl.h_n_sel;
  if (n_sel < 0) n_sel = 0;
  out->n_selected = (uint64_t)n_sel;
  if (n_sel == 0) {
    out->out_bytes = sl.h_out;
    out->out_len = 0;
    out->out_line_idx = sl.h_sel_idx;
    out->out_row_off = sl.h_row_off;
    return WARPJQ_OK;
  }

  CUDA_TRY(cudaMemcpy(sl.h_row_off, sl.d_row_off,
                      (size_t)(n_sel + 1) * sizeof(uint64_t),
                      cudaMemcpyDeviceToHost),
           "D2H(row offsets)");
  uint64_t total = sl.h_row_off[n_sel];
  if (total > ctx->out_cap) {
    // Belt and braces: k_emit already refuses to store a row that would not
    // fit, so reaching here means the device state is not what we expect --
    // under memory pressure from another process sharing the card, for
    // instance. Degrade to "redo this chunk on the CPU" rather than failing
    // the run: the answer stays correct either way, and a tool that dies
    // because something else was using the GPU is not much use.
    out->chunk_overflow = 1;
    out->n_selected = 0;
    out->out_len = 0;
    return WARPJQ_OK;
  }
  CUDA_TRY(cudaMemcpy(sl.h_sel_idx, sl.d_sel_idx,
                      (size_t)n_sel * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost),
           "D2H(selected line indices)");
  CUDA_TRY(cudaMemcpy(sl.h_out, sl.d_out, (size_t)total, cudaMemcpyDeviceToHost),
           "D2H(rows)");

  out->out_bytes = sl.h_out;
  out->out_len = total;
  out->out_line_idx = sl.h_sel_idx;
  out->out_row_off = sl.h_row_off;
  return WARPJQ_OK;
}

}  // extern "C"
