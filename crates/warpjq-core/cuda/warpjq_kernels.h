/* warpjq device ABI.
 *
 * Everything crossing this boundary is plain C with fixed-width types and no
 * ownership transfer: the Rust side allocates nothing here except through
 * warpjq_ctx_create/destroy, and every pointer handed in is borrowed for the
 * duration of the call.
 *
 * The struct layouts are mirrored by #[repr(C)] types in src/query/mod.rs and
 * src/gpu/ffi.rs. There is a startup assertion (warpjq_abi_check) that
 * compares sizeof() on both sides, because a silently mismatched struct here
 * produces plausible-looking wrong answers rather than a crash.
 */
#ifndef WARPJQ_KERNELS_H
#define WARPJQ_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- status codes ---------------------------------------------------- */

typedef enum {
  WARPJQ_OK = 0,
  WARPJQ_ERR_CUDA = 1,
  WARPJQ_ERR_NO_DEVICE = 2,
  WARPJQ_ERR_OOM = 3,
  WARPJQ_ERR_ABI = 4,
  WARPJQ_ERR_INVALID_ARG = 5
} warpjq_status;

/* Per-line outcome, written by the evaluation kernel. */
typedef enum {
  WARPJQ_LINE_OK = 0,
  WARPJQ_LINE_INVALID = 1,   /* malformed JSON */
  WARPJQ_LINE_TYPE_ERROR = 2,/* e.g. .a.b where .a is a number */
  WARPJQ_LINE_FALLBACK = 3,  /* kernel cannot decide this line exactly */
  WARPJQ_LINE_BLANK = 4      /* empty or whitespace-only */
} warpjq_line_status;

/* ---- program tables (mirror of query::FlatProgram) -------------------- */

enum { WARPJQ_STEP_KEY = 0, WARPJQ_STEP_INDEX = 1 };

enum {
  WARPJQ_LIT_NULL = 0,
  WARPJQ_LIT_FALSE = 1,
  WARPJQ_LIT_TRUE = 2,
  WARPJQ_LIT_NUM = 3,
  WARPJQ_LIT_STR = 4
};

enum {
  WARPJQ_COND_CMP = 0,
  WARPJQ_COND_TRUTHY = 1,
  WARPJQ_COND_AND = 2,
  WARPJQ_COND_OR = 3,
  WARPJQ_COND_NOT = 4
};

enum {
  WARPJQ_OP_EQ = 0,
  WARPJQ_OP_NE = 1,
  WARPJQ_OP_LT = 2,
  WARPJQ_OP_LE = 3,
  WARPJQ_OP_GT = 4,
  WARPJQ_OP_GE = 5
};

/* What to produce for each surviving line. */
enum {
  WARPJQ_OUT_PASSTHROUGH = 0,
  WARPJQ_OUT_PATH = 1,
  WARPJQ_OUT_PROJECT = 2,
  WARPJQ_OUT_AGG = 3
};

enum {
  WARPJQ_AGG_COUNT = 0,
  WARPJQ_AGG_SUM = 1,
  WARPJQ_AGG_MIN = 2,
  WARPJQ_AGG_MAX = 3,
  WARPJQ_AGG_AVG = 4
};

typedef struct {
  uint32_t kind;     /* WARPJQ_STEP_* */
  uint32_t index;    /* array index, for WARPJQ_STEP_INDEX */
  uint32_t key_off;  /* into blob */
  uint32_t key_len;
  uint32_t key_hash; /* FNV-1a 32 of the decoded key */
} warpjq_step;

typedef struct {
  uint32_t step_off;
  uint32_t step_count;
} warpjq_path;

typedef struct {
  uint32_t path;
  uint32_t op;       /* WARPJQ_OP_* */
  uint32_t lit_kind; /* WARPJQ_LIT_* */
  uint32_t lit_off;  /* into blob, for WARPJQ_LIT_STR */
  uint32_t lit_len;
  uint32_t _pad;
  double lit_num;
} warpjq_cmp;

typedef struct {
  uint32_t op;  /* WARPJQ_COND_* */
  uint32_t arg; /* cmp index, or path id for TRUTHY */
} warpjq_cond_op;

/* The whole compiled query, as flat arrays. Uploaded once per run. */
typedef struct {
  const warpjq_step *steps;
  uint32_t n_steps;
  const warpjq_path *paths;
  uint32_t n_paths;
  const warpjq_cmp *cmps;
  uint32_t n_cmps;
  const warpjq_cond_op *cond_rpn;
  uint32_t n_cond;
  const uint8_t *blob;
  uint32_t n_blob;

  /* Which paths the kernel must resolve, and where each lands in the slot
   * table. slot_of_path[path_id] is the slot index, or 0xFFFFFFFF. */
  const uint32_t *needed_paths;
  uint32_t n_needed;
  const uint32_t *slot_of_path;

  uint32_t output_kind; /* WARPJQ_OUT_* */
  uint32_t output_path; /* for WARPJQ_OUT_PATH */
  uint32_t agg_kind;    /* WARPJQ_AGG_* */
  uint32_t agg_path;    /* 0xFFFFFFFF for count */
  uint32_t group_path;  /* 0xFFFFFFFF when ungrouped */
  uint32_t has_filter;

  /* Projection: n_fields values, each preceded by prefix[i] literal bytes
   * (e.g. `{"a":` then `,"b":`) and followed by the suffix. Precomputed on the
   * host so the kernel never formats a key name. */
  uint32_t n_fields;
  const uint32_t *field_paths;
  const uint32_t *prefix_off;
  const uint32_t *prefix_len;
  uint32_t suffix_off;
  uint32_t suffix_len;
  /* 1 when the projection must be rendered as CSV cells rather than JSON. */
  uint32_t csv_mode;
} warpjq_program;

/* ---- results --------------------------------------------------------- */

/* Per-block aggregate partials, merged on the host so that the final numbers
 * do not depend on how the input happened to be chunked. */
typedef struct {
  uint64_t count;
  uint64_t numeric;
  double sum;
  double min;
  double max;
  uint32_t saw_non_numeric;
  uint32_t _pad;
} warpjq_agg_partial;

/* One occupied slot of the device group table. */
typedef struct {
  uint32_t key_off; /* offset into the chunk, not the blob */
  uint32_t key_len;
  uint32_t key_kind; /* json kind of the group key */
  uint32_t _pad;
  uint64_t count;
  uint64_t numeric;
  double sum;
  double min;
  double max;
} warpjq_group;

typedef struct {
  uint64_t n_lines;
  uint64_t n_blank;
  uint64_t n_invalid;
  uint64_t n_type_error;
  uint64_t n_fallback;
  uint64_t n_selected;

  /* Assembled output rows, in input order. Valid until the next submit on
   * this slot. NULL for aggregate queries. */
  const uint8_t *out_bytes;
  uint64_t out_len;
  /* Line index of each emitted row, ascending. Lets the host interleave rows
   * it computed itself for fallback lines without reordering anything. */
  const uint32_t *out_line_idx;
  /* Byte offset of each row within out_bytes; n_selected + 1 entries. */
  const uint64_t *out_row_off;

  /* Lines the kernel declined, with their byte ranges so the host can re-run
   * just those on the CPU and splice the rows back into place. Three parallel
   * arrays, n_fallback long.
   *
   * NOT sorted. The kernel appends with an atomicAdd, so these arrive in
   * whatever order the blocks retired, and the consumer sorts. Enforcing it
   * here as well would put one invariant in two places with neither site
   * aware of the other, which is how both eventually get removed. */
  const uint32_t *fallback_idx;
  const uint32_t *fallback_off;
  const uint32_t *fallback_len;
  /* Set when there were more declined lines than the fallback arrays hold,
   * more lines in the chunk than the buffers were sized for, or assembled
   * output larger than the reserved buffer. In every case the kernels declined
   * to write rather than overrunning, and the host redoes the whole chunk on
   * the CPU. */
  uint32_t chunk_overflow;
  /* Index within this chunk of the first line that failed to parse, or
   * 0xFFFFFFFF when there was none. Lets --strict name the offending line. */
  uint32_t first_invalid;

  /* Ungrouped aggregate result, already reduced across the chunk. */
  warpjq_agg_partial agg;

  /* Grouped aggregate results. */
  const warpjq_group *groups;
  uint32_t n_groups;
  /* Set when the device hash table overflowed; the host must redo the chunk
   * on the CPU. Never silently wrong. */
  uint32_t group_overflow;
} warpjq_chunk_result;

/* ---- lifecycle ------------------------------------------------------- */

typedef struct warpjq_ctx warpjq_ctx;

/* Fails if no CUDA device is usable. Cheap enough to call for `--backend gpu`
 * validation before opening the input. */
warpjq_status warpjq_probe(char *err, size_t err_len);

/* Confirms the Rust and C++ views of the ABI structs agree. */
warpjq_status warpjq_abi_check(const uint64_t *sizes, size_t n);

warpjq_status warpjq_ctx_create(const warpjq_program *prog,
                                uint64_t max_chunk_bytes, uint32_t n_slots,
                                warpjq_ctx **out, char *err, size_t err_len);

void warpjq_ctx_destroy(warpjq_ctx *ctx);

/* Pinned staging buffer for `slot`. The host fills this, then submits. */
uint8_t *warpjq_slot_buffer(warpjq_ctx *ctx, uint32_t slot);
uint64_t warpjq_slot_capacity(const warpjq_ctx *ctx);

/* Queues H2D + kernels for `slot`. Returns immediately; the caller is free to
 * fill another slot while this one runs. That overlap is the entire point of
 * the design: PCIe transfer, not the kernels, is the budget. */
warpjq_status warpjq_submit(warpjq_ctx *ctx, uint32_t slot, uint64_t n_bytes,
                            char *err, size_t err_len);

/* Blocks until `slot` is done and fills `out`. Pointers in `out` remain valid
 * until the next submit on the same slot. */
warpjq_status warpjq_wait(warpjq_ctx *ctx, uint32_t slot,
                          warpjq_chunk_result *out, char *err, size_t err_len);

/* Human-readable device name, for --stats and bench headers. */
warpjq_status warpjq_device_name(char *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* WARPJQ_KERNELS_H */
