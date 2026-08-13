//! A validating JSON scanner that returns **raw input slices**, never a DOM.
//!
//! Why not simdjson? A DOM forces a re-serialisation on the way out, and that
//! reformats numbers: `1.0` becomes `1`, `1e3` becomes `1000`, and integers
//! past 2^53 lose digits. jq 1.7 preserves the original literal when a number
//! passes through untouched, so a DOM round-trip would make warpjq disagree
//! with jq on exactly the inputs people notice. Handing back a slice of the
//! input sidesteps the whole class of bug, and it is what the GPU kernel
//! does too, which is what makes the two backends byte-comparable.

use std::cmp::Ordering;

use crate::query::{CmpOp, Literal, Step};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The path did not resolve. Renders as `null`, as in jq.
    Missing,
    Null,
    Bool,
    Num,
    Str,
    Arr,
    Obj,
}

impl Kind {
    /// jq's total order across types: null < false < true < numbers < strings
    /// < arrays < objects. `Missing` sorts as `null`.
    fn rank(self, raw: &[u8]) -> u8 {
        match self {
            Kind::Missing | Kind::Null => 0,
            Kind::Bool => {
                if raw == b"true" {
                    2
                } else {
                    1
                }
            }
            Kind::Num => 3,
            Kind::Str => 4,
            Kind::Arr => 5,
            Kind::Obj => 6,
        }
    }
}

/// An extracted value: a type tag plus the exact bytes it occupied in the
/// input. For `Str`, `raw` **includes** the surrounding quotes.
#[derive(Copy, Clone, Debug)]
pub struct Slot<'a> {
    pub kind: Kind,
    pub raw: &'a [u8],
}

pub const MISSING: Slot<'static> = Slot {
    kind: Kind::Missing,
    raw: b"null",
};

impl<'a> Slot<'a> {
    /// jq truthiness: everything except `null` and `false` is truthy.
    pub fn is_truthy(&self) -> bool {
        !matches!(self.kind, Kind::Missing | Kind::Null) && self.raw != b"false"
    }

    /// The bytes between the quotes, for `Str`. Still escaped.
    pub fn str_inner(&self) -> &'a [u8] {
        debug_assert_eq!(self.kind, Kind::Str);
        &self.raw[1..self.raw.len() - 1]
    }

    pub fn as_f64(&self) -> Option<f64> {
        if self.kind != Kind::Num {
            return None;
        }
        std::str::from_utf8(self.raw).ok()?.parse().ok()
    }

    /// What this value serialises to in output position. Identical to `raw`
    /// except that a missing path prints as `null`.
    pub fn out_bytes(&self) -> &[u8] {
        self.raw
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    pub offset: usize,
    pub detail: String,
}

impl JsonError {
    fn at(offset: usize, detail: impl Into<String>) -> Self {
        JsonError {
            offset,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at byte {}", self.detail, self.offset)
    }
}

/// The outcome of resolving a path against a line.
#[derive(Debug)]
pub enum Lookup<'a> {
    Found(Slot<'a>),
    /// jq would raise "Cannot index X with Y" here. We surface it so the
    /// caller can choose to skip the line or abort.
    TypeError(String),
    Invalid(JsonError),
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

#[inline]
fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

/// Which container each open nesting level is, for the iterative scanner.
///
/// The first 64 levels live in a single word, so the overwhelmingly common
/// case costs no allocation at all; anything deeper spills to a `Vec` that is
/// only ever touched by input that genuinely nests that far.
struct Depth {
    /// Bit `d` set means the container at level `d` is an array.
    bits: u64,
    /// Levels 64 and beyond.
    deep: Vec<bool>,
    len: usize,
}

impl Depth {
    #[inline]
    fn new() -> Depth {
        Depth {
            bits: 0,
            deep: Vec::new(),
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, is_array: bool) {
        if self.len < 64 {
            if is_array {
                self.bits |= 1u64 << self.len;
            } else {
                self.bits &= !(1u64 << self.len);
            }
        } else {
            self.deep.push(is_array);
        }
        self.len += 1;
    }

    #[inline]
    fn pop(&mut self) {
        self.len -= 1;
        if self.len >= 64 {
            self.deep.pop();
        }
    }

    #[inline]
    fn top_is_array(&self) -> bool {
        let d = self.len - 1;
        if d < 64 {
            (self.bits >> d) & 1 == 1
        } else {
            self.deep[d - 64]
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Copy, Clone)]
enum ScanState {
    Value,
    Key,
    AfterValue,
}

/// Advances past one complete JSON value starting at `i`, validating as it
/// goes. Returns the index one past the value.
///
/// **Iterative on purpose.** The natural formulation here is mutual recursion
/// between value/object/array, and that is what this used to be. A single line
/// of `[[[[...` a few tens of kilobytes long, well inside the default
/// `--max-line-bytes`, then overflows the thread stack, which is an
/// unrecoverable abort rather than an error, and `panic = "abort"` in the
/// release profile removes even the possibility of catching it. That made any
/// pipeline feeding warpjq untrusted NDJSON trivially killable, and it took the
/// GPU path down with it: the kernel correctly declines anything past its
/// 64-level stack and hands the line to *this* function.
///
/// Nesting is bounded only by the length of the line, since every level costs
/// at least one byte, so there is no depth limit here beyond that.
pub fn skip_value(b: &[u8], start: usize) -> Result<usize, JsonError> {
    let mut i = start;
    let mut state = ScanState::Value;
    let mut depth = Depth::new();

    loop {
        match state {
            ScanState::Value => {
                i = skip_ws(b, i);
                let c = *b
                    .get(i)
                    .ok_or_else(|| JsonError::at(i, "unexpected end of input"))?;
                match c {
                    b'{' => {
                        i = skip_ws(b, i + 1);
                        if b.get(i) == Some(&b'}') {
                            i += 1;
                            state = ScanState::AfterValue;
                        } else {
                            depth.push(false);
                            state = ScanState::Key;
                        }
                    }
                    b'[' => {
                        i = skip_ws(b, i + 1);
                        if b.get(i) == Some(&b']') {
                            i += 1;
                            state = ScanState::AfterValue;
                        } else {
                            depth.push(true);
                            state = ScanState::Value;
                        }
                    }
                    b'"' => {
                        i = skip_string(b, i)?;
                        state = ScanState::AfterValue;
                    }
                    b't' => {
                        i = expect_lit(b, i, b"true")?;
                        state = ScanState::AfterValue;
                    }
                    b'f' => {
                        i = expect_lit(b, i, b"false")?;
                        state = ScanState::AfterValue;
                    }
                    b'n' => {
                        i = expect_lit(b, i, b"null")?;
                        state = ScanState::AfterValue;
                    }
                    b'-' | b'0'..=b'9' => {
                        i = skip_number(b, i)?;
                        state = ScanState::AfterValue;
                    }
                    other => {
                        return Err(JsonError::at(
                            i,
                            format!("unexpected character `{}`", other as char),
                        ))
                    }
                }
            }
            ScanState::Key => {
                i = skip_ws(b, i);
                if b.get(i) != Some(&b'"') {
                    return Err(JsonError::at(i, "expected a string key"));
                }
                i = skip_string(b, i)?;
                i = skip_ws(b, i);
                if b.get(i) != Some(&b':') {
                    return Err(JsonError::at(i, "expected `:` after object key"));
                }
                i += 1;
                state = ScanState::Value;
            }
            ScanState::AfterValue => {
                if depth.is_empty() {
                    return Ok(i);
                }
                i = skip_ws(b, i);
                let c = *b
                    .get(i)
                    .ok_or_else(|| JsonError::at(i, "unexpected end of input"))?;
                let arr = depth.top_is_array();
                match (c, arr) {
                    (b',', true) => {
                        i += 1;
                        state = ScanState::Value;
                    }
                    (b',', false) => {
                        i += 1;
                        state = ScanState::Key;
                    }
                    (b'}', false) | (b']', true) => {
                        i += 1;
                        depth.pop();
                    }
                    _ => {
                        return Err(JsonError::at(
                            i,
                            if arr {
                                "expected `,` or `]`"
                            } else {
                                "expected `,` or `}`"
                            },
                        ))
                    }
                }
            }
        }
    }
}

fn expect_lit(b: &[u8], i: usize, lit: &[u8]) -> Result<usize, JsonError> {
    if b.len() >= i + lit.len() && &b[i..i + lit.len()] == lit {
        Ok(i + lit.len())
    } else {
        Err(JsonError::at(
            i,
            format!("expected `{}`", std::str::from_utf8(lit).unwrap()),
        ))
    }
}

pub fn skip_string(b: &[u8], i: usize) -> Result<usize, JsonError> {
    debug_assert_eq!(b[i], b'"');
    let mut j = i + 1;
    while j < b.len() {
        match b[j] {
            b'"' => return Ok(j + 1),
            b'\\' => {
                let e = *b
                    .get(j + 1)
                    .ok_or_else(|| JsonError::at(j, "unterminated escape"))?;
                match e {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => j += 2,
                    b'u' => {
                        if j + 6 > b.len() || !b[j + 2..j + 6].iter().all(u8::is_ascii_hexdigit) {
                            return Err(JsonError::at(j, "malformed `\\u` escape"));
                        }
                        j += 6;
                    }
                    other => {
                        return Err(JsonError::at(
                            j,
                            format!("invalid escape `\\{}`", other as char),
                        ))
                    }
                }
            }
            // Raw control characters are illegal inside a JSON string.
            c if c < 0x20 => return Err(JsonError::at(j, "unescaped control character in string")),
            _ => j += 1,
        }
    }
    Err(JsonError::at(i, "unterminated string"))
}

fn skip_number(b: &[u8], i: usize) -> Result<usize, JsonError> {
    let mut j = i;
    if b[j] == b'-' {
        j += 1;
    }
    let int_start = j;
    while j < b.len() && b[j].is_ascii_digit() {
        j += 1;
    }
    if j == int_start {
        return Err(JsonError::at(i, "number with no digits"));
    }
    // JSON forbids leading zeros: `01` is invalid, `0` and `0.5` are fine.
    if b[int_start] == b'0' && j - int_start > 1 {
        return Err(JsonError::at(i, "number has a leading zero"));
    }
    if j < b.len() && b[j] == b'.' {
        j += 1;
        let frac_start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j == frac_start {
            return Err(JsonError::at(j, "number has no digits after `.`"));
        }
    }
    if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
        j += 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let exp_start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j == exp_start {
            return Err(JsonError::at(j, "number has no digits in its exponent"));
        }
    }
    Ok(j)
}

/// Validates that `line` is exactly one JSON value with nothing but
/// whitespace after it.
pub fn validate(line: &[u8]) -> Result<(), JsonError> {
    let end = skip_value(line, 0)?;
    let rest = skip_ws(line, end);
    if rest != line.len() {
        return Err(JsonError::at(rest, "trailing content after JSON value"));
    }
    Ok(())
}

fn kind_of(b: &[u8], i: usize) -> Kind {
    match b[i] {
        b'{' => Kind::Obj,
        b'[' => Kind::Arr,
        b'"' => Kind::Str,
        b't' | b'f' => Kind::Bool,
        b'n' => Kind::Null,
        _ => Kind::Num,
    }
}

/// Resolves `steps` against the JSON value in `line`.
///
/// Semantics follow jq: a missing key or an out-of-range index yields
/// `Missing` (which prints as `null`) and indexing `null` keeps yielding
/// `Missing`, but indexing a scalar is a type error.
///
/// This does **not** validate the whole line. It descends only as far as the
/// path requires. Callers that need "is this line valid JSON at all" call
/// [`validate`] once per line and then run their lookups; splitting the two
/// keeps a five-path projection from re-validating the line five times.
pub fn lookup<'a>(line: &'a [u8], steps: &[Step]) -> Lookup<'a> {
    let mut start = skip_ws(line, 0);
    if start >= line.len() {
        return Lookup::Invalid(JsonError::at(0, "empty value"));
    }
    let mut kind = kind_of(line, start);
    // Only the identity path needs the full extent up front; every other path
    // gets its extent from the descent below.
    let mut end = if steps.is_empty() {
        match skip_value(line, start) {
            Ok(e) => e,
            Err(e) => return Lookup::Invalid(e),
        }
    } else {
        line.len()
    };

    for step in steps {
        // Indexing null/missing yields null, and keeps doing so.
        if matches!(kind, Kind::Missing | Kind::Null) {
            return Lookup::Found(MISSING);
        }
        match step {
            Step::Key(key) => {
                if kind != Kind::Obj {
                    return Lookup::TypeError(format!(
                        "cannot index {} with \"{key}\"",
                        type_name(kind)
                    ));
                }
                match object_get(line, start, key.as_bytes()) {
                    Ok(Some((s, e))) => {
                        start = s;
                        end = e;
                        kind = kind_of(line, s);
                    }
                    Ok(None) => return Lookup::Found(MISSING),
                    Err(e) => return Lookup::Invalid(e),
                }
            }
            Step::Index(idx) => {
                if kind != Kind::Arr {
                    return Lookup::TypeError(format!(
                        "cannot index {} with number",
                        type_name(kind)
                    ));
                }
                match array_get(line, start, *idx) {
                    Ok(Some((s, e))) => {
                        start = s;
                        end = e;
                        kind = kind_of(line, s);
                    }
                    Ok(None) => return Lookup::Found(MISSING),
                    Err(e) => return Lookup::Invalid(e),
                }
            }
        }
    }

    Lookup::Found(Slot {
        kind,
        raw: &line[start..end],
    })
}

fn type_name(k: Kind) -> &'static str {
    match k {
        Kind::Missing | Kind::Null => "null",
        Kind::Bool => "boolean",
        Kind::Num => "number",
        Kind::Str => "string",
        Kind::Arr => "array",
        Kind::Obj => "object",
    }
}

/// Finds `key` in the object starting at `i`. Last duplicate key wins, which
/// is what jq does.
fn object_get(b: &[u8], i: usize, key: &[u8]) -> Result<Option<(usize, usize)>, JsonError> {
    let mut j = skip_ws(b, i + 1);
    if b.get(j) == Some(&b'}') {
        return Ok(None);
    }
    let mut found = None;
    loop {
        j = skip_ws(b, j);
        let key_start = j;
        j = skip_string(b, j)?;
        let key_raw = &b[key_start + 1..j - 1];
        j = skip_ws(b, j);
        if b.get(j) != Some(&b':') {
            return Err(JsonError::at(j, "expected `:` after object key"));
        }
        let val_start = skip_ws(b, j + 1);
        let val_end = skip_value(b, val_start)?;
        if key_matches(key_raw, key) {
            found = Some((val_start, val_end));
        }
        j = skip_ws(b, val_end);
        match b.get(j) {
            Some(b',') => j += 1,
            Some(b'}') => return Ok(found),
            _ => return Err(JsonError::at(j, "expected `,` or `}`")),
        }
    }
}

fn array_get(b: &[u8], i: usize, idx: u32) -> Result<Option<(usize, usize)>, JsonError> {
    let mut j = skip_ws(b, i + 1);
    if b.get(j) == Some(&b']') {
        return Ok(None);
    }
    let mut n = 0u32;
    loop {
        let start = skip_ws(b, j);
        let end = skip_value(b, start)?;
        if n == idx {
            return Ok(Some((start, end)));
        }
        n += 1;
        j = skip_ws(b, end);
        match b.get(j) {
            Some(b',') => j += 1,
            Some(b']') => return Ok(None),
            _ => return Err(JsonError::at(j, "expected `,` or `]`")),
        }
    }
}

/// Compares a possibly-escaped JSON string body against a decoded key.
/// The common case (no backslash) is a plain `memcmp`.
pub fn key_matches(raw: &[u8], decoded: &[u8]) -> bool {
    if !raw.contains(&b'\\') {
        return raw == decoded;
    }
    let mut buf = Vec::with_capacity(raw.len());
    if unescape_into(raw, &mut buf).is_err() {
        return false;
    }
    buf == decoded
}

/// Decodes JSON string escapes into `out`.
pub fn unescape_into(raw: &[u8], out: &mut Vec<u8>) -> Result<(), JsonError> {
    let mut i = 0;
    while i < raw.len() {
        if raw[i] != b'\\' {
            out.push(raw[i]);
            i += 1;
            continue;
        }
        let e = *raw
            .get(i + 1)
            .ok_or_else(|| JsonError::at(i, "unterminated escape"))?;
        match e {
            b'"' => out.push(b'"'),
            b'\\' => out.push(b'\\'),
            b'/' => out.push(b'/'),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'u' => {
                let hex = std::str::from_utf8(
                    raw.get(i + 2..i + 6)
                        .ok_or_else(|| JsonError::at(i, "truncated `\\u`"))?,
                )
                .map_err(|_| JsonError::at(i, "bad `\\u`"))?;
                let mut cp =
                    u32::from_str_radix(hex, 16).map_err(|_| JsonError::at(i, "bad `\\u`"))?;
                i += 6;
                if (0xD800..0xDC00).contains(&cp)
                    && raw.get(i) == Some(&b'\\')
                    && raw.get(i + 1) == Some(&b'u')
                {
                    if let Some(lo_hex) = raw
                        .get(i + 2..i + 6)
                        .and_then(|h| std::str::from_utf8(h).ok())
                    {
                        if let Ok(lo) = u32::from_str_radix(lo_hex, 16) {
                            if (0xDC00..0xE000).contains(&lo) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                i += 6;
                            }
                        }
                    }
                }
                let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
                let mut tmp = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                continue;
            }
            other => {
                return Err(JsonError::at(
                    i,
                    format!("invalid escape `\\{}`", other as char),
                ))
            }
        }
        i += 2;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Three-way compare of an extracted value against a query literal, using
/// jq's cross-type ordering.
pub fn compare(slot: &Slot<'_>, lit: &Literal) -> Ordering {
    let lrank = match lit {
        Literal::Null => 0,
        Literal::Bool(false) => 1,
        Literal::Bool(true) => 2,
        Literal::Num(_) => 3,
        Literal::Str(_) => 4,
    };
    let srank = slot.kind.rank(slot.raw);
    if srank != lrank {
        return srank.cmp(&lrank);
    }
    match lit {
        Literal::Null | Literal::Bool(_) => Ordering::Equal,
        Literal::Num(n) => slot
            .as_f64()
            .map(|v| v.partial_cmp(n).unwrap_or(Ordering::Equal))
            .unwrap_or(Ordering::Equal),
        Literal::Str(s) => {
            let inner = slot.str_inner();
            if !inner.contains(&b'\\') {
                inner.cmp(s.as_bytes())
            } else {
                let mut buf = Vec::with_capacity(inner.len());
                if unescape_into(inner, &mut buf).is_err() {
                    return Ordering::Equal;
                }
                buf.as_slice().cmp(s.as_bytes())
            }
        }
    }
}

pub fn eval_cmp(slot: &Slot<'_>, op: CmpOp, lit: &Literal) -> bool {
    op.accepts(compare(slot, lit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::Step;

    fn key(k: &str) -> Step {
        Step::Key(k.into())
    }

    fn get<'a>(line: &'a str, steps: &[Step]) -> Slot<'a> {
        match lookup(line.as_bytes(), steps) {
            Lookup::Found(s) => s,
            other => panic!("{line}: {other:?}"),
        }
    }

    #[test]
    fn extracts_top_level_fields() {
        let s = get(r#"{"a":1,"b":"x"}"#, &[key("a")]);
        assert_eq!(s.kind, Kind::Num);
        assert_eq!(s.raw, b"1");
    }

    #[test]
    fn preserves_the_original_number_spelling() {
        // A DOM round-trip would turn these into 1, 1000 and a rounded float.
        for (json, expect) in [
            (r#"{"a":1.0}"#, "1.0"),
            (r#"{"a":1e3}"#, "1e3"),
            (r#"{"a":-0.0}"#, "-0.0"),
            (
                r#"{"a":123456789012345678901234567890}"#,
                "123456789012345678901234567890",
            ),
        ] {
            assert_eq!(get(json, &[key("a")]).raw, expect.as_bytes(), "{json}");
        }
    }

    #[test]
    fn extracts_nested_and_indexed() {
        let line = r#"{"r":{"p":["a","b",{"deep":7}]}}"#;
        let s = get(line, &[key("r"), key("p"), Step::Index(2), key("deep")]);
        assert_eq!(s.raw, b"7");
    }

    #[test]
    fn missing_key_is_null_not_an_error() {
        assert_eq!(get(r#"{"a":1}"#, &[key("z")]).kind, Kind::Missing);
        // Indexing through a missing key keeps yielding null, as in jq.
        assert_eq!(get(r#"{"a":1}"#, &[key("z"), key("y")]).kind, Kind::Missing);
    }

    #[test]
    fn out_of_range_index_is_null() {
        assert_eq!(
            get(r#"{"a":[1]}"#, &[key("a"), Step::Index(9)]).kind,
            Kind::Missing
        );
    }

    #[test]
    fn indexing_a_scalar_is_a_type_error() {
        assert!(matches!(
            lookup(br#"{"a":1}"#, &[key("a"), key("b")]),
            Lookup::TypeError(_)
        ));
        assert!(matches!(
            lookup(br#"{"a":{}}"#, &[key("a"), Step::Index(0)]),
            Lookup::TypeError(_)
        ));
    }

    #[test]
    fn last_duplicate_key_wins() {
        assert_eq!(get(r#"{"a":1,"a":2}"#, &[key("a")]).raw, b"2");
    }

    #[test]
    fn handles_escaped_and_unicode_keys() {
        assert_eq!(get(r#"{"a\"b":5}"#, &[key("a\"b")]).raw, b"5");
        assert_eq!(get(r#"{"é":5}"#, &[key("é")]).raw, b"5");
        assert_eq!(get(r#"{"é":5}"#, &[key("é")]).raw, b"5");
    }

    #[test]
    fn braces_inside_strings_do_not_confuse_the_scanner() {
        let line = r#"{"msg":"}{[],:\"","x":9}"#;
        assert_eq!(get(line, &[key("x")]).raw, b"9");
    }

    #[test]
    fn nested_objects_are_returned_whole() {
        let s = get(r#"{"a":{"b":[1,2]},"c":0}"#, &[key("a")]);
        assert_eq!(s.kind, Kind::Obj);
        assert_eq!(s.raw, br#"{"b":[1,2]}"#);
    }

    #[test]
    fn rejects_malformed_input() {
        for bad in [
            r#"{"a":}"#,
            r#"{"a" 1}"#,
            r#"{a:1}"#,
            r#"{"a":01}"#,
            r#"{"a":1.}"#,
            r#"{"a":1e}"#,
            r#"{"a":"x}"#,
            r#"{"a":tru}"#,
            r#"{"a":1}x"#,
            "",
            r#"{"a":"x
y"}"#,
        ] {
            assert!(validate(bad.as_bytes()).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn deep_nesting_does_not_overflow_the_stack() {
        // Regression: this used to be mutually recursive and aborted the
        // process (an abort, not an error) on input a fraction of the
        // default --max-line-bytes. Both a valid deep document and a
        // truncated one must return normally.
        for n in [1_000usize, 100_000, 1_000_000] {
            let mut v = vec![b'['; n];
            v.extend(std::iter::repeat(b']').take(n));
            assert!(validate(&v).is_ok(), "depth {n} should be valid");

            // Unterminated: must be a clean error, not a crash.
            let open_only = vec![b'['; n];
            assert!(validate(&open_only).is_err());

            // Objects nest through a different state in the machine.
            let mut o = Vec::new();
            for _ in 0..n {
                o.extend_from_slice(br#"{"a":"#);
            }
            o.push(b'1');
            o.extend(std::iter::repeat(b'}').take(n));
            assert!(validate(&o).is_ok(), "object depth {n} should be valid");
        }
    }

    #[test]
    fn lookup_through_deep_nesting_terminates() {
        // The path walk calls skip_value once per sibling, so it must stay
        // iterative too.
        let n = 200_000;
        let mut v = Vec::from(*br#"{"a":"#);
        v.extend(std::iter::repeat(b'[').take(n));
        v.extend(std::iter::repeat(b']').take(n));
        v.extend_from_slice(br#","b":7}"#);
        assert_eq!(get(std::str::from_utf8(&v).unwrap(), &[key("b")]).raw, b"7");
    }

    #[test]
    fn mismatched_brackets_are_rejected_at_every_depth() {
        assert!(validate(b"[[[[]]]}").is_err());
        assert!(validate(br#"{"a":[}"#).is_err());
        assert!(validate(br#"{"a":]}"#).is_err());
        assert!(validate(b"[}").is_err());
        assert!(validate(br#"{"a":1]"#).is_err());
    }

    #[test]
    fn accepts_awkward_but_valid_input() {
        for good in [
            r#"{}"#,
            r#"[]"#,
            r#"  {"a" : [ 1 , 2 ] }  "#,
            r#""just a string""#,
            r#"null"#,
            r#"-0"#,
            r#"{"a":"A😀"}"#,
        ] {
            assert!(validate(good.as_bytes()).is_ok(), "should accept: {good:?}");
        }
    }

    #[test]
    fn truthiness_matches_jq() {
        assert!(!get(r#"{"a":null}"#, &[key("a")]).is_truthy());
        assert!(!get(r#"{"a":false}"#, &[key("a")]).is_truthy());
        assert!(!get(r#"{"a":1}"#, &[key("z")]).is_truthy());
        assert!(get(r#"{"a":0}"#, &[key("a")]).is_truthy());
        assert!(get(r#"{"a":""}"#, &[key("a")]).is_truthy());
        assert!(get(r#"{"a":[]}"#, &[key("a")]).is_truthy());
    }

    #[test]
    fn compares_across_types_in_jq_order() {
        let num = get(r#"{"a":5}"#, &[key("a")]);
        let s = get(r#"{"a":"5"}"#, &[key("a")]);
        // numbers sort before strings
        assert_eq!(compare(&num, &Literal::Str("5".into())), Ordering::Less);
        assert_eq!(compare(&s, &Literal::Num(5.0)), Ordering::Greater);
        assert_eq!(compare(&num, &Literal::Num(5.0)), Ordering::Equal);
        // null is the bottom of the order
        let n = get(r#"{"a":null}"#, &[key("a")]);
        assert_eq!(compare(&n, &Literal::Num(0.0)), Ordering::Less);
        assert_eq!(compare(&n, &Literal::Null), Ordering::Equal);
    }

    #[test]
    fn compares_escaped_strings_by_decoded_value() {
        let s = get(r#"{"a":"café"}"#, &[key("a")]);
        assert!(eval_cmp(&s, CmpOp::Eq, &Literal::Str("café".into())));
    }

    #[test]
    fn number_comparison_is_numeric_not_lexical() {
        let s = get(r#"{"a":9}"#, &[key("a")]);
        assert!(eval_cmp(&s, CmpOp::Lt, &Literal::Num(10.0)));
        let s = get(r#"{"a":1e3}"#, &[key("a")]);
        assert!(eval_cmp(&s, CmpOp::Eq, &Literal::Num(1000.0)));
    }
}
