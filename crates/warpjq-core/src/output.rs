//! Output formatting: NDJSON, CSV, and bare counts.
//!
//! Output order is always input order. Compaction on the GPU preserves it and
//! the CPU backend never reorders; nothing here is allowed to sort a stream.
//! Users grep and diff this output, so a run that shuffles rows, even
//! "equivalently", is a bug.

use std::io::{self, Write};

use crate::agg::{format_number, AggState, GroupKey};
use crate::json::{Kind, Slot};
use crate::query::{AggKind, Output, Program};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Format {
    #[default]
    Ndjson,
    Csv,
    /// Suppress rows, print only how many there were.
    CountOnly,
}

/// Appends `bytes` as a quoted, escaped JSON string.
pub fn write_json_string(out: &mut Vec<u8>, bytes: &[u8]) {
    out.push(b'"');
    for &c in bytes {
        match c {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0c => out.extend_from_slice(b"\\f"),
            c if c < 0x20 => {
                out.extend_from_slice(format!("\\u{:04x}", c).as_bytes());
            }
            c => out.push(c),
        }
    }
    out.push(b'"');
}

/// Appends one CSV cell, quoting only when RFC 4180 requires it.
pub fn write_csv_cell(out: &mut Vec<u8>, bytes: &[u8]) {
    let needs_quotes = bytes
        .iter()
        .any(|&c| c == b',' || c == b'"' || c == b'\n' || c == b'\r');
    if !needs_quotes {
        out.extend_from_slice(bytes);
        return;
    }
    out.push(b'"');
    for &c in bytes {
        if c == b'"' {
            out.push(b'"');
        }
        out.push(c);
    }
    out.push(b'"');
}

/// A CSV cell for an extracted value: strings unquoted and unescaped, `null`
/// and missing as an empty cell, everything else as its raw JSON text.
fn csv_value(out: &mut Vec<u8>, slot: &Slot<'_>) {
    match slot.kind {
        Kind::Missing | Kind::Null => {}
        Kind::Str => {
            let inner = slot.str_inner();
            if inner.contains(&b'\\') {
                let mut buf = Vec::with_capacity(inner.len());
                if crate::json::unescape_into(inner, &mut buf).is_ok() {
                    write_csv_cell(out, &buf);
                    return;
                }
            }
            write_csv_cell(out, inner);
        }
        _ => write_csv_cell(out, slot.raw),
    }
}

/// Buffers formatted rows and flushes them in large writes.
///
/// The buffer matters more than it looks: writing 10 million small lines to a
/// pipe one `write` at a time costs more than the query does.
pub struct Writer<W: Write> {
    inner: W,
    buf: Vec<u8>,
    format: Format,
    header_written: bool,
    rows: u64,
    flush_at: usize,
}

const DEFAULT_FLUSH_AT: usize = 1 << 20;

impl<W: Write> Writer<W> {
    pub fn new(inner: W, format: Format) -> Self {
        Writer {
            inner,
            buf: Vec::with_capacity(DEFAULT_FLUSH_AT + 64 * 1024),
            format,
            header_written: false,
            rows: 0,
            flush_at: DEFAULT_FLUSH_AT,
        }
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn format(&self) -> Format {
        self.format
    }

    fn maybe_flush(&mut self) -> io::Result<()> {
        if self.buf.len() >= self.flush_at {
            self.inner.write_all(&self.buf)?;
            self.buf.clear();
        }
        Ok(())
    }

    /// Flushes and hands back the sink plus the row count.
    ///
    /// Used by the parallel CPU backend, where each worker formats into its
    /// own `Vec<u8>` and the main thread concatenates them in input order.
    pub fn into_inner(mut self) -> io::Result<(W, u64)> {
        self.inner.write_all(&self.buf)?;
        self.buf.clear();
        Ok((self.inner, self.rows))
    }

    pub fn finish(mut self) -> io::Result<(W, u64)> {
        if self.format == Format::CountOnly {
            self.buf.clear();
            let mut b = itoa::Buffer::new();
            self.buf.extend_from_slice(b.format(self.rows).as_bytes());
            self.buf.push(b'\n');
        }
        self.inner.write_all(&self.buf)?;
        self.inner.flush()?;
        Ok((self.inner, self.rows))
    }

    /// Appends bytes that a worker already formatted, adding `rows` to the
    /// tally. The parallel CPU backend uses this to splice per-slice output
    /// into the stream in input order.
    pub fn write_raw(&mut self, bytes: &[u8], rows: u64) -> io::Result<()> {
        self.rows += rows;
        if self.format == Format::CountOnly {
            return Ok(());
        }
        self.buf.extend_from_slice(bytes);
        self.maybe_flush()
    }

    /// Emits the CSV header row once, if the query has a static shape.
    pub fn write_header(&mut self, program: &Program) -> io::Result<()> {
        if self.format != Format::Csv || self.header_written {
            return Ok(());
        }
        self.header_written = true;
        if let Some(cols) = program.csv_headers() {
            for (i, c) in cols.iter().enumerate() {
                if i > 0 {
                    self.buf.push(b',');
                }
                write_csv_cell(&mut self.buf, c.as_bytes());
            }
            self.buf.push(b'\n');
        }
        Ok(())
    }

    /// A whole input line, byte for byte.
    pub fn passthrough(&mut self, line: &[u8]) -> io::Result<()> {
        self.rows += 1;
        if self.format == Format::CountOnly {
            return Ok(());
        }
        if self.format == Format::Csv {
            // `.` has no columns to speak of, so emit the line as one cell.
            write_csv_cell(&mut self.buf, line);
        } else {
            self.buf.extend_from_slice(line);
        }
        self.buf.push(b'\n');
        self.maybe_flush()
    }

    /// A single extracted value.
    pub fn value(&mut self, slot: &Slot<'_>) -> io::Result<()> {
        self.rows += 1;
        if self.format == Format::CountOnly {
            return Ok(());
        }
        match self.format {
            Format::Csv => csv_value(&mut self.buf, slot),
            _ => self.buf.extend_from_slice(slot.out_bytes()),
        }
        self.buf.push(b'\n');
        self.maybe_flush()
    }

    /// One projected object. `keys` and `slots` are parallel.
    pub fn projection(&mut self, keys: &[String], slots: &[Slot<'_>]) -> io::Result<()> {
        debug_assert_eq!(keys.len(), slots.len());
        self.rows += 1;
        if self.format == Format::CountOnly {
            return Ok(());
        }
        match self.format {
            Format::Csv => {
                for (i, s) in slots.iter().enumerate() {
                    if i > 0 {
                        self.buf.push(b',');
                    }
                    csv_value(&mut self.buf, s);
                }
            }
            _ => {
                self.buf.push(b'{');
                for (i, (k, s)) in keys.iter().zip(slots).enumerate() {
                    if i > 0 {
                        self.buf.push(b',');
                    }
                    write_json_string(&mut self.buf, k.as_bytes());
                    self.buf.push(b':');
                    self.buf.extend_from_slice(s.out_bytes());
                }
                self.buf.push(b'}');
            }
        }
        self.buf.push(b'\n');
        self.maybe_flush()
    }

    /// The single row produced by an ungrouped aggregate.
    pub fn aggregate(&mut self, kind: AggKind, state: &AggState) -> io::Result<()> {
        self.rows += 1;
        let value = match state.finish(kind) {
            Some(v) => format_number(v),
            None => "null".to_string(),
        };
        if self.format == Format::CountOnly {
            // `--count` on an aggregate reports the aggregate itself; a bare
            // "1" would be useless.
            self.rows = 0;
            self.buf.clear();
            self.buf.extend_from_slice(value.as_bytes());
            self.buf.push(b'\n');
            self.format = Format::Ndjson;
            return Ok(());
        }
        match self.format {
            Format::Csv => self.buf.extend_from_slice(value.as_bytes()),
            _ => {
                // A bare number is valid NDJSON and pipes straight into other
                // tools; wrapping it in an object would just make people
                // reach for another jq call.
                self.buf.extend_from_slice(value.as_bytes());
            }
        }
        self.buf.push(b'\n');
        self.maybe_flush()
    }

    /// One row per group, in the order given.
    pub fn groups(
        &mut self,
        key_name: &str,
        kind: AggKind,
        groups: &[(GroupKey, AggState)],
    ) -> io::Result<()> {
        for (key, state) in groups {
            self.rows += 1;
            let value = match state.finish(kind) {
                Some(v) => format_number(v),
                None => "null".to_string(),
            };
            match self.format {
                Format::Csv => {
                    write_csv_cell(&mut self.buf, &key.to_plain());
                    self.buf.push(b',');
                    self.buf.extend_from_slice(value.as_bytes());
                }
                _ => {
                    self.buf.push(b'{');
                    write_json_string(&mut self.buf, key_name.as_bytes());
                    self.buf.push(b':');
                    self.buf.extend_from_slice(&key.to_json());
                    self.buf.push(b',');
                    write_json_string(&mut self.buf, kind.as_str().as_bytes());
                    self.buf.push(b':');
                    self.buf.extend_from_slice(value.as_bytes());
                    self.buf.push(b'}');
                }
            }
            self.buf.push(b'\n');
            self.maybe_flush()?;
        }
        // `--count` over groups means "how many groups", which is genuinely
        // useful, so leave `rows` alone here.
        Ok(())
    }
}

/// The JSON key name used for the group column: the last path step, so
/// `group_by(.req.host)` produces `{"host": ..., "count": ...}`.
pub fn group_key_name(program: &Program) -> String {
    match program.group_by {
        Some(id) => match program.path(id).steps.last() {
            Some(crate::query::Step::Key(k)) => k.clone(),
            Some(crate::query::Step::Index(i)) => format!("[{i}]"),
            None => "key".to_string(),
        },
        None => "key".to_string(),
    }
}

/// True when the query's output shape has named columns for CSV.
pub fn csv_is_meaningful(program: &Program) -> bool {
    !matches!(program.output, Output::Passthrough)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(raw: &str, kind: Kind) -> Slot<'_> {
        Slot {
            kind,
            raw: raw.as_bytes(),
        }
    }

    fn render(f: impl FnOnce(&mut Writer<Vec<u8>>), format: Format) -> String {
        let mut w = Writer::new(Vec::new(), format);
        f(&mut w);
        let (bytes, _) = w.into_inner().unwrap();
        String::from_utf8(bytes).unwrap()
    }

    #[test]
    fn json_strings_escape_control_characters() {
        // 0x01 has no short escape, so it must be emitted as a \u sequence;
        // passing the raw byte through would produce invalid JSON.
        let mut out = Vec::new();
        write_json_string(&mut out, b"a\nb\"c\\d\x01");
        assert_eq!(out, br#""a\nb\"c\\d\u0001""#);
    }

    #[test]
    fn csv_quotes_only_when_required() {
        let mut out = Vec::new();
        write_csv_cell(&mut out, b"plain");
        assert_eq!(out, b"plain");
        out.clear();
        write_csv_cell(&mut out, b"a,b");
        assert_eq!(out, br#""a,b""#);
        out.clear();
        write_csv_cell(&mut out, b"say \"hi\"");
        assert_eq!(out, br#""say ""hi""""#);
    }

    #[test]
    fn projection_preserves_raw_number_text() {
        let s = render(
            |w| {
                w.projection(&["n".to_string()], &[slot("1.500", Kind::Num)])
                    .unwrap()
            },
            Format::Ndjson,
        );
        assert_eq!(s, "{\"n\":1.500}\n");
    }

    #[test]
    fn csv_renders_strings_unquoted_and_nulls_empty() {
        let s = render(
            |w| {
                w.projection(
                    &["a".into(), "b".into()],
                    &[slot(r#""hi""#, Kind::Str), slot("null", Kind::Null)],
                )
                .unwrap()
            },
            Format::Csv,
        );
        assert_eq!(s, "hi,\n");
    }

    #[test]
    fn csv_unescapes_string_payloads() {
        let s = render(
            |w| {
                w.projection(&["a".into()], &[slot(r#""a\nb""#, Kind::Str)])
                    .unwrap()
            },
            Format::Csv,
        );
        // The newline forces quoting.
        assert_eq!(s, "\"a\nb\"\n");
    }

    #[test]
    fn count_only_prints_just_the_count() {
        let mut w = Writer::new(Vec::new(), Format::CountOnly);
        for _ in 0..7 {
            w.passthrough(b"{}").unwrap();
        }
        assert_eq!(w.rows(), 7);
        let (bytes, rows) = w.into_inner().unwrap();
        // Rows are suppressed on the way through...
        assert!(bytes.is_empty());
        assert_eq!(rows, 7);
        // ...and only `finish` emits the tally.
        let mut w = Writer::new(Vec::new(), Format::CountOnly);
        for _ in 0..7 {
            w.passthrough(b"{}").unwrap();
        }
        let mut w2 = Writer::new(Vec::new(), Format::CountOnly);
        for _ in 0..7 {
            w2.passthrough(b"{}").unwrap();
        }
        let (sink, _) = w2.finish().unwrap();
        assert_eq!(sink, b"7\n");
    }

    #[test]
    fn group_rows_use_the_last_path_segment_as_the_key_name() {
        let p = crate::query::parse("group_by(.req.host) | count").unwrap();
        assert_eq!(group_key_name(&p), "host");
    }

    #[test]
    fn group_output_is_ndjson_objects() {
        let mut st = AggState::default();
        st.push_count();
        st.push_count();
        let s = render(
            |w| {
                w.groups(
                    "host",
                    AggKind::Count,
                    &[(GroupKey::Str(b"a".to_vec()), st)],
                )
                .unwrap()
            },
            Format::Ndjson,
        );
        assert_eq!(s, "{\"host\":\"a\",\"count\":2}\n");
    }
}
