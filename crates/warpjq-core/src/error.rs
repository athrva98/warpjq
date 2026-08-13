//! Errors that a human can act on.
//!
//! Design rule from the project plan: a warpjq error must never look like a
//! CUDA stack trace. Query errors carry a caret pointing at the offending
//! byte; runtime errors say what warpjq was doing and what to try next.

use std::fmt;

pub type QueryResult<T> = std::result::Result<T, QueryError>;

/// A syntax or semantics error in the user's query, with source context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryError {
    pub source_text: String,
    pub at: usize,
    pub message: String,
    pub hint: Option<String>,
}

impl QueryError {
    pub fn new(src: &str, at: usize, message: impl Into<String>) -> Self {
        QueryError {
            source_text: src.to_string(),
            at: at.min(src.len()),
            message: message.into(),
            hint: None,
        }
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "invalid query: {}", self.message)?;
        writeln!(f, "  {}", self.source_text)?;
        // Count characters, not bytes, so the caret lands correctly on
        // queries containing non-ASCII key names.
        let col = self.source_text[..self.at].chars().count();
        writeln!(f, "  {}^", " ".repeat(col))?;
        if let Some(h) = &self.hint {
            write!(f, "  help: {h}")?;
        }
        Ok(())
    }
}

impl std::error::Error for QueryError {}

/// Anything that can go wrong while actually processing data.
#[derive(Debug, thiserror::Error)]
pub enum WarpError {
    #[error(transparent)]
    Query(#[from] QueryError),

    #[error("{0}")]
    Io(#[from] std::io::Error),

    #[error("malformed JSON on line {line} of {file}: {detail}\n  help: pass --skip-invalid to drop bad lines instead of stopping")]
    MalformedLine {
        file: String,
        line: u64,
        detail: String,
    },

    #[error("line {line} is {len} bytes, which exceeds the {limit} byte limit\n  help: raise it with --max-line-bytes")]
    LineTooLong { line: u64, len: usize, limit: usize },

    #[error("GPU unavailable: {0}\n  help: warpjq falls back to the CPU backend automatically; pass --backend cpu to silence this, or --backend gpu to make it fatal")]
    GpuUnavailable(String),

    #[error("CUDA error in {op}: {detail}")]
    Cuda { op: &'static str, detail: String },

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, WarpError>;

impl WarpError {
    pub fn other(msg: impl Into<String>) -> Self {
        WarpError::Other(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_caret_lands_under_the_offending_byte() {
        let e = QueryError::new("select(.a)", 7, "nope");
        let rendered = e.to_string();
        let lines: Vec<&str> = rendered.lines().collect();
        let src_line = lines[1];
        let caret_line = lines[2];
        let caret_col = caret_line.find('^').unwrap();
        // Both lines are indented by two spaces, so the column indices line up.
        assert_eq!(&src_line[caret_col..caret_col + 1], ".");
    }

    #[test]
    fn the_caret_counts_characters_not_bytes() {
        // A byte-based column would drift right by one per multi-byte
        // character and point at the wrong place, or inside one.
        let src = "select(.日本 == 1)";
        let at = src.find("==").unwrap();
        let e = QueryError::new(src, at, "nope");
        let rendered = e.to_string();
        let lines: Vec<&str> = rendered.lines().collect();
        let caret_col = lines[2].find('^').unwrap();
        let src_col = lines[1].chars().take(caret_col).count();
        let rest: String = lines[1].chars().skip(src_col).collect();
        assert!(rest.starts_with("=="), "caret pointed at: {rest}");
    }

    #[test]
    fn an_offset_past_the_end_is_clamped_rather_than_panicking() {
        let e = QueryError::new("abc", 999, "ran off the end");
        assert_eq!(e.at, 3);
        let _ = e.to_string();
    }

    #[test]
    fn rendering_never_panics_on_multibyte_sources() {
        for src in ["", ".", "日本語", "select(.😀)", "a\u{0}b"] {
            for at in 0..=src.len() {
                // Only character boundaries are reachable from the lexer, but
                // rendering must be total regardless.
                if src.is_char_boundary(at) {
                    let _ = QueryError::new(src, at, "x").to_string();
                }
            }
        }
    }

    #[test]
    fn hints_are_shown_when_present_and_omitted_when_not() {
        let plain = QueryError::new("x", 0, "bad").to_string();
        assert!(!plain.contains("help:"));
        let hinted = QueryError::new("x", 0, "bad")
            .with_hint("try something else")
            .to_string();
        assert!(hinted.contains("help: try something else"));
    }

    #[test]
    fn runtime_errors_explain_what_to_do_next() {
        let e = WarpError::MalformedLine {
            file: "a.ndjson".into(),
            line: 42,
            detail: "unexpected `}`".into(),
        };
        let s = e.to_string();
        assert!(s.contains("a.ndjson"), "{s}");
        assert!(s.contains("42"), "{s}");
        assert!(s.contains("--skip-invalid"), "should suggest a flag: {s}");

        let e = WarpError::LineTooLong {
            line: 7,
            len: 1 << 30,
            limit: 1 << 20,
        };
        assert!(e.to_string().contains("--max-line-bytes"));

        // The GPU message must never be a bare CUDA string; it has to say what
        // the user can do.
        let e = WarpError::GpuUnavailable("no device".into());
        let s = e.to_string();
        assert!(s.contains("--backend cpu"), "{s}");
        assert!(s.contains("falls back"), "{s}");
    }

    #[test]
    fn cuda_errors_name_the_operation_that_failed() {
        let e = WarpError::Cuda {
            op: "uploading a chunk",
            detail: "out of memory".into(),
        };
        let s = e.to_string();
        assert!(s.contains("uploading a chunk"), "{s}");
        assert!(s.contains("out of memory"), "{s}");
    }

    #[test]
    fn query_errors_convert_into_the_runtime_error_type() {
        let q = QueryError::new("x", 0, "bad");
        let w: WarpError = q.clone().into();
        assert!(matches!(w, WarpError::Query(_)));
        assert_eq!(w.to_string(), q.to_string());
    }

    #[test]
    fn io_errors_pass_through_unwrapped() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let w: WarpError = io.into();
        assert_eq!(w.to_string(), "gone");
    }

    #[test]
    fn other_builds_from_anything_stringy() {
        assert_eq!(WarpError::other("boom").to_string(), "boom");
        assert_eq!(WarpError::other(String::from("boom")).to_string(), "boom");
    }
}
