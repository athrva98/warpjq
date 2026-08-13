//! End-to-end: run a real query over arbitrary bytes as if they were NDJSON.
//!
//! This is the target that exercises the chunker, the line splitter, the
//! evaluator and the writers together, which is where the interesting bugs
//! live. The ordering bug this project actually shipped and then caught came
//! from that seam, not from the parser.

#![no_main]

use libfuzzer_sys::fuzz_target;
use warpjq_core::exec::{run_bytes, OnInvalid, Options, Preference};
use warpjq_core::output::Format;

const QUERIES: &[&str] = &[
    ".",
    ".a",
    ".a.b",
    ".a[0]",
    "select(.a == 1)",
    r#"select(.a == "x")"#,
    "select(.a)",
    "{x: .a, y: .b}",
    "count",
    "sum(.a)",
    "min(.a)",
    "group_by(.a) | count",
];

fuzz_target!(|data: &[u8]| {
    // First byte picks the query and format, the rest is the document. That
    // keeps the corpus useful as it is minimised.
    if data.is_empty() {
        return;
    }
    let query = QUERIES[(data[0] as usize) % QUERIES.len()];
    let body = &data[1..];

    let program = warpjq_core::parse(query).expect("fixed queries must parse");

    for format in [Format::Ndjson, Format::CountOnly] {
        let options = Options {
            format,
            // Tiny chunks so a few hundred bytes still crosses boundaries.
            chunk_bytes: 64,
            max_line_bytes: 4096,
            on_invalid: OnInvalid::Skip,
            threads: 2,
        };
        if warpjq_core::exec::validate(&program, &options).is_err() {
            continue;
        }
        let Ok((out, stats)) = run_bytes(&program, body, &options, Preference::Cpu) else {
            continue;
        };
        // Output must be well-formed enough to be a line stream.
        if !out.is_empty() {
            assert_eq!(*out.last().unwrap(), b'\n', "output must end with a newline");
        }
        assert!(stats.lines_out <= stats.lines_in.max(1) + 1);
    }
});
