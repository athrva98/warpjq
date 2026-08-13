//! Properties that must hold no matter how the work is sliced up.
//!
//! Chunk size, thread count and backend are all implementation detail: none of
//! them may change a single output byte. Every serious bug this project has
//! had lived in exactly that seam --
//!
//!   * a chunk routed to the CPU mid-pipeline emitted its rows ahead of a GPU
//!     chunk still in flight, so streamed input came out in the wrong order;
//!   * the streamed reader let a chunk exceed the slot capacity, which sent
//!     every chunk down that same broken path;
//!   * `--count` dropped kernel-declined lines because the merge inferred
//!     "no row" from "no bytes".
//!
//! None of those were parser bugs. They were all "the answer depends on how it
//! was divided up" bugs, which is what this file is for.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use warpjq_core::exec::{run_bytes, OnInvalid, Options, Preference};
use warpjq_core::output::Format;

fn gpu_available() -> bool {
    warpjq_core::exec::GpuStatus::detect().is_available()
}

fn prefs() -> Vec<Preference> {
    if gpu_available() {
        vec![Preference::Cpu, Preference::Gpu]
    } else {
        vec![Preference::Cpu]
    }
}

/// A corpus with enough variety that most queries match some but not all rows.
fn corpus(seed: u64, lines: usize) -> Vec<u8> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let hosts = ["a", "b", "c", "web-01", "日本"];
    let mut out = String::new();
    for i in 0..lines {
        if i % 41 == 0 {
            out.push('\n'); // blank lines are skipped, not counted
        }
        if i % 97 == 0 {
            out.push_str("{oops not json\n");
        }
        out.push_str(&format!(
            r#"{{"i":{i},"status":{},"bytes":{},"host":"{}","msg":"line {i}","nested":{{"v":{}}}}}"#,
            [200, 301, 404, 500][rng.gen_range(0..4)],
            rng.gen_range(0..100000),
            hosts[rng.gen_range(0..hosts.len())],
            rng.gen_range(0..500),
        ));
        out.push('\n');
    }
    out.into_bytes()
}

/// The full spread of query shapes, since each exercises a different output
/// path (passthrough, single value, projection, scalar reduce, grouped reduce).
const QUERIES: &[&str] = &[
    ".",
    ".i",
    ".nested.v",
    "select(.status == 500)",
    "select(.status >= 400) | {i: .i, h: .host}",
    "count",
    "select(.status == 500) | count",
    "sum(.bytes)",
    "min(.bytes)",
    "max(.bytes)",
    "avg(.bytes)",
    "group_by(.host) | count",
    "group_by(.status) | sum(.bytes)",
];

fn opts(chunk_bytes: usize, threads: usize, format: Format) -> Options {
    Options {
        format,
        chunk_bytes,
        threads,
        on_invalid: OnInvalid::Skip,
        ..Default::default()
    }
}

fn run(q: &str, data: &[u8], o: &Options, pref: Preference) -> String {
    let program = warpjq_core::parse(q).unwrap_or_else(|e| panic!("{q}: {e}"));
    let (bytes, _) =
        run_bytes(&program, data, o, pref).unwrap_or_else(|e| panic!("{q} [{pref:?}]: {e}"));
    String::from_utf8_lossy(&bytes).into_owned()
}

// ---------------------------------------------------------------------------
// Chunk size
// ---------------------------------------------------------------------------

#[test]
fn output_is_independent_of_chunk_size() {
    let data = corpus(1, 3000);
    // Sizes chosen to straddle every interesting case: smaller than a line,
    // around one line, prime numbers that land mid-line, and larger than the
    // whole input.
    let sizes = [1, 2, 7, 61, 100, 997, 4096, 65536, 1 << 20, 1 << 24];

    for pref in prefs() {
        for q in QUERIES {
            let baseline = run(q, &data, &opts(1 << 20, 4, Format::Ndjson), pref);
            for size in sizes {
                let got = run(q, &data, &opts(size, 4, Format::Ndjson), pref);
                assert_eq!(
                    got, baseline,
                    "`{q}` on {pref:?} changed at chunk size {size}"
                );
            }
        }
    }
}

#[test]
fn output_is_independent_of_chunk_size_for_every_format() {
    let data = corpus(2, 800);
    for pref in prefs() {
        for (q, format) in [
            ("{i: .i, h: .host}", Format::Csv),
            ("select(.status == 500)", Format::CountOnly),
            ("group_by(.host) | count", Format::Csv),
        ] {
            let baseline = run(q, &data, &opts(1 << 20, 4, format), pref);
            for size in [3, 64, 1000, 65536] {
                let got = run(q, &data, &opts(size, 4, format), pref);
                assert_eq!(
                    got, baseline,
                    "`{q}` ({format:?}) on {pref:?} changed at chunk size {size}"
                );
            }
        }
    }
}

#[test]
fn a_chunk_boundary_landing_exactly_on_a_newline_is_handled() {
    // Off-by-one country. Build lines of a known width and set the chunk size
    // to exact multiples, so boundaries land on, just before, and just after
    // each terminator.
    let line = br#"{"i":1234567890123}"#; // 19 bytes + newline = 20
    let mut data = Vec::new();
    for _ in 0..200 {
        data.extend_from_slice(line);
        data.push(b'\n');
    }
    for pref in prefs() {
        let baseline = run(".i", &data, &opts(1 << 20, 1, Format::Ndjson), pref);
        assert_eq!(baseline.lines().count(), 200);
        for size in [19, 20, 21, 39, 40, 41, 199, 200, 201] {
            let got = run(".i", &data, &opts(size, 1, Format::Ndjson), pref);
            assert_eq!(got, baseline, "{pref:?} broke at chunk size {size}");
        }
    }
}

#[test]
fn input_without_a_trailing_newline_is_not_truncated() {
    for pref in prefs() {
        for tail in ["", "\n"] {
            let data = format!("{{\"a\":1}}\n{{\"a\":2}}{tail}");
            for size in [1, 3, 8, 9, 17, 4096] {
                let got = run(".a", data.as_bytes(), &opts(size, 1, Format::Ndjson), pref);
                assert_eq!(
                    got,
                    "1\n2\n",
                    "{pref:?} lost the last line at chunk size {size} \
                     (trailing newline: {})",
                    !tail.is_empty()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Thread count
// ---------------------------------------------------------------------------

#[test]
fn output_is_independent_of_thread_count() {
    let data = corpus(3, 4000);
    for q in QUERIES {
        let baseline = run(q, &data, &opts(1 << 16, 1, Format::Ndjson), Preference::Cpu);
        for threads in [2, 3, 4, 8, 16, 32] {
            let got = run(
                q,
                &data,
                &opts(1 << 16, threads, Format::Ndjson),
                Preference::Cpu,
            );
            assert_eq!(got, baseline, "`{q}` changed with {threads} threads");
        }
    }
}

#[test]
fn thread_count_does_not_perturb_floating_point_sums() {
    // Summation order changes with the slice count, and floating point
    // addition is not associative. The merge has to be deterministic anyway,
    // so this pins that it is.
    let mut data = String::new();
    for i in 0..5000 {
        data.push_str(&format!("{{\"x\":{}.{}}}\n", i % 1000, i % 97));
    }
    let baseline = run(
        "sum(.x)",
        data.as_bytes(),
        &opts(1 << 16, 1, Format::Ndjson),
        Preference::Cpu,
    );
    for threads in [2, 4, 8, 16] {
        let got = run(
            "sum(.x)",
            data.as_bytes(),
            &opts(1 << 16, threads, Format::Ndjson),
            Preference::Cpu,
        );
        assert_eq!(got, baseline, "sum drifted with {threads} threads");
    }
}

// ---------------------------------------------------------------------------
// Cross-cutting properties
// ---------------------------------------------------------------------------

#[test]
fn filtering_never_reorders_or_rewrites_lines() {
    let data = corpus(4, 1500);
    let text = String::from_utf8_lossy(&data).into_owned();
    let valid: Vec<&str> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && warpjq_core::json::validate(l.as_bytes()).is_ok())
        .collect();

    for pref in prefs() {
        let out = run(
            "select(.status >= 400)",
            &data,
            &opts(1 << 12, 4, Format::Ndjson),
            pref,
        );
        // Every emitted line must appear in the input, in order, unmodified.
        let mut it = valid.iter();
        for line in out.lines() {
            assert!(
                it.any(|l| *l == line),
                "{pref:?} emitted a line that is not the next matching input \
                 line: {line}"
            );
        }
    }
}

#[test]
fn count_always_equals_the_number_of_rows() {
    let data = corpus(5, 2000);
    for pref in prefs() {
        for (rows_q, count_q) in [
            (".", "count"),
            ("select(.status == 500)", "select(.status == 500) | count"),
            (
                "select(.status >= 400) | {i: .i}",
                "select(.status >= 400) | count",
            ),
        ] {
            let rows = run(rows_q, &data, &opts(1 << 13, 4, Format::Ndjson), pref)
                .lines()
                .count();
            let counted: usize = run(count_q, &data, &opts(1 << 13, 4, Format::Ndjson), pref)
                .trim()
                .parse()
                .unwrap();
            assert_eq!(rows, counted, "`{rows_q}` vs `{count_q}` on {pref:?}");
        }
    }
}

#[test]
fn grouped_totals_reconcile_with_ungrouped_ones() {
    let data = corpus(6, 2500);
    for pref in prefs() {
        let total: f64 = run(
            "sum(.bytes)",
            &data,
            &opts(1 << 13, 4, Format::Ndjson),
            pref,
        )
        .trim()
        .parse()
        .unwrap();
        let grouped = run(
            "group_by(.host) | sum(.bytes)",
            &data,
            &opts(1 << 13, 4, Format::Ndjson),
            pref,
        );
        let sum: f64 = grouped
            .lines()
            .map(|l| {
                let at = l.rfind(':').unwrap();
                l[at + 1..].trim_end_matches('}').parse::<f64>().unwrap()
            })
            .sum();
        assert_eq!(total, sum, "group sums do not reconcile on {pref:?}");

        let n: u64 = run("count", &data, &opts(1 << 13, 4, Format::Ndjson), pref)
            .trim()
            .parse()
            .unwrap();
        let grouped_n: u64 = run(
            "group_by(.host) | count",
            &data,
            &opts(1 << 13, 4, Format::Ndjson),
            pref,
        )
        .lines()
        .map(|l| {
            let at = l.rfind(':').unwrap();
            l[at + 1..].trim_end_matches('}').parse::<u64>().unwrap()
        })
        .sum();
        assert_eq!(n, grouped_n, "group counts do not reconcile on {pref:?}");
    }
}

#[test]
fn group_output_is_sorted_and_free_of_duplicate_keys() {
    let data = corpus(7, 2000);
    for pref in prefs() {
        for size in [1 << 10, 1 << 16] {
            let grouped = run(
                "group_by(.host) | count",
                &data,
                &opts(size, 4, Format::Ndjson),
                pref,
            );
            let keys: Vec<&str> = grouped.lines().collect();
            let mut sorted = keys.clone();
            sorted.sort();
            assert_eq!(keys, sorted, "groups are not in key order on {pref:?}");
            let mut deduped = keys.clone();
            deduped.dedup();
            assert_eq!(
                keys.len(),
                deduped.len(),
                "a group key appears twice on {pref:?} at chunk size {size}"
            );
        }
    }
}

#[test]
fn an_empty_or_blank_input_produces_the_same_answer_everywhere() {
    for pref in prefs() {
        for data in [&b""[..], b"\n", b"\n\n\n", b"   \n  \n"] {
            for size in [1, 4, 4096] {
                let o = opts(size, 2, Format::Ndjson);
                assert_eq!(run(".", data, &o, pref), "");
                assert_eq!(run("count", data, &o, pref).trim(), "0");
                assert_eq!(run("sum(.x)", data, &o, pref).trim(), "0");
                assert_eq!(run("min(.x)", data, &o, pref).trim(), "null");
                assert_eq!(run("avg(.x)", data, &o, pref).trim(), "null");
                assert_eq!(run("group_by(.h) | count", data, &o, pref), "");
            }
        }
    }
}

#[test]
fn malformed_line_handling_is_independent_of_slicing() {
    // The line that fails must be the same line regardless of where the
    // chunker happened to cut.
    let data = corpus(8, 1200);
    for pref in prefs() {
        let baseline = run(".i", &data, &opts(1 << 20, 1, Format::Ndjson), pref);
        for (size, threads) in [(7, 1), (64, 2), (1000, 4), (1 << 15, 8)] {
            let got = run(".i", &data, &opts(size, threads, Format::Ndjson), pref);
            assert_eq!(
                got, baseline,
                "{pref:?} skipped a different set of lines at chunk {size} / \
                 {threads} threads"
            );
        }
    }
}
