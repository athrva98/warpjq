//! End-to-end tests that drive the real binary.
//!
//! Everything here goes through `main`, argv and exit codes, because that is
//! the surface users actually touch. The library had good coverage while
//! `run.rs` and `bench.rs` had literally none: 587 lines including the
//! benchmark harness whose output ends up in the README.
//!
//! Exit code contract, asserted throughout:
//!   0  the query ran and produced at least one row (or was an aggregate)
//!   1  the query ran and matched nothing, or a runtime failure
//!   2  the query itself was invalid

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_warpjq");

const LOG: &str = concat!(
    r#"{"status":200,"bytes":10,"host":"a","msg":"ok"}"#,
    "\n",
    r#"{"status":500,"bytes":20,"host":"b","msg":"boom"}"#,
    "\n",
    r#"{"status":500,"bytes":30,"host":"a"}"#,
    "\n",
    r#"{"status":404,"bytes":40,"host":"c","msg":"missing"}"#,
    "\n",
);

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Out {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Out {
    fn lines(&self) -> Vec<&str> {
        self.stdout.lines().collect()
    }
}

/// A scratch directory unique to each test, cleaned up on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!(
            "warpjq-cli-test-{}-{}-{:?}",
            std::process::id(),
            tag,
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    fn write(&self, name: &str, contents: &str) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, contents).expect("write fixture");
        p
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(args: &[&str]) -> Out {
    run_with_stdin(args, None)
}

fn run_with_stdin(args: &[&str], stdin: Option<&str>) -> Out {
    let mut cmd = Command::new(BIN);
    cmd.args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        });
    let mut child = cmd.spawn().expect("spawn warpjq");
    if let Some(data) = stdin {
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(data.as_bytes())
            .expect("write stdin");
    }
    let out = child.wait_with_output().expect("wait");
    Out {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n"),
        stderr: String::from_utf8_lossy(&out.stderr).replace("\r\n", "\n"),
    }
}

fn p(path: &Path) -> String {
    path.display().to_string()
}

// ---------------------------------------------------------------------------
// Exit codes
// ---------------------------------------------------------------------------

#[test]
fn exits_zero_when_rows_matched() {
    let s = Scratch::new("exit0");
    let f = s.write("a.ndjson", LOG);
    let out = run(&["select(.status == 500)", &p(&f)]);
    assert_eq!(out.code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.lines().len(), 2);
}

#[test]
fn exits_one_when_nothing_matched_like_grep() {
    let s = Scratch::new("exit1");
    let f = s.write("a.ndjson", LOG);
    let out = run(&["select(.status == 999)", &p(&f)]);
    assert_eq!(
        out.code, 1,
        "no matches should exit 1 so `if warpjq ...` works"
    );
    assert!(out.stdout.is_empty());
}

#[test]
fn aggregates_exit_zero_even_when_they_count_nothing() {
    // `count` returning 0 is a successful answer, not "no match".
    let s = Scratch::new("exit-agg");
    let f = s.write("a.ndjson", LOG);
    let out = run(&["select(.status == 999) | count", &p(&f)]);
    assert_eq!(out.code, 0);
    assert_eq!(out.stdout.trim(), "0");
}

#[test]
fn exits_two_on_an_invalid_query() {
    let s = Scratch::new("exit2");
    let f = s.write("a.ndjson", LOG);
    for bad in ["select(status == 500)", "reduce .[] as $x (0; .+$x)", "{"] {
        let out = run(&[bad, &p(&f)]);
        assert_eq!(out.code, 2, "`{bad}` should exit 2, stderr: {}", out.stderr);
        assert!(
            out.stderr.contains("invalid query"),
            "`{bad}` stderr: {}",
            out.stderr
        );
    }
}

#[test]
fn exits_nonzero_and_explains_when_the_file_is_missing() {
    let out = run(&[".", "definitely-not-a-real-file.ndjson"]);
    assert_ne!(out.code, 0);
    assert!(
        out.stderr.contains("could not open"),
        "stderr: {}",
        out.stderr
    );
}

// ---------------------------------------------------------------------------
// Input plumbing
// ---------------------------------------------------------------------------

#[test]
fn reads_stdin_when_no_files_are_given() {
    let out = run_with_stdin(&["select(.status == 500) | count"], Some(LOG));
    assert_eq!(out.code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "2");
}

#[test]
fn concatenates_multiple_files_in_argument_order() {
    let s = Scratch::new("multifile");
    let a = s.write("a.ndjson", "{\"n\":1}\n{\"n\":2}\n");
    let b = s.write("b.ndjson", "{\"n\":3}\n");
    let out = run(&[".n", &p(&a), &p(&b)]);
    assert_eq!(out.code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.lines(), vec!["1", "2", "3"]);
}

#[test]
fn aggregates_span_all_input_files() {
    let s = Scratch::new("multifile-agg");
    let a = s.write("a.ndjson", "{\"n\":1}\n{\"n\":2}\n");
    let b = s.write("b.ndjson", "{\"n\":3}\n");
    let out = run(&["sum(.n)", &p(&a), &p(&b)]);
    assert_eq!(out.stdout.trim(), "6");
}

#[test]
fn handles_an_empty_file_without_complaint() {
    let s = Scratch::new("empty");
    let f = s.write("empty.ndjson", "");
    let out = run(&["count", &p(&f)]);
    assert_eq!(out.code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "0");
}

// ---------------------------------------------------------------------------
// Output formats
// ---------------------------------------------------------------------------

#[test]
fn count_flag_prints_only_a_number() {
    let s = Scratch::new("countflag");
    let f = s.write("a.ndjson", LOG);
    for flag in ["--count", "-c"] {
        let out = run(&["select(.status == 500)", flag, &p(&f)]);
        assert_eq!(out.stdout.trim(), "2", "with {flag}");
    }
}

#[test]
fn csv_emits_a_header_then_rows() {
    let s = Scratch::new("csv");
    let f = s.write("a.ndjson", LOG);
    let out = run(&["{h: .host, s: .status}", "--csv", &p(&f)]);
    assert_eq!(out.code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.lines()[0], "h,s");
    assert_eq!(out.lines()[1], "a,200");
}

#[test]
fn csv_on_a_whole_line_query_is_refused_with_a_hint() {
    let s = Scratch::new("csv-bad");
    let f = s.write("a.ndjson", LOG);
    let out = run(&[".", "--csv", &p(&f)]);
    assert_ne!(out.code, 0);
    assert!(out.stderr.contains("--csv needs"), "stderr: {}", out.stderr);
    assert!(out.stderr.contains("help:"), "stderr: {}", out.stderr);
}

#[test]
fn csv_and_count_cannot_be_combined() {
    let out = run(&[".a", "--csv", "--count"]);
    assert_ne!(out.code, 0);
}

// ---------------------------------------------------------------------------
// Malformed input policy
// ---------------------------------------------------------------------------

const MIXED: &str = "{\"a\":1}\n{not json\n{\"a\":2}\n";

#[test]
fn malformed_lines_are_skipped_with_a_warning_by_default() {
    let s = Scratch::new("malformed");
    let f = s.write("m.ndjson", MIXED);
    let out = run(&[".a", &p(&f)]);
    assert_eq!(out.code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.lines(), vec!["1", "2"]);
    assert!(
        out.stderr.contains("malformed"),
        "expected a warning, stderr: {}",
        out.stderr
    );
}

#[test]
fn skip_invalid_suppresses_the_warning() {
    let s = Scratch::new("skipinvalid");
    let f = s.write("m.ndjson", MIXED);
    let out = run(&[".a", "--skip-invalid", &p(&f)]);
    assert_eq!(out.lines(), vec!["1", "2"]);
    assert!(
        out.stderr.is_empty(),
        "--skip-invalid should be silent, stderr: {}",
        out.stderr
    );
}

#[test]
fn strict_aborts_and_names_the_offending_line() {
    let s = Scratch::new("strict");
    let f = s.write("m.ndjson", MIXED);
    let out = run(&[".a", "--strict", &p(&f)]);
    assert_ne!(out.code, 0);
    assert!(
        out.stderr.contains(":2:") || out.stderr.contains("line 2"),
        "should point at line 2, stderr: {}",
        out.stderr
    );
}

#[test]
fn strict_and_skip_invalid_cannot_be_combined() {
    let out = run(&[".a", "--strict", "--skip-invalid"]);
    assert_ne!(out.code, 0);
}

// ---------------------------------------------------------------------------
// Flags that take sizes
// ---------------------------------------------------------------------------

#[test]
fn output_is_identical_across_chunk_sizes() {
    // The chunker is the seam where the streamed-ordering bug lived.
    let s = Scratch::new("chunksize");
    let mut data = String::new();
    for i in 0..500 {
        data.push_str(&format!("{{\"i\":{i}}}\n"));
    }
    let f = s.write("big.ndjson", &data);
    let baseline = run(&[".i", &p(&f)]).stdout;
    for size in ["1", "7", "64", "1KB", "64KB", "1MB"] {
        let out = run(&[".i", "--chunk-size", size, &p(&f)]);
        assert_eq!(out.code, 0, "chunk-size {size} stderr: {}", out.stderr);
        assert_eq!(out.stdout, baseline, "chunk-size {size} changed the output");
    }
}

#[test]
fn rejects_nonsense_sizes() {
    let s = Scratch::new("badsize");
    let f = s.write("a.ndjson", LOG);
    for bad in ["banana", "-1GB", "inf", "nanMB"] {
        let out = run(&[".a", "--chunk-size", bad, &p(&f)]);
        assert_ne!(out.code, 0, "`{bad}` should be rejected");
    }
    let out = run(&[".a", "--chunk-size", "0", &p(&f)]);
    assert_ne!(out.code, 0, "zero chunk size should be rejected");
}

#[test]
fn max_line_bytes_is_enforced() {
    let s = Scratch::new("maxline");
    let long = "x".repeat(5000);
    let f = s.write("long.ndjson", &format!("{{\"a\":\"{long}\"}}\n"));
    let out = run(&[".a", "--max-line-bytes", "100", &p(&f)]);
    assert_ne!(out.code, 0);
    assert!(
        out.stderr.contains("limit") || out.stderr.contains("bytes"),
        "stderr: {}",
        out.stderr
    );
}

#[test]
fn thread_count_does_not_change_the_output() {
    let s = Scratch::new("threads");
    let mut data = String::new();
    for i in 0..2000 {
        data.push_str(&format!("{{\"i\":{i},\"k\":{}}}\n", i % 7));
    }
    let f = s.write("t.ndjson", &data);
    let baseline = run(&[".i", "-j", "1", &p(&f)]).stdout;
    for j in ["2", "3", "8", "16"] {
        let out = run(&[".i", "-j", j, &p(&f)]);
        assert_eq!(out.stdout, baseline, "-j {j} changed the output");
    }
    // And for a grouped aggregate, where merging is order-sensitive.
    let g1 = run(&["group_by(.k) | sum(.i)", "-j", "1", &p(&f)]).stdout;
    for j in ["2", "8"] {
        let out = run(&["group_by(.k) | sum(.i)", "-j", j, &p(&f)]);
        assert_eq!(out.stdout, g1, "-j {j} changed the grouped result");
    }
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

#[test]
fn backend_auto_always_works() {
    let s = Scratch::new("auto");
    let f = s.write("a.ndjson", LOG);
    let out = run(&["count", "--backend", "auto", &p(&f)]);
    assert_eq!(out.code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "4");
}

#[test]
fn backend_cpu_and_auto_agree() {
    let s = Scratch::new("backends");
    let f = s.write("a.ndjson", LOG);
    for q in [
        "select(.status == 500)",
        "{h: .host}",
        "group_by(.host) | count",
        "sum(.bytes)",
    ] {
        let a = run(&[q, "--backend", "auto", &p(&f)]);
        let c = run(&[q, "--backend", "cpu", &p(&f)]);
        assert_eq!(a.stdout, c.stdout, "backends disagree on `{q}`");
    }
}

#[test]
fn backend_gpu_either_works_or_explains_itself() {
    let s = Scratch::new("gpu");
    let f = s.write("a.ndjson", LOG);
    let out = run(&["count", "--backend", "gpu", &p(&f)]);
    if out.code == 0 {
        assert_eq!(out.stdout.trim(), "4");
    } else {
        // Never a raw CUDA dump; always a sentence about the GPU.
        assert!(
            out.stderr.contains("GPU") || out.stderr.contains("gpu"),
            "stderr: {}",
            out.stderr
        );
    }
}

#[test]
fn stats_reports_throughput_and_line_counts() {
    let s = Scratch::new("stats");
    let f = s.write("a.ndjson", LOG);
    let out = run(&["count", "--stats", &p(&f)]);
    assert!(out.stderr.contains("GB/s"), "stderr: {}", out.stderr);
    assert!(out.stderr.contains("lines read"), "stderr: {}", out.stderr);
}

// ---------------------------------------------------------------------------
// gen
// ---------------------------------------------------------------------------

#[test]
fn gen_is_deterministic_for_a_given_seed() {
    let s = Scratch::new("gen-seed");
    let a = s.path("a.ndjson");
    let b = s.path("b.ndjson");
    for out_path in [&a, &b] {
        let out = run(&[
            "gen",
            "--preset",
            "nginx",
            "--size",
            "200KB",
            "--seed",
            "1234",
            "-o",
            &p(out_path),
        ]);
        assert_eq!(out.code, 0, "stderr: {}", out.stderr);
    }
    let da = std::fs::read(&a).unwrap();
    let db = std::fs::read(&b).unwrap();
    assert_eq!(da, db, "same seed must reproduce the same bytes");
    assert!(da.len() >= 200_000);
}

#[test]
fn gen_seeds_differ() {
    let s = Scratch::new("gen-seeds");
    let a = s.path("a.ndjson");
    let b = s.path("b.ndjson");
    run(&[
        "gen",
        "--preset",
        "nginx",
        "--size",
        "100KB",
        "--seed",
        "1",
        "-o",
        &p(&a),
    ]);
    run(&[
        "gen",
        "--preset",
        "nginx",
        "--size",
        "100KB",
        "--seed",
        "2",
        "-o",
        &p(&b),
    ]);
    assert_ne!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
}

#[test]
fn every_gen_preset_round_trips_through_its_own_example_query() {
    let s = Scratch::new("gen-presets");
    for preset in ["nginx", "cloudtrail", "k8s", "nested"] {
        let f = s.path(&format!("{preset}.ndjson"));
        let g = run(&[
            "gen",
            "--preset",
            preset,
            "--size",
            "300KB",
            "--seed",
            "7",
            "-o",
            &p(&f),
        ]);
        assert_eq!(g.code, 0, "{preset}: {}", g.stderr);
        // The tool must be able to read back what it wrote.
        let c = run(&["count", &p(&f)]);
        assert_eq!(c.code, 0, "{preset}: {}", c.stderr);
        assert!(
            c.stdout.trim().parse::<u64>().unwrap() > 10,
            "{preset} produced too few lines"
        );
        assert!(
            c.stderr.is_empty(),
            "{preset} generated lines warpjq itself rejects: {}",
            c.stderr
        );
    }
}

#[test]
fn gen_writes_to_stdout_when_no_output_file_is_given() {
    let out = run(&["gen", "--preset", "nginx", "--size", "20KB", "--seed", "3"]);
    assert_eq!(out.code, 0);
    assert!(out.stdout.len() >= 20_000);
    assert!(out.stdout.starts_with('{'));
}

#[test]
fn gen_lists_its_presets() {
    let out = run(&["gen", "--list"]);
    assert_eq!(out.code, 0);
    for preset in ["nginx", "cloudtrail", "k8s", "nested"] {
        assert!(out.stdout.contains(preset), "missing {preset}");
    }
}

#[test]
fn gen_rejects_an_unknown_preset_and_lists_the_real_ones() {
    let out = run(&["gen", "--preset", "not-a-preset", "--size", "1KB"]);
    assert_ne!(out.code, 0);
    assert!(out.stderr.contains("nginx"), "stderr: {}", out.stderr);
}

#[test]
fn gen_rejects_a_zero_size() {
    let out = run(&["gen", "--preset", "nginx", "--size", "0"]);
    assert_ne!(out.code, 0);
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------

#[test]
fn bench_prints_a_table_with_the_commands_it_ran() {
    let s = Scratch::new("bench");
    let f = s.path("b.ndjson");
    run(&[
        "gen",
        "--preset",
        "nginx",
        "--size",
        "300KB",
        "--seed",
        "5",
        "-o",
        &p(&f),
    ]);
    let out = run(&[
        "bench",
        "select(.status == 500) | count",
        &p(&f),
        "--runs",
        "1",
        "--warmup",
        "0",
    ]);
    assert_eq!(out.code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("engine"), "stdout: {}", out.stdout);
    assert!(
        out.stdout.contains("warpjq (cpu)"),
        "stdout: {}",
        out.stdout
    );
    // Reproducibility: the exact command behind each row must be printed.
    assert!(out.stdout.contains("Commands:"), "stdout: {}", out.stdout);
    assert!(
        out.stdout.contains("--backend cpu"),
        "stdout: {}",
        out.stdout
    );
    // And the promise that no kernel-only timing exists.
    assert!(out.stdout.contains("end-to-end"), "stdout: {}", out.stdout);
}

#[test]
fn bench_rejects_an_invalid_query_before_timing_anything() {
    let s = Scratch::new("bench-badq");
    let f = s.write("b.ndjson", LOG);
    let out = run(&["bench", "select(status == 1)", &p(&f)]);
    assert_eq!(out.code, 2);
    assert!(
        out.stderr.contains("invalid query"),
        "stderr: {}",
        out.stderr
    );
}

#[test]
fn bench_explains_a_missing_file_rather_than_panicking() {
    let out = run(&["bench", "count", "no-such-file.ndjson"]);
    assert_ne!(out.code, 0);
    assert!(
        out.stderr.contains("could not stat"),
        "stderr: {}",
        out.stderr
    );
}

// ---------------------------------------------------------------------------
// Misc surface
// ---------------------------------------------------------------------------

#[test]
fn version_and_help_work() {
    let v = run(&["--version"]);
    assert_eq!(v.code, 0);
    assert!(v.stdout.contains("warpjq"));

    let h = run(&["--help"]);
    assert_eq!(h.code, 0);
    // The help text is where people learn the supported subset.
    assert!(h.stdout.contains("group_by"), "stdout: {}", h.stdout);
    assert!(h.stdout.contains("select"), "stdout: {}", h.stdout);
}

#[test]
fn no_arguments_at_all_is_a_usage_error() {
    let out = run(&[]);
    assert_eq!(out.code, 2);
}

#[test]
fn crlf_input_is_handled_like_lf() {
    let s = Scratch::new("crlf");
    let lf = s.write("lf.ndjson", LOG);
    let crlf = s.write("crlf.ndjson", &LOG.replace('\n', "\r\n"));
    let a = run(&["select(.status == 500) | count", &p(&lf)]);
    let b = run(&["select(.status == 500) | count", &p(&crlf)]);
    assert_eq!(a.stdout, b.stdout);
}

#[test]
fn a_utf8_bom_does_not_break_the_first_line() {
    let s = Scratch::new("bom");
    let f = s.path("bom.ndjson");
    let mut bytes = vec![0xEF, 0xBB, 0xBF];
    bytes.extend_from_slice(LOG.as_bytes());
    std::fs::write(&f, &bytes).unwrap();
    let out = run(&["count", &p(&f)]);
    assert_eq!(out.stdout.trim(), "4", "stderr: {}", out.stderr);
}

#[test]
fn unicode_survives_the_round_trip_through_argv_and_output() {
    let s = Scratch::new("unicode");
    let f = s.write("u.ndjson", "{\"host\":\"日本\"}\n{\"host\":\"a\"}\n");
    let out = run(&["select(.host == \"日本\")", &p(&f)]);
    assert_eq!(out.code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.lines(), vec!["{\"host\":\"日本\"}"]);
}

/// The GPU backend reads real files through its own pinned reader, not the
/// chunker. The differential suite runs everything through `Input::from_bytes`,
/// which is a stream, so none of it covers that path; only tests that go
/// through a file on disk do. These are those tests.
mod pinned_reader {
    use super::*;

    /// `--backend gpu` refuses with this when there is no device, or when the
    /// binary has no CUDA at all, which is every CI runner but the GPU job.
    /// Matched on the phrase run.rs actually prints; guessing at the wording
    /// made these tests fail on all three CPU platforms rather than skip.
    fn no_gpu(stderr: &str) -> bool {
        stderr.contains("the GPU is not usable")
    }

    fn gpu_and_cpu_agree(args: &[&str], file: &str) {
        let gpu = run(&[&["--backend", "gpu"], args, &[file]].concat());
        let cpu = run(&[&["--backend", "cpu"], args, &[file]].concat());
        if no_gpu(&gpu.stderr) {
            eprintln!("cli: SKIPPING, no GPU");
            return;
        }
        assert_eq!(gpu.code, cpu.code, "exit codes differ, gpu: {}", gpu.stderr);
        assert_eq!(
            gpu.stdout, cpu.stdout,
            "output differs\ngpu stderr: {}",
            gpu.stderr
        );
    }

    #[test]
    fn a_line_longer_than_the_chunk_is_handled_without_reading_the_rest() {
        // The chunk buffer cannot hold this line, but the limit allows it, so
        // the reader has to take the line on its own and resume on the device.
        // Before this was fixed the whole remainder of the file went to the
        // CPU as one chunk, which on a large input meant faulting all of it.
        let s = Scratch::new("longline");
        let big = "x".repeat(2 << 20);
        let mut data = String::new();
        data.push_str("{\"a\":1,\"tag\":\"before\"}\n");
        data.push_str(&format!("{{\"a\":2,\"pad\":\"{big}\"}}\n"));
        // Enough after the long line to need several more chunks, which only
        // run if the reader resumed rather than handing off the remainder.
        for i in 0..200_000 {
            data.push_str(&format!("{{\"a\":{},\"tag\":\"after\"}}\n", i % 5));
        }
        let f = s.write("long.ndjson", &data);
        gpu_and_cpu_agree(
            &["--chunk-size", "1MB", "--max-line-bytes", "8MB", "count"],
            &p(&f),
        );
        gpu_and_cpu_agree(
            &[
                "--chunk-size",
                "1MB",
                "--max-line-bytes",
                "8MB",
                "select(.a == 2) | count",
            ],
            &p(&f),
        );
        gpu_and_cpu_agree(
            &[
                "--chunk-size",
                "1MB",
                "--max-line-bytes",
                "8MB",
                "group_by(.tag) | count",
            ],
            &p(&f),
        );
    }

    #[test]
    fn a_line_past_the_limit_is_an_error_not_a_fallback() {
        let s = Scratch::new("longline-err");
        let big = "x".repeat(2 << 20);
        let f = s.write("over.ndjson", &format!("{{\"a\":\"{big}\"}}\n"));
        let out = run(&[
            "--backend",
            "gpu",
            "--chunk-size",
            "1MB",
            "--max-line-bytes",
            "1MB",
            ".a",
            &p(&f),
        ]);
        if no_gpu(&out.stderr) {
            eprintln!("cli: SKIPPING, no GPU");
            return;
        }
        assert_ne!(out.code, 0, "stdout: {}", out.stdout);
        assert!(out.stderr.contains("limit"), "stderr: {}", out.stderr);
    }

    #[test]
    fn many_chunks_from_a_file_agree_with_the_cpu() {
        // Small chunks so a single modest file crosses many buffer boundaries,
        // exercising the re-read of the tail past each newline.
        let s = Scratch::new("multichunk");
        let mut data = String::new();
        for i in 0..120_000 {
            data.push_str(&format!(
                "{{\"status\":{},\"host\":\"h{}\",\"bytes\":{}}}\n",
                if i % 7 == 0 { 500 } else { 200 },
                i % 4,
                i * 3
            ));
        }
        let f = s.write("multi.ndjson", &data);
        for cs in ["64KB", "256KB", "1MB"] {
            gpu_and_cpu_agree(
                &["--chunk-size", cs, "select(.status == 500) | count"],
                &p(&f),
            );
            gpu_and_cpu_agree(
                &["--chunk-size", cs, "group_by(.host) | sum(.bytes)"],
                &p(&f),
            );
            gpu_and_cpu_agree(&["--chunk-size", cs, "select(.status == 500)"], &p(&f));
        }
    }

    #[test]
    fn reading_ahead_agrees_with_the_cpu() {
        // Past 32 chunks the backend allocates a third slot and the reader
        // runs a chunk ahead of the device. At the default chunk size that
        // needs a 2 GB fixture; a small chunk reaches the same code path on a
        // file a test can write. Without this the prefetch protocol, which is
        // where a slot could be refilled while still in flight, is only ever
        // exercised by hand.
        let s = Scratch::new("prefetch");
        let mut data = String::new();
        for i in 0..400_000 {
            data.push_str(&format!(
                "{{\"status\":{},\"host\":\"h{}\",\"bytes\":{}}}
",
                if i % 9 == 0 { 500 } else { 200 },
                i % 6,
                i * 7
            ));
        }
        let f = s.write("pf.ndjson", &data);
        // ~16 MB over 1 MB chunks is ~16 chunks: two slots, no read ahead.
        // Over 256 KB it is ~64 chunks: three slots, reading ahead.
        for cs in ["1MB", "256KB", "128KB"] {
            gpu_and_cpu_agree(&["--chunk-size", cs, "count"], &p(&f));
            gpu_and_cpu_agree(
                &["--chunk-size", cs, "select(.status == 500) | count"],
                &p(&f),
            );
            gpu_and_cpu_agree(
                &["--chunk-size", cs, "group_by(.host) | sum(.bytes)"],
                &p(&f),
            );
            gpu_and_cpu_agree(&["--chunk-size", cs, "select(.status == 500)"], &p(&f));
        }
    }

    #[test]
    fn a_file_with_no_trailing_newline_keeps_its_last_line() {
        let s = Scratch::new("no-trailing-nl");
        let f = s.write("t.ndjson", "{\"a\":1}\n{\"a\":2}\n{\"a\":3}");
        gpu_and_cpu_agree(&["count"], &p(&f));
        gpu_and_cpu_agree(&["select(.a == 3)"], &p(&f));
    }
}
