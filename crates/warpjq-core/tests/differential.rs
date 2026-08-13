//! Differential testing against the CPU engine and against real jq.
//!
//! The claim "a GPU JSON tool that agrees with jq" is only worth anything if
//! it is checked against adversarial input rather than the happy path. So this
//! harness generates NDJSON designed to break parsers: unicode, escapes,
//! surrogate pairs, integers past 2^53, duplicate keys, deep nesting, empty
//! containers, CRLF, blank lines, and deliberately malformed lines.
//!
//! Two different assertions, deliberately not conflated:
//!
//! * **GPU == CPU, byte for byte**, over the full hostile corpus. No
//!   exceptions; the two engines are the same algorithm and must agree.
//! * **warpjq == jq**, byte for byte, over a corpus that excludes the
//!   spellings jq rewrites on output (`\uXXXX`, `\/`, exponent-form numbers).
//!   Those exclusions are not a corpus that happens to avoid the problem --
//!   they are named here, asserted individually by
//!   `rendering_differences_from_jq_are_documented`, and the *semantics* are
//!   separately asserted to match by
//!   `query_semantics_match_jq_even_where_rendering_differs`.
//!
//! The GPU comparison is skipped (loudly) when there is no device. The jq
//! comparison is skipped when jq is not installed, which is worth watching
//! for, because a silent skip is how the jq tests sat green and unexercised
//! until jq was actually installed and immediately found real divergences.
//!
//! Run with `--include-ignored` for the slow, large-input cases, and with
//! `--test-threads=1` when a GPU is present: each test builds its own device
//! context, and the default parallelism will exhaust a small card.

use std::process::{Command, Stdio};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use warpjq_core::exec::{OnInvalid, Options, Preference};
use warpjq_core::output::Format;
use warpjq_core::query::{AggKind, Output};

// ---------------------------------------------------------------------------
// Input generation
// ---------------------------------------------------------------------------

const KEYS: &[&str] = &[
    "status", "bytes", "host", "msg", "nested", "arr", "flag", "n",
];
const HOSTS: &[&str] = &["a", "b", "c", "web-01", "ünïcøde", "with\\\"quote", "日本"];

/// Which awkward spellings a corpus is allowed to contain.
///
/// Both flags exist to keep the jq oracle honest. Excluding a spelling is
/// fine; excluding it silently is not, so each flag names what it drops and
/// why, and a separate test asserts the difference it is compensating for.
#[derive(Copy, Clone, Debug)]
struct CorpusOpts {
    /// `\uXXXX`, `\/` and exponent-form numbers, which jq rewrites on output
    /// at every version.
    renormalised: bool,
    /// Number literals only jq 1.7 and later preserve: integers past 2^53,
    /// trailing-zero decimals, and `1.0`. jq 1.6 prints these through a
    /// double and loses digits.
    fragile_numbers: bool,
}

impl CorpusOpts {
    /// Everything. Used for CPU-versus-GPU comparisons, where both engines
    /// must agree exactly whatever the input looks like.
    fn all() -> CorpusOpts {
        CorpusOpts {
            renormalised: true,
            fragile_numbers: true,
        }
    }

    /// Only what the installed jq reproduces byte for byte.
    fn for_jq() -> CorpusOpts {
        CorpusOpts {
            renormalised: false,
            fragile_numbers: jq_preserves_number_literals(),
        }
    }
}

/// Values chosen to sit on the edges every JSON implementation gets wrong.
///
/// See [`CorpusOpts`] for what each flag drops and why. Everything dropped
/// from the jq oracle is still exercised in the CPU-versus-GPU comparison,
/// where both engines must agree exactly regardless of what jq does.
fn hard_value(rng: &mut ChaCha8Rng, depth: u32, opts: CorpusOpts) -> String {
    if opts.renormalised && rng.gen_range(0..14) == 0 {
        return [
            r#""\u0041""#,
            r#""\u00e9""#,
            r#""\u65e5 mixed with literal text""#,
            r#""\ud83d\ude00""#,
            r#""\/slash""#,
            r#""1e3 is a string here""#,
        ][rng.gen_range(0..6)]
        .to_string();
    }
    match rng.gen_range(0..20) {
        0 => "null".into(),
        1 => "true".into(),
        2 => "false".into(),
        3 => "0".into(),
        4 => "-0".into(),
        // Past 2^53, where a round-trip through f64 loses digits. jq only
        // stopped doing that in 1.7.
        5 => if opts.fragile_numbers {
            "9007199254740993"
        } else {
            "9007199254740"
        }
        .into(),
        6 => if opts.fragile_numbers {
            "123456789012345678901234567890"
        } else {
            "1234567890"
        }
        .into(),
        // jq rewrites exponent notation (`1e3` becomes `1E+3`) while warpjq
        // preserves the spelling, so these only appear in corpora that are
        // not used as a jq oracle. The difference itself is asserted by
        // `jq_preserves_number_literals_like_warpjq_does`.
        7 => if opts.renormalised { "1e3" } else { "1000" }.into(),
        8 => if opts.renormalised {
            "-2.5E+10"
        } else {
            "-25000000000"
        }
        .into(),
        // jq 1.6 prints `1.0` as `1`.
        9 => if opts.fragile_numbers { "1.0" } else { "1" }.into(),
        10 => "0.1".into(),
        11 => r#""""#.into(),
        12 => r#""plain""#.into(),
        13 => r#""with \"escaped\" quotes""#.into(),
        14 => r#""tab\there\nnewline\\backslash""#.into(),
        // Surrogate pair for U+1F600.
        15 => r#""emoji 😀 and é""#.into(),
        16 => r#""日本語 ☃ café""#.into(),
        17 => "{}".into(),
        18 => "[]".into(),
        _ => {
            if depth == 0 {
                format!("{}", rng.gen_range(0..1000))
            } else if rng.gen_bool(0.5) {
                let n = rng.gen_range(0..4);
                let items: Vec<String> = (0..n).map(|_| hard_value(rng, depth - 1, opts)).collect();
                format!("[{}]", items.join(","))
            } else {
                let n = rng.gen_range(1..4);
                let items: Vec<String> = (0..n)
                    .map(|i| {
                        format!(
                            "{:?}:{}",
                            KEYS[(i as usize) % KEYS.len()],
                            hard_value(rng, depth - 1, opts)
                        )
                    })
                    .collect();
                format!("{{{}}}", items.join(","))
            }
        }
    }
}

/// One well-formed line with a predictable outer shape, so queries can
/// actually reach into it, and hostile values inside.
fn good_line(rng: &mut ChaCha8Rng, opts: CorpusOpts) -> String {
    let status = [200, 301, 404, 500, 503][rng.gen_range(0..5)];
    let host = HOSTS[rng.gen_range(0..HOSTS.len())];
    format!(
        r#"{{"status":{status},"bytes":{},"host":{:?},"msg":{},"nested":{{"deep":{{"v":{}}}}},"arr":[{},{},{}],"flag":{},"n":{}}}"#,
        rng.gen_range(0..100000),
        host,
        hard_value(rng, 1, opts),
        rng.gen_range(0..500),
        hard_value(rng, 0, opts),
        rng.gen_range(0..10),
        hard_value(rng, 0, opts),
        if rng.gen_bool(0.3) {
            "null"
        } else if rng.gen_bool(0.5) {
            "true"
        } else {
            "false"
        },
        hard_value(rng, 2, opts),
    )
}

fn malformed_line(rng: &mut ChaCha8Rng) -> String {
    let options = [
        r#"{"a":}"#,
        r#"{"a" 1}"#,
        r#"{a:1}"#,
        r#"{"a":01}"#,
        r#"{"a":1.}"#,
        r#"{"a":1e}"#,
        r#"{"a":"unterminated"#,
        r#"{"a":tru}"#,
        r#"{"a":1}trailing"#,
        r#"[1,2"#,
        r#"{"a":"bad \q escape"}"#,
        r#"not json at all"#,
    ];
    options[rng.gen_range(0..options.len())].to_string()
}

/// Builds a corpus. `bad_ratio` is the fraction of lines that are malformed.
fn corpus(seed: u64, lines: usize, bad_ratio: f64, crlf: bool) -> Vec<u8> {
    corpus_inner(seed, lines, bad_ratio, crlf, CorpusOpts::all())
}

/// A corpus safe to compare against jq: none of the spellings jq rewrites on
/// output (`\uXXXX`, `\/`, exponent-form numbers).
fn jq_corpus(seed: u64, lines: usize) -> Vec<u8> {
    corpus_inner(seed, lines, 0.0, false, CorpusOpts::for_jq())
}

fn corpus_inner(seed: u64, lines: usize, bad_ratio: f64, crlf: bool, opts: CorpusOpts) -> Vec<u8> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = String::new();
    for i in 0..lines {
        // Blank lines, which NDJSON readers must skip.
        if i > 0 && i % 37 == 0 {
            out.push('\n');
        }
        if i > 0 && i % 53 == 0 {
            out.push_str("   \n");
        }
        if rng.gen_bool(bad_ratio) {
            out.push_str(&malformed_line(&mut rng));
        } else {
            out.push_str(&good_line(&mut rng, opts));
        }
        out.push('\n');
    }
    if crlf {
        out = out.replace('\n', "\r\n");
    }
    out.into_bytes()
}

// ---------------------------------------------------------------------------
// The query matrix
// ---------------------------------------------------------------------------

/// Every query shape the v0.1 subset supports, over fields that exist and
/// fields that do not.
fn queries() -> Vec<&'static str> {
    vec![
        ".",
        ".status",
        ".host",
        ".msg",
        ".missing",
        ".nested.deep.v",
        ".arr[1]",
        ".arr[9]",
        "select(.status == 500)",
        "select(.status != 500)",
        "select(.status >= 404)",
        "select(.status < 300)",
        r#"select(.host == "a")"#,
        r#"select(.host == "ünïcøde")"#,
        "select(.flag)",
        "select(.flag | not)",
        "select(.flag == null)",
        "select(.flag == true)",
        r#"select(.status == 500 and .host == "b")"#,
        r#"select(.status == 200 or .status == 404)"#,
        "select(.nested.deep.v > 250)",
        "{h: .host, s: .status}",
        "{h: .host, m: .msg, missing: .nope}",
        "{v: .nested.deep.v, a: .arr[0]}",
        "select(.status == 500) | {h: .host}",
        "count",
        "select(.status == 500) | count",
        "sum(.bytes)",
        "min(.bytes)",
        "max(.bytes)",
        "avg(.bytes)",
        "select(.status >= 400) | sum(.bytes)",
        "sum(.nested.deep.v)",
        "sum(.missing)",
        "group_by(.host) | count",
        "group_by(.status) | count",
        "group_by(.host) | sum(.bytes)",
        "group_by(.flag) | count",
        "group_by(.status) | avg(.bytes)",
        "select(.status == 500) | group_by(.host) | count",
    ]
}

fn formats() -> Vec<Format> {
    vec![Format::Ndjson, Format::Csv, Format::CountOnly]
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn opts(format: Format) -> Options {
    Options {
        format,
        // Small chunks force multi-chunk paths, slot reuse and cross-chunk
        // merging even on modest corpora, which is where ordering bugs live.
        chunk_bytes: 1 << 16,
        on_invalid: OnInvalid::Skip,
        threads: 4,
        ..Default::default()
    }
}

fn run(query: &str, data: &[u8], format: Format, pref: Preference) -> String {
    let program = warpjq_core::parse(query).unwrap_or_else(|e| panic!("{query}: {e}"));
    let (bytes, _) = warpjq_core::exec::run_bytes(&program, data, &opts(format), pref)
        .unwrap_or_else(|e| panic!("{query} [{pref:?}]: {e}"));
    String::from_utf8_lossy(&bytes).into_owned()
}

fn supports_csv(query: &str) -> bool {
    let p = warpjq_core::parse(query).unwrap();
    warpjq_core::output::csv_is_meaningful(&p)
}

fn gpu_available() -> bool {
    warpjq_core::exec::GpuStatus::detect().is_available()
}

fn format_name(f: Format) -> &'static str {
    match f {
        Format::Ndjson => "ndjson",
        Format::Csv => "csv",
        Format::CountOnly => "count",
    }
}

/// The core assertion: for every query and every output format, the two
/// backends must produce identical bytes.
fn assert_backends_agree(data: &[u8], label: &str) {
    if !gpu_available() {
        eprintln!(
            "differential: SKIPPING the GPU comparison for `{label}`: {}",
            warpjq_core::exec::GpuStatus::detect().reason()
        );
        return;
    }
    let mut checked = 0;
    for q in queries() {
        for f in formats() {
            // `--csv` on a whole-line query has no columns and is rejected by
            // both backends alike; nothing to compare.
            if f == Format::Csv && !supports_csv(q) {
                continue;
            }
            let cpu = run(q, data, f, Preference::Cpu);
            let gpu = run(q, data, f, Preference::Gpu);
            if cpu != gpu {
                let (c, g) = first_difference(&cpu, &gpu);
                panic!(
                    "GPU and CPU disagree\n  corpus: {label}\n  query:  {q}\n  \
                     format: {}\n  cpu: {c}\n  gpu: {g}",
                    format_name(f)
                );
            }
            checked += 1;
        }
    }
    eprintln!("differential: {checked} query/format pairs agree on `{label}`");
}

fn first_difference(a: &str, b: &str) -> (String, String) {
    for (i, (la, lb)) in a.lines().zip(b.lines()).enumerate() {
        if la != lb {
            return (format!("line {i}: {la}"), format!("line {i}: {lb}"));
        }
    }
    (
        format!("{} lines", a.lines().count()),
        format!("{} lines", b.lines().count()),
    )
}

#[test]
fn backends_agree_on_hostile_input() {
    assert_backends_agree(&corpus(1, 4000, 0.0, false), "clean");
}

#[test]
fn backends_agree_with_malformed_lines_mixed_in() {
    assert_backends_agree(&corpus(2, 4000, 0.15, false), "15% malformed");
}

#[test]
fn backends_agree_on_crlf_input() {
    assert_backends_agree(&corpus(3, 2000, 0.05, true), "crlf");
}

#[test]
fn backends_agree_across_many_seeds() {
    for seed in 10..16 {
        assert_backends_agree(&corpus(seed, 800, 0.08, false), &format!("seed {seed}"));
    }
}

#[test]
fn backends_agree_on_edge_case_corpus() {
    // Hand-written cases that a generator is unlikely to hit.
    let lines: Vec<&str> = vec![
        r#"{}"#,
        r#"{"status":500}"#,
        r#"{"status":500,"status":200}"#, // duplicate key: last wins
        r#"{"host":"a","host":"b"}"#,
        r#"{"msg":"}{[],:\" tricky"}"#, // structure inside a string
        r#"{"bytes":9007199254740993}"#,
        r#"{"bytes":-0}"#,
        r#"{"bytes":1e308}"#,
        r#"{"bytes":1e-308}"#,
        r#"{"host":"AB"}"#,     // escapes that decode to ASCII
        r#"{"host":"😀"}"#,     // surrogate pair
        r#"{"host":"\ud83d"}"#, // lone surrogate
        r#"{"nested":{"deep":{"v":1}}}"#,
        r#"{"arr":[]}"#,
        r#"{"arr":[[[[1]]]]}"#,
        r#"{"flag":false}"#,
        r#"{"flag":null}"#,
        r#"   {"status":404}   "#, // leading/trailing whitespace
        r#"{"ünïcøde key":1,"status":500}"#,
        r#"{"a":{"b":{"c":{"d":{"e":{"f":{"g":1}}}}}}}"#,
    ];
    let mut data = lines.join("\n").into_bytes();
    data.push(b'\n');
    assert_backends_agree(&data, "hand-written edge cases");
}

#[test]
fn backends_agree_when_every_line_is_malformed() {
    assert_backends_agree(&corpus(4, 500, 1.0, false), "all malformed");
}

#[test]
fn backends_agree_when_a_projection_expands_far_past_the_output_buffer() {
    // Regression: the device output buffer is sized at 1.5x the chunk, but a
    // projection with several named fields over short lines expands well past
    // that. `k_emit` used to write from prefix-sum offsets with no bound
    // check, and the capacity was only tested afterwards on the host, so the
    // overrun had already happened. It showed up as
    // "an illegal memory access was encountered", i.e. undefined behaviour.
    let mut data = Vec::new();
    for i in 0..40_000 {
        data.extend_from_slice(format!("{{\"a\":{}}}\n", 1_000_000_000 + i).as_bytes());
    }
    let q = "{alpha: .a, bravo: .a, charlie: .a, delta: .a, echo: .a, foxtrot: .a}";

    if !gpu_available() {
        eprintln!("differential: SKIPPING the output-expansion case: no GPU");
        return;
    }
    // Deliberately tiny chunks so the expansion beats the buffer immediately.
    let options = Options {
        format: Format::Ndjson,
        chunk_bytes: 1 << 16,
        on_invalid: OnInvalid::Skip,
        threads: 4,
        ..Default::default()
    };
    let program = warpjq_core::parse(q).unwrap();
    let (cpu, _) =
        warpjq_core::exec::run_bytes(&program, &data, &options, Preference::Cpu).unwrap();
    let (gpu, _) =
        warpjq_core::exec::run_bytes(&program, &data, &options, Preference::Gpu).unwrap();
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "expanding projection produced different byte counts"
    );
    assert!(cpu == gpu, "expanding projection diverged between backends");
}

/// The same shape at the real chunk size, which is what actually overruns.
///
/// `out_cap` is `chunk * 1.5 + 1MB`, so at the small chunk sizes the fast
/// tests use, the flat 1 MB of slack swallows the expansion and the bug never
/// fires. Reproducing it needs a default-sized chunk and enough data to fill
/// one, which is why this is separated out rather than folded above.
#[test]
#[ignore = "slow: needs a default-sized chunk to overrun the output buffer"]
fn a_projection_that_overruns_the_output_buffer_falls_back_instead() {
    if !gpu_available() {
        eprintln!("differential: SKIPPING the overrun case: no GPU");
        return;
    }
    // 26-byte lines sit just under the max_lines ceiling (chunk / 24), so the
    // chunk stays on the device instead of being rejected for line count.
    let line = b"{\"a\":1234567890123456789}\n";
    assert_eq!(line.len(), 26);
    let mut data = Vec::with_capacity(90 << 20);
    while data.len() < (90 << 20) {
        data.extend_from_slice(line);
    }
    let q = "{alpha: .a, bravo: .a, charlie: .a, delta: .a, echo: .a, foxtrot: .a}";
    let program = warpjq_core::parse(q).unwrap();
    let options = Options {
        format: Format::Ndjson,
        on_invalid: OnInvalid::Skip,
        ..Default::default()
    };
    let (cpu, _) =
        warpjq_core::exec::run_bytes(&program, &data, &options, Preference::Cpu).unwrap();
    let (gpu, stats) =
        warpjq_core::exec::run_bytes(&program, &data, &options, Preference::Gpu).unwrap();
    assert!(cpu == gpu, "expanding projection diverged between backends");
    assert!(
        stats.gpu_fallback_lines > 0,
        "expected the device to decline these chunks rather than overrun the buffer"
    );
}

#[test]
fn backends_agree_on_deeply_nested_lines() {
    // The kernel declines past its 64-level stack and hands the line to the
    // CPU evaluator, which used to be recursive and abort the process.
    let mut data = Vec::new();
    data.extend_from_slice(b"{\"a\":1}\n");
    let n = 5_000;
    data.extend_from_slice(b"{\"a\":");
    data.extend(std::iter::repeat(b'[').take(n));
    data.extend(std::iter::repeat(b']').take(n));
    data.extend_from_slice(b"}\n");
    data.extend_from_slice(b"{\"a\":2}\n");
    assert_backends_agree(&data, "deeply nested lines");
}

#[test]
fn backends_agree_on_empty_and_tiny_input() {
    for (data, label) in [
        (Vec::new(), "empty"),
        (b"\n".to_vec(), "one blank line"),
        (b"{}\n".to_vec(), "one empty object"),
        (b"{}".to_vec(), "no trailing newline"),
        (b"\n\n\n".to_vec(), "only blank lines"),
    ] {
        assert_backends_agree(&data, label);
    }
}

#[test]
#[ignore = "slow: generates ~200 MB to exercise multi-chunk pipelining"]
fn backends_agree_on_a_multi_chunk_corpus() {
    let mut data = Vec::new();
    for seed in 0..50 {
        data.extend_from_slice(&corpus(seed, 20_000, 0.02, false));
    }
    eprintln!("differential: corpus is {} MB", data.len() >> 20);
    assert_backends_agree(&data, "multi-chunk");
}

// ---------------------------------------------------------------------------
// jq
// ---------------------------------------------------------------------------

fn jq_path() -> Option<String> {
    let cmd = if cfg!(windows) { "where" } else { "which" };
    let out = Command::new(cmd).arg("jq").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next().map(|l| l.trim().to_string())
}

/// The installed jq's version as (major, minor), if jq is present.
fn jq_version() -> Option<(u32, u32)> {
    let jq = jq_path()?;
    let out = Command::new(jq).arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    // "jq-1.6", "jq-1.7.1", and 1.8 onwards print "jq-1.8.2".
    let v = text.trim().trim_start_matches("jq-");
    let mut parts = v.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().unwrap_or(0);
    Some((major, minor))
}

/// True when the installed jq echoes number literals back unchanged.
///
/// jq 1.6 parses every number into a double and re-prints it, so
/// `123456789012345678901234567890` comes back as
/// `123456789012345680000000000000` and `1.0` as `1`. jq 1.7 keeps the
/// original text when the value is not modified. Ubuntu 22.04 still ships
/// 1.6, so this is not a hypothetical.
fn jq_preserves_number_literals() -> bool {
    jq_version().map(|v| v >= (1, 7)).unwrap_or(true)
}

/// Runs jq over `data`, via a temporary file rather than a pipe.
///
/// Feeding jq through stdin deadlocks: this writes the whole corpus before
/// reading any output, so once jq fills its ~64 KB stdout pipe buffer it blocks
/// writing, stops draining stdin, and both processes wait for each other. It is
/// a race that depends on corpus size and buffer sizes, so it passed for a
/// while and then hung. A file has no such failure mode, and it is also how
/// `warpjq bench` invokes jq, so the two agree on what is being measured.
fn run_jq(jq: &str, args: &[&str], expr: &str, data: &[u8]) -> Option<String> {
    let path = std::env::temp_dir().join(format!(
        "warpjq-jq-input-{}-{:p}.ndjson",
        std::process::id(),
        data.as_ptr()
    ));
    std::fs::write(&path, data).ok()?;
    let out = Command::new(jq)
        .args(args)
        .arg(expr)
        .arg(&path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok();
    let _ = std::fs::remove_file(&path);
    // jq writes CRLF on Windows. Normalising here keeps the comparisons about
    // content; a real difference in line endings is not something warpjq could
    // produce, since every writer in it emits a bare '\n'.
    Some(
        String::from_utf8_lossy(&out?.stdout)
            .replace("\r\n", "\n")
            .to_string(),
    )
}

/// Compares against real jq on the queries where the translation is exact.
///
/// Only clean input is used here: jq aborts on the first malformed line while
/// warpjq skips it, so a corpus with bad lines is comparing two different
/// documented behaviours, not two implementations.
#[test]
fn matches_jq_on_clean_input() {
    let Some(jq) = jq_path() else {
        eprintln!(
            "differential: SKIPPING the jq comparison: jq is not on PATH. \
             Install it to make this test meaningful."
        );
        return;
    };
    let data = jq_corpus(7, 1500);

    // (warpjq query, jq expression, jq flags)
    let cases: Vec<(&str, String, Vec<&str>)> = vec![
        (".", ".".into(), vec!["-c"]),
        (".status", ".status".into(), vec!["-c"]),
        (".missing", ".missing".into(), vec!["-c"]),
        (".nested.deep.v", ".nested.deep.v".into(), vec!["-c"]),
        (".arr[1]", ".arr[1]".into(), vec!["-c"]),
        (
            "select(.status == 500)",
            "select(.status == 500)".into(),
            vec!["-c"],
        ),
        (
            r#"select(.host == "a")"#,
            r#"select(.host == "a")"#.into(),
            vec!["-c"],
        ),
        (
            "select(.status >= 404)",
            "select(.status >= 404)".into(),
            vec!["-c"],
        ),
        ("select(.flag)", "select(.flag)".into(), vec!["-c"]),
        (
            "select(.flag | not)",
            "select(.flag | not)".into(),
            vec!["-c"],
        ),
        (
            "{h: .host, s: .status}",
            "{h: .host, s: .status}".into(),
            vec!["-c"],
        ),
        ("count", "reduce inputs as $x (0; .+1)".into(), vec!["-n"]),
        (
            "select(.status == 500) | count",
            "reduce (inputs | select(.status == 500)) as $x (0; .+1)".into(),
            vec!["-n"],
        ),
        (
            "sum(.bytes)",
            "reduce (inputs | .bytes) as $x (0; .+$x)".into(),
            vec!["-n"],
        ),
        (
            "max(.bytes)",
            "[inputs | .bytes] | max".into(),
            vec!["-n", "-c"],
        ),
        (
            "min(.bytes)",
            "[inputs | .bytes] | min".into(),
            vec!["-n", "-c"],
        ),
    ];

    let prefs: Vec<Preference> = if gpu_available() {
        vec![Preference::Cpu, Preference::Gpu]
    } else {
        eprintln!("differential: jq comparison will only cover the CPU backend");
        vec![Preference::Cpu]
    };

    let mut checked = 0;
    for (wq, jqe, flags) in &cases {
        let Some(expected) = run_jq(&jq, flags, jqe, &data) else {
            panic!("jq failed to run for `{jqe}`");
        };
        for pref in &prefs {
            let got = run(wq, &data, Format::Ndjson, *pref);
            assert_eq!(
                got.trim_end(),
                expected.trim_end(),
                "warpjq [{pref:?}] disagrees with jq\n  warpjq: {wq}\n  jq:     {jqe}"
            );
            checked += 1;
        }
    }
    eprintln!("differential: {checked} warpjq/jq comparisons agree");
}

/// The places warpjq and jq deliberately render differently, pinned.
///
/// warpjq hands back slices of the input; jq re-serialises. For numbers those
/// agree (jq preserves the mantissa, which is what makes the no-DOM design
/// worth having), but for strings jq decodes `\uXXXX` and `\/` while warpjq
/// preserves the input spelling.
///
/// This is asserted rather than avoided. A corpus that quietly steps around
/// the difference would let it drift in either direction unnoticed; this
/// fails the moment warpjq's rendering changes *or* jq's does.
#[test]
fn rendering_differences_from_jq_are_documented() {
    let Some(jq) = jq_path() else {
        eprintln!("differential: SKIPPING the rendering comparison: jq is not on PATH");
        return;
    };

    // (input line, what jq prints for `.s`, what warpjq prints for `.s`)
    let cases: &[(&str, &str, &str)] = &[
        // Agreed: escapes jq must also emit, and literal UTF-8.
        (r#"{"s":"plain"}"#, r#""plain""#, r#""plain""#),
        (r#"{"s":"tab\there"}"#, r#""tab\there""#, r#""tab\there""#),
        (r#"{"s":"q\"q"}"#, r#""q\"q""#, r#""q\"q""#),
        (r#"{"s":"\b\f"}"#, r#""\b\f""#, r#""\b\f""#),
        // A control character has no short escape, so both spell it \u0001.
        (r#"{"s":"\u0001"}"#, r#""\u0001""#, r#""\u0001""#),
        (r#"{"s":""}"#, r#""""#, r#""""#),
        // Diverging: jq decodes the escape, warpjq keeps the input spelling.
        (r#"{"s":"\u0041"}"#, r#""A""#, r#""\u0041""#),
        (r#"{"s":"\u00e9"}"#, "\"\u{e9}\"", r#""\u00e9""#),
        (
            r#"{"s":"\ud83d\ude00"}"#,
            "\"\u{1f600}\"",
            r#""\ud83d\ude00""#,
        ),
        (r#"{"s":"\/slash"}"#, r#""/slash""#, r#""\/slash""#),
    ];

    for (line, want_jq, want_warpjq) in cases {
        let data = format!("{line}\n");
        let got_jq = run_jq(&jq, &["-c"], ".s", data.as_bytes())
            .unwrap_or_else(|| panic!("jq failed on {line}"));
        assert_eq!(
            got_jq.trim(),
            *want_jq,
            "jq changed its rendering of {line}; update this test and the README"
        );
        let got = run(".s", data.as_bytes(), Format::Ndjson, Preference::Cpu);
        assert_eq!(
            got.trim(),
            *want_warpjq,
            "warpjq changed its rendering of {line}"
        );
    }
}

/// jq preserves number literals, which is the premise of the no-DOM design.
///
/// If this ever fails, the "extracted values are slices of the input" argument
/// in the README needs rewriting, not the code.
#[test]
fn jq_preserves_number_literals_like_warpjq_does() {
    let Some(jq) = jq_path() else {
        eprintln!("differential: SKIPPING the number comparison: jq is not on PATH");
        return;
    };
    let modern = jq_preserves_number_literals();
    eprintln!(
        "differential: jq {:?} {} preserve number literals",
        jq_version(),
        if modern { "does" } else { "does NOT" }
    );

    // warpjq preserves all of these unconditionally, because it never builds
    // a DOM. Whether jq agrees depends on its version, so assert the actual
    // behaviour in both directions rather than the behaviour of whichever jq
    // happened to be installed when the test was written.
    for lit in [
        "1.0",
        "-0.0",
        "0.10",
        "100",
        "9007199254740993",
        "123456789012345678901234567890",
    ] {
        let data = format!("{{\"a\":{lit}}}\n");
        let got = run(".a", data.as_bytes(), Format::Ndjson, Preference::Cpu);
        assert_eq!(got.trim(), lit, "warpjq no longer preserves `{lit}`");

        let got_jq = run_jq(&jq, &["-c"], ".a", data.as_bytes()).unwrap();
        if modern {
            assert_eq!(
                got_jq.trim(),
                lit,
                "jq {:?} no longer preserves `{lit}`; the no-DOM rationale \
                 needs revisiting",
                jq_version()
            );
        } else if lit == "100" {
            // Small integers survive a double round-trip intact even on 1.6.
            assert_eq!(got_jq.trim(), lit);
        }
    }

    // jq 1.6 destroys precision that warpjq keeps. Assert that concretely, so
    // the README's claim about the no-DOM design is backed on old jq too.
    if !modern {
        let data = b"{\"a\":123456789012345678901234567890}\n";
        let got_jq = run_jq(&jq, &["-c"], ".a", data).unwrap();
        assert_ne!(
            got_jq.trim(),
            "123456789012345678901234567890",
            "jq {:?} was expected to lose digits here",
            jq_version()
        );
        assert_eq!(
            run(".a", data, Format::Ndjson, Preference::Cpu).trim(),
            "123456789012345678901234567890"
        );
    }

    // Exponent notation diverges at every jq version, differently.
    // 1.7 and later canonicalise the spelling; 1.6 evaluates it.
    let data = b"{\"a\":1e3}\n";
    let got_jq = run_jq(&jq, &["-c"], ".a", data).unwrap();
    assert_eq!(got_jq.trim(), if modern { "1E+3" } else { "1000" });
    assert_eq!(
        run(".a", data, Format::Ndjson, Preference::Cpu).trim(),
        "1e3"
    );
}

/// Rendering differs; *meaning* must not.
///
/// This is the assertion that actually matters. A query has to select, group
/// and count the same lines as jq regardless of how either tool spells the
/// values on the way out.
#[test]
fn query_semantics_match_jq_even_where_rendering_differs() {
    let Some(jq) = jq_path() else {
        eprintln!("differential: SKIPPING the semantics comparison: jq is not on PATH");
        return;
    };
    // The same two logical values, each spelled two different ways.
    let data = concat!(
        // The same two logical values, each spelled two different ways: once
        // as literal UTF-8, once as a \uXXXX escape.
        r#"{"s":"\u65e5","n":1}"#,
        "\n",
        "{\"s\":\"\u{65e5}\",\"n\":2}\n",
        r#"{"s":"\u0041","n":3}"#,
        "\n",
        r#"{"s":"A","n":4}"#,
        "\n",
    )
    .as_bytes();

    let prefs: Vec<Preference> = if gpu_available() {
        vec![Preference::Cpu, Preference::Gpu]
    } else {
        vec![Preference::Cpu]
    };

    for pref in prefs {
        // A filter must match both spellings of the same string.
        let counted = run("select(.s == \"A\") | count", data, Format::Ndjson, pref);
        let jq_counted = run_jq(
            &jq,
            &["-n"],
            r#"reduce (inputs | select(.s == "A")) as $x (0; .+1)"#,
            data,
        )
        .unwrap();
        assert_eq!(counted.trim(), jq_counted.trim(), "filter on {pref:?}");
        assert_eq!(counted.trim(), "2", "both spellings should have matched");

        // group_by must unify them, and its output goes through the decoder
        // on both sides, so it is byte-identical to jq.
        let grouped = run("group_by(.s) | count", data, Format::Ndjson, pref);
        let jq_grouped = run_jq(
            &jq,
            &["-n", "-c"],
            "[inputs] | group_by(.s) | map({s: .[0].s, count: length}) \
             | sort_by(.s|tostring) | .[]",
            data,
        )
        .unwrap();
        assert_eq!(
            grouped.trim_end(),
            jq_grouped.trim_end(),
            "group_by on {pref:?}"
        );
    }
}

/// `group_by` needs jq to slurp, so it gets its own case with a hand-written
/// equivalent expression.
#[test]
fn matches_jq_on_group_by() {
    let Some(jq) = jq_path() else {
        eprintln!("differential: SKIPPING the jq group_by comparison: jq is not on PATH");
        return;
    };
    let data = jq_corpus(8, 800);
    let expected = run_jq(
        &jq,
        &["-n", "-c"],
        r#"[inputs] | group_by(.host) | map({host: .[0].host, count: length}) | sort_by(.host) | .[]"#,
        &data,
    )
    .expect("jq failed");

    let prefs: Vec<Preference> = if gpu_available() {
        vec![Preference::Cpu, Preference::Gpu]
    } else {
        vec![Preference::Cpu]
    };
    for pref in prefs {
        let got = run("group_by(.host) | count", &data, Format::Ndjson, pref);
        assert_eq!(
            got.trim_end(),
            expected.trim_end(),
            "group_by disagrees with jq on the {pref:?} backend"
        );
    }
}

// ---------------------------------------------------------------------------
// Properties that hold regardless of backend
// ---------------------------------------------------------------------------

#[test]
fn passthrough_output_is_a_subsequence_of_the_input() {
    // Whatever the filter does, it must never reorder or rewrite lines.
    let data = corpus(11, 1000, 0.0, false);
    let input_lines: Vec<&str> = std::str::from_utf8(&data)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();

    let prefs: Vec<Preference> = if gpu_available() {
        vec![Preference::Cpu, Preference::Gpu]
    } else {
        vec![Preference::Cpu]
    };
    for pref in prefs {
        let out = run("select(.status >= 400)", &data, Format::Ndjson, pref);
        let mut it = input_lines.iter();
        for line in out.lines() {
            assert!(
                it.any(|l| *l == line),
                "output line is not present in input order ({pref:?}): {line}"
            );
        }
    }
}

#[test]
fn count_agrees_with_the_number_of_rows_emitted() {
    let data = corpus(12, 1500, 0.05, false);
    let prefs: Vec<Preference> = if gpu_available() {
        vec![Preference::Cpu, Preference::Gpu]
    } else {
        vec![Preference::Cpu]
    };
    for pref in prefs {
        // `.` is itself an output stage, so its counting form is a bare
        // `count` rather than `. | count`.
        for (rows_q, count_q) in [
            ("select(.status == 500)", "select(.status == 500) | count"),
            ("select(.flag)", "select(.flag) | count"),
            (".", "count"),
        ] {
            let rows = run(rows_q, &data, Format::Ndjson, pref).lines().count();
            let counted: usize = run(count_q, &data, Format::Ndjson, pref)
                .trim()
                .parse()
                .unwrap();
            assert_eq!(rows, counted, "`{rows_q}` row count mismatch on {pref:?}");
        }
    }
}

#[test]
fn group_totals_sum_to_the_ungrouped_total() {
    let data = corpus(13, 1200, 0.05, false);
    let prefs: Vec<Preference> = if gpu_available() {
        vec![Preference::Cpu, Preference::Gpu]
    } else {
        vec![Preference::Cpu]
    };
    for pref in prefs {
        let total: u64 = run("count", &data, Format::Ndjson, pref)
            .trim()
            .parse()
            .unwrap();
        let grouped = run("group_by(.host) | count", &data, Format::Ndjson, pref);
        let sum: u64 = grouped
            .lines()
            .map(|l| {
                let at = l.rfind(':').unwrap();
                l[at + 1..].trim_end_matches('}').parse::<u64>().unwrap()
            })
            .sum();
        assert_eq!(
            total, sum,
            "group counts do not sum to the total on {pref:?}"
        );
    }
}

#[test]
fn every_supported_query_shape_is_covered_by_the_matrix() {
    // Guards against adding an Output variant and forgetting to test it.
    let mut seen_passthrough = false;
    let mut seen_path = false;
    let mut seen_project = false;
    let mut seen_aggs = std::collections::HashSet::new();
    let mut seen_group = false;
    for q in queries() {
        let p = warpjq_core::parse(q).unwrap();
        if p.group_by.is_some() {
            seen_group = true;
        }
        match &p.output {
            Output::Passthrough => seen_passthrough = true,
            Output::Path(_) => seen_path = true,
            Output::Project(_) => seen_project = true,
            Output::Agg { kind, .. } => {
                seen_aggs.insert(*kind);
            }
        }
    }
    assert!(seen_passthrough && seen_path && seen_project && seen_group);
    for k in [
        AggKind::Count,
        AggKind::Sum,
        AggKind::Min,
        AggKind::Max,
        AggKind::Avg,
    ] {
        assert!(seen_aggs.contains(&k), "no query in the matrix uses {k:?}");
    }
}
