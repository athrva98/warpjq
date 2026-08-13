//! JSON parser conformance, in the style of nst/JSONTestSuite.
//!
//! The scanner is the foundation everything else sits on, and until now it was
//! tested by a dozen hand-picked cases plus whatever the fuzzed corpus happened
//! to generate. That is not conformance testing. This file enumerates the
//! accept/reject boundary explicitly:
//!
//! * `MUST_ACCEPT`: valid JSON by RFC 8259. Rejecting any of these is a bug.
//! * `MUST_REJECT`: invalid JSON. Accepting any of these is a bug, and the
//!   more dangerous direction: a parser that accepts garbage silently produces
//!   wrong answers rather than an error.
//! * `IMPLEMENTATION_DEFINED`: cases RFC 8259 leaves open. We assert what
//!   warpjq does so the behaviour cannot drift unnoticed, and record where jq
//!   differs.
//!
//! Every case is checked three ways where possible: warpjq's CPU scanner,
//! warpjq's GPU kernel (which must agree exactly), and real jq.

use warpjq_core::exec::{OnInvalid, Options, Preference};
use warpjq_core::json;
use warpjq_core::output::Format;

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

/// Valid JSON. One value per entry; NDJSON means these appear one per line, so
/// nothing here may contain a raw newline.
const MUST_ACCEPT: &[(&str, &str)] = &[
    ("empty object", r#"{}"#),
    ("empty array", r#"[]"#),
    ("empty string", r#""""#),
    ("null", r#"null"#),
    ("true", r#"true"#),
    ("false", r#"false"#),
    ("zero", r#"0"#),
    ("negative zero", r#"-0"#),
    ("integer", r#"123"#),
    ("negative integer", r#"-123"#),
    ("real", r#"1.5"#),
    ("real with leading zero", r#"0.5"#),
    ("exponent lowercase", r#"1e3"#),
    ("exponent uppercase", r#"1E3"#),
    ("exponent plus", r#"1e+3"#),
    ("exponent minus", r#"1e-3"#),
    ("real with exponent", r#"1.5e10"#),
    ("huge integer", r#"123456789012345678901234567890"#),
    ("past 2^53", r#"9007199254740993"#),
    ("tiny exponent", r#"1e-400"#),
    ("giant exponent", r#"1e400"#),
    ("simple object", r#"{"a":1}"#),
    ("nested object", r#"{"a":{"b":{"c":1}}}"#),
    ("object with array", r#"{"a":[1,2,3]}"#),
    ("array of objects", r#"[{"a":1},{"b":2}]"#),
    ("mixed array", r#"[1,"two",null,true,{},[]]"#),
    ("duplicate keys", r#"{"a":1,"a":2}"#),
    ("empty key", r#"{"":1}"#),
    (
        "whitespace everywhere",
        "{ \"a\" : [ 1 , 2 ] , \"b\" : { } }",
    ),
    ("tab whitespace", "{\t\"a\"\t:\t1\t}"),
    ("escaped quote", r#"{"a":"say \"hi\""}"#),
    ("escaped backslash", r#"{"a":"c:\\path"}"#),
    ("escaped solidus", r#"{"a":"a\/b"}"#),
    ("escaped control chars", r#"{"a":"\b\f\n\r\t"}"#),
    ("unicode escape", r#"{"a":"\u0041"}"#),
    ("unicode escape uppercase hex", r#"{"a":"\u00E9"}"#),
    ("surrogate pair", r#"{"a":"\ud83d\ude00"}"#),
    ("literal utf8 2 byte", "{\"a\":\"é\"}"),
    ("literal utf8 3 byte", "{\"a\":\"日\"}"),
    ("literal utf8 4 byte", "{\"a\":\"😀\"}"),
    ("non-ascii key", "{\"日本\":1}"),
    ("structural chars inside a string", r#"{"a":"{}[],:\"" }"#),
    ("deeply nested arrays", r#"[[[[[[[[[[1]]]]]]]]]]"#),
    ("deeply nested objects", r#"{"a":{"a":{"a":{"a":1}}}}"#),
    ("array of empties", r#"[[],[],{},{}]"#),
    ("top-level string", r#""hello""#),
    ("top-level number", r#"42"#),
    ("del character in string", "{\"a\":\"\u{7f}\"}"),
];

/// Invalid JSON. Accepting any of these silently corrupts results.
const MUST_REJECT: &[(&str, &str)] = &[
    ("empty input", r#""#),
    ("whitespace only", r#"   "#),
    ("bare word", r#"nope"#),
    ("truncated true", r#"tru"#),
    ("truncated null", r#"nul"#),
    ("capitalised literal", r#"True"#),
    ("unquoted key", r#"{a:1}"#),
    ("single-quoted key", r#"{'a':1}"#),
    ("single-quoted string", r#"{"a":'b'}"#),
    ("missing value", r#"{"a":}"#),
    ("missing colon", r#"{"a" 1}"#),
    ("missing comma", r#"{"a":1 "b":2}"#),
    ("trailing comma in object", r#"{"a":1,}"#),
    ("trailing comma in array", r#"[1,2,]"#),
    ("leading comma in array", r#"[,1]"#),
    ("double comma", r#"[1,,2]"#),
    ("unclosed object", r#"{"a":1"#),
    ("unclosed array", r#"[1,2"#),
    ("unclosed string", r#"{"a":"b}"#),
    ("unopened object", r#""a":1}"#),
    ("mismatched brackets", r#"{"a":[}"#),
    ("mismatched brackets 2", r#"[}"#),
    ("mismatched brackets 3", r#"{]"#),
    ("close wrong container", r#"[1,2}"#),
    ("trailing content", r#"{"a":1} junk"#),
    ("two values", r#"{"a":1}{"b":2}"#),
    ("leading zero", r#"{"a":01}"#),
    ("leading zeros negative", r#"{"a":-01}"#),
    ("plus sign", r#"{"a":+1}"#),
    ("bare decimal point", r#"{"a":.5}"#),
    ("trailing decimal point", r#"{"a":1.}"#),
    ("no exponent digits", r#"{"a":1e}"#),
    ("exponent sign only", r#"{"a":1e+}"#),
    ("double exponent", r#"{"a":1e3e4}"#),
    ("hex number", r#"{"a":0x10}"#),
    ("infinity literal", r#"{"a":Infinity}"#),
    ("nan literal", r#"{"a":NaN}"#),
    ("bare minus", r#"{"a":-}"#),
    ("invalid escape", r#"{"a":"\q"}"#),
    ("truncated unicode escape", r#"{"a":"\u00"}"#),
    ("non-hex unicode escape", r#"{"a":"\uZZZZ"}"#),
    ("lone backslash", r#"{"a":"\"}"#),
    ("raw tab in string", "{\"a\":\"\t\"}"),
    ("raw control char in string", "{\"a\":\"\u{1}\"}"),
    ("object key is a number", r#"{1:2}"#),
    ("array as object key", r#"{[]:1}"#),
    ("colon in array", r#"[1:2]"#),
    ("comment", r#"{"a":1} // note"#),
    ("just a comma", r#","#),
    ("just a colon", r#":"#),
    ("just a brace", r#"{"#),
    ("just a bracket", r#"["#),
];

/// RFC 8259 leaves these open, or they sit outside what NDJSON can express.
/// The point is not that our answer is the only defensible one, but that it is
/// written down.
const IMPLEMENTATION_DEFINED: &[(&str, &str, bool, &str)] = &[
    (
        "lone high surrogate",
        r#"{"a":"\ud83d"}"#,
        true,
        "warpjq accepts and preserves it; jq rejects the line outright",
    ),
    (
        "lone low surrogate",
        r#"{"a":"\ude00"}"#,
        true,
        "same: accepted, passed through unchanged",
    ),
    (
        "reversed surrogate pair",
        r#"{"a":"\ude00\ud83d"}"#,
        true,
        "accepted; each half is preserved as written",
    ),
    (
        "escaped null character",
        r#"{"a":"\u0000"}"#,
        true,
        "valid JSON; survives as an escape rather than a raw NUL byte",
    ),
    (
        "very deep nesting",
        r#"[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]"#,
        true,
        "no depth limit on the CPU scanner; the kernel defers past 64 levels",
    ),
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn gpu_available() -> bool {
    warpjq_core::exec::GpuStatus::detect().is_available()
}

/// Does warpjq's pipeline accept this line? Uses `--strict`, so a rejected
/// line fails the run rather than being skipped.
fn warpjq_accepts(line: &str, pref: Preference) -> bool {
    let program = warpjq_core::parse(".").unwrap();
    let options = Options {
        format: Format::Ndjson,
        on_invalid: OnInvalid::Abort,
        chunk_bytes: 1 << 16,
        threads: 1,
        ..Default::default()
    };
    let data = format!("{line}\n");
    warpjq_core::exec::run_bytes(&program, data.as_bytes(), &options, pref).is_ok()
}

fn jq_path() -> Option<String> {
    let cmd = if cfg!(windows) { "where" } else { "which" };
    let out = std::process::Command::new(cmd).arg("jq").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next().map(|l| l.trim().to_string())
}

fn jq_accepts(jq: &str, line: &str) -> bool {
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "warpjq-conf-{}-{:?}.json",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::write(&path, line.as_bytes()).unwrap();
    let out = std::process::Command::new(jq)
        .arg("-e")
        .arg(".")
        .arg(&path)
        .output();
    let _ = std::fs::remove_file(&path);
    // `-e` makes jq exit non-zero for a false/null result too, so look at
    // stderr for a parse error instead of relying on the exit code alone.
    match out {
        Ok(o) => !String::from_utf8_lossy(&o.stderr).contains("error"),
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// The scanner in isolation
// ---------------------------------------------------------------------------

#[test]
fn scanner_accepts_every_valid_document() {
    let mut failures = Vec::new();
    for (name, doc) in MUST_ACCEPT {
        if json::validate(doc.as_bytes()).is_err() {
            failures.push(format!("  rejected valid `{name}`: {doc}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} valid documents were rejected:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn scanner_rejects_every_invalid_document() {
    let mut failures = Vec::new();
    for (name, doc) in MUST_REJECT {
        if json::validate(doc.as_bytes()).is_ok() {
            failures.push(format!("  ACCEPTED invalid `{name}`: {doc}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} invalid documents were accepted. These silently produce wrong \
         answers rather than errors:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn implementation_defined_behaviour_is_pinned() {
    for (name, doc, expect_accept, why) in IMPLEMENTATION_DEFINED {
        let got = json::validate(doc.as_bytes()).is_ok();
        assert_eq!(
            got, *expect_accept,
            "`{name}` changed behaviour ({why}): {doc}"
        );
    }
}

#[test]
fn every_error_carries_an_offset_inside_the_input() {
    // A diagnostic that points outside the line would slice-panic when
    // rendered next to the source.
    for (name, doc) in MUST_REJECT {
        if let Err(e) = json::validate(doc.as_bytes()) {
            assert!(
                e.offset <= doc.len(),
                "`{name}` reported offset {} for a {}-byte document",
                e.offset,
                doc.len()
            );
            assert!(!e.detail.is_empty(), "`{name}` gave an empty message");
            let _ = e.to_string();
        }
    }
}

// ---------------------------------------------------------------------------
// Through the whole pipeline, on both backends
// ---------------------------------------------------------------------------

#[test]
fn pipeline_agrees_with_the_scanner_on_every_case() {
    let prefs: Vec<Preference> = if gpu_available() {
        vec![Preference::Cpu, Preference::Gpu]
    } else {
        eprintln!("conformance: SKIPPING the GPU half: no device");
        vec![Preference::Cpu]
    };

    for pref in prefs {
        let mut failures = Vec::new();
        for (name, doc) in MUST_ACCEPT {
            if !warpjq_accepts(doc, pref) {
                failures.push(format!("  {pref:?} rejected valid `{name}`: {doc}"));
            }
        }
        for (name, doc) in MUST_REJECT {
            // An empty or whitespace-only line is skipped as a blank line by
            // the NDJSON reader before it ever reaches the parser, which is
            // the documented behaviour, not a parse.
            if doc.trim().is_empty() {
                continue;
            }
            if warpjq_accepts(doc, pref) {
                failures.push(format!("  {pref:?} ACCEPTED invalid `{name}`: {doc}"));
            }
        }
        assert!(
            failures.is_empty(),
            "{} conformance failures on {pref:?}:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn both_backends_reach_the_same_verdict_on_every_case() {
    if !gpu_available() {
        eprintln!("conformance: SKIPPING the backend comparison: no device");
        return;
    }
    let all = MUST_ACCEPT
        .iter()
        .copied()
        .chain(MUST_REJECT.iter().copied())
        .chain(IMPLEMENTATION_DEFINED.iter().map(|(n, d, _, _)| (*n, *d)));
    for (name, doc) in all {
        if doc.trim().is_empty() {
            continue;
        }
        let cpu = warpjq_accepts(doc, Preference::Cpu);
        let gpu = warpjq_accepts(doc, Preference::Gpu);
        assert_eq!(
            cpu, gpu,
            "backends disagree on `{name}` (cpu={cpu}, gpu={gpu}): {doc}"
        );
    }
}

// ---------------------------------------------------------------------------
// Against jq
// ---------------------------------------------------------------------------

/// Documents where jq accepts input that RFC 8259 forbids and warpjq rejects.
///
/// Every one of these is jq being *lenient*, not warpjq being wrong, and in
/// most cases jq does not merely accept the document, it silently rewrites the
/// value. `{"a":01}` becomes `{"a":1}`; `{"a":NaN}` becomes `{"a":null}`;
/// `{"a":Infinity}` becomes the largest finite double. On log data, that is
/// corruption being papered over. warpjq refuses the line instead.
const JQ_ACCEPTS_WE_REJECT: &[(&str, &str, &str)] = &[
    (
        "two values on one line",
        r#"{"a":1}{"b":2}"#,
        "jq's model is a stream of values; NDJSON is one value per line",
    ),
    ("leading zero", r#"{"a":01}"#, "jq silently reads it as 1"),
    (
        "negative leading zero",
        r#"{"a":-01}"#,
        "jq silently reads it as -1",
    ),
    ("plus sign", r#"{"a":+1}"#, "jq silently reads it as 1"),
    (
        "bare decimal point",
        r#"{"a":.5}"#,
        "jq silently reads it as 0.5",
    ),
    (
        "trailing decimal point",
        r#"{"a":1.}"#,
        "jq silently reads it as 1",
    ),
    (
        "Infinity literal",
        r#"{"a":Infinity}"#,
        "jq clamps it to the largest finite double",
    ),
    ("NaN literal", r#"{"a":NaN}"#, "jq turns it into null"),
];

/// The reverse direction: warpjq accepts, jq rejects.
///
/// jq's surrogate handling is asymmetric: a lone *high* surrogate is a parse
/// error but a lone *low* surrogate is accepted and decoded to U+FFFD.
const WE_ACCEPT_JQ_REJECTS: &[(&str, &str, &str)] = &[
    (
        "lone high surrogate",
        r#"{"a":"\ud83d"}"#,
        "jq: parse error; warpjq preserves the escape",
    ),
    (
        "reversed surrogate pair",
        r#"{"a":"\ude00\ud83d"}"#,
        "jq: parse error on the high half",
    ),
];

#[test]
fn the_jq_accept_reject_divergences_are_exactly_the_documented_set() {
    let Some(jq) = jq_path() else {
        eprintln!("conformance: SKIPPING the jq comparison: jq is not on PATH");
        return;
    };

    let documented_lenient: Vec<&str> = JQ_ACCEPTS_WE_REJECT.iter().map(|(_, d, _)| *d).collect();
    let documented_strict: Vec<&str> = WE_ACCEPT_JQ_REJECTS.iter().map(|(_, d, _)| *d).collect();

    let mut undocumented = Vec::new();

    for (name, doc) in MUST_ACCEPT {
        if !jq_accepts(&jq, doc) && !documented_strict.contains(doc) {
            undocumented.push(format!("  jq REJECTS `{name}` which we accept: {doc}"));
        }
    }
    for (name, doc) in MUST_REJECT {
        if doc.trim().is_empty() {
            continue; // jq treats empty input as no input, not an error
        }
        if jq_accepts(&jq, doc) && !documented_lenient.contains(doc) {
            undocumented.push(format!("  jq ACCEPTS `{name}` which we reject: {doc}"));
        }
    }

    assert!(
        undocumented.is_empty(),
        "{} undocumented divergences from jq. Each is either a warpjq bug or \
         belongs in the README's \"Differences from jq\" section and in the \
         tables in this file:\n{}",
        undocumented.len(),
        undocumented.join("\n")
    );
}

#[test]
fn jq_is_still_lenient_exactly_where_we_say_it_is() {
    // If jq tightens up, these entries should move out of the README rather
    // than sit there claiming a difference that no longer exists.
    let Some(jq) = jq_path() else {
        eprintln!("conformance: SKIPPING: jq is not on PATH");
        return;
    };
    for (name, doc, why) in JQ_ACCEPTS_WE_REJECT {
        assert!(
            json::validate(doc.as_bytes()).is_err(),
            "warpjq should reject `{name}`: {doc}"
        );
        assert!(
            jq_accepts(&jq, doc),
            "jq no longer accepts `{name}` ({why}); the README is stale: {doc}"
        );
    }
}

#[test]
fn we_are_still_permissive_exactly_where_we_say_we_are() {
    let Some(jq) = jq_path() else {
        eprintln!("conformance: SKIPPING: jq is not on PATH");
        return;
    };
    for (name, doc, why) in WE_ACCEPT_JQ_REJECTS {
        assert!(
            json::validate(doc.as_bytes()).is_ok(),
            "warpjq should accept `{name}`: {doc}"
        );
        assert!(
            !jq_accepts(&jq, doc),
            "jq now accepts `{name}` ({why}); the README is stale: {doc}"
        );
    }
    // And the asymmetry that makes this list shorter than it looks: a lone
    // *low* surrogate is fine by jq, so it is not a divergence at all.
    assert!(jq_accepts(&jq, r#"{"a":"\ude00"}"#));
    assert!(json::validate(br#"{"a":"\ude00"}"#).is_ok());
}
