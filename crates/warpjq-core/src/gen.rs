//! Seeded NDJSON generators.
//!
//! The point of `warpjq gen` is reproducible benchmarks. Anyone can run the
//! same command on their own hardware and get a byte-identical file, so a
//! benchmark claim in the README is checkable without a 10 GB download. The
//! seed is part of the contract: changing a preset's output for a given seed
//! is a breaking change and needs a version bump.

use std::io::{self, Write};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Preset {
    /// nginx-style access logs: flat, short, mostly strings and small ints.
    Nginx,
    /// CloudTrail-ish audit events: deeply nested, long keys, some arrays.
    CloudTrail,
    /// Kubernetes pod logs: a nested `kubernetes` block plus a free-text message.
    K8s,
    /// Deliberately hostile: deep nesting, unicode, escapes, big numbers,
    /// long strings. This is the preset that makes benchmark numbers honest.
    Nested,
}

impl Preset {
    pub fn parse(s: &str) -> Option<Preset> {
        match s.to_ascii_lowercase().as_str() {
            "nginx" => Some(Preset::Nginx),
            "cloudtrail" => Some(Preset::CloudTrail),
            "k8s" | "kubernetes" => Some(Preset::K8s),
            "nested" | "worst" | "worstcase" => Some(Preset::Nested),
            _ => None,
        }
    }

    pub fn all() -> &'static [Preset] {
        &[
            Preset::Nginx,
            Preset::CloudTrail,
            Preset::K8s,
            Preset::Nested,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            Preset::Nginx => "nginx",
            Preset::CloudTrail => "cloudtrail",
            Preset::K8s => "k8s",
            Preset::Nested => "nested",
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Preset::Nginx => "flat access logs, ~180 B/line",
            Preset::CloudTrail => "nested audit events, ~700 B/line",
            Preset::K8s => "pod logs with a nested metadata block, ~400 B/line",
            Preset::Nested => "worst case: deep nesting, unicode, escapes, big numbers",
        }
    }

    /// A representative query for this preset, used by `--bench` and the docs.
    pub fn example_query(self) -> &'static str {
        match self {
            Preset::Nginx => "select(.status == 500) | count",
            Preset::CloudTrail => "select(.errorCode != null) | {t: .eventTime, e: .errorCode}",
            Preset::K8s => "group_by(.kubernetes.namespace) | count",
            Preset::Nested => "select(.a.b.c.d.value > 500) | count",
        }
    }
}

const HOSTS: &[&str] = &[
    "web-01", "web-02", "web-03", "api-01", "api-02", "cache-01", "db-01", "edge-07",
];
const PATHS: &[&str] = &[
    "/",
    "/index.html",
    "/api/v1/users",
    "/api/v1/orders",
    "/static/app.js",
    "/health",
    "/login",
    "/search?q=widgets",
    "/api/v1/orders/12345",
];
const METHODS: &[&str] = &["GET", "GET", "GET", "POST", "PUT", "DELETE"];
const AGENTS: &[&str] = &[
    "Mozilla/5.0 (X11; Linux x86_64)",
    "curl/8.4.0",
    "Go-http-client/2.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
];
const REGIONS: &[&str] = &["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-2"];
const NAMESPACES: &[&str] = &["default", "kube-system", "prod", "staging", "observability"];
const LEVELS: &[&str] = &["info", "info", "info", "warn", "error", "debug"];
const EVENTS: &[&str] = &[
    "AssumeRole",
    "GetObject",
    "PutObject",
    "CreateUser",
    "DeleteBucket",
    "DescribeInstances",
];
const ERRORS: &[&str] = &["AccessDenied", "ThrottlingException", "ValidationError"];

/// Weighted status codes: mostly 200, with enough 500s to make
/// `select(.status == 500)` return a believable number of rows.
fn status(rng: &mut ChaCha8Rng) -> u16 {
    match rng.gen_range(0..1000) {
        0..=879 => 200,
        880..=929 => 304,
        930..=959 => 404,
        960..=979 => 301,
        980..=989 => 403,
        _ => 500,
    }
}

fn pick<'a>(rng: &mut ChaCha8Rng, xs: &[&'a str]) -> &'a str {
    xs[rng.gen_range(0..xs.len())]
}

/// Writes lines until at least `target_bytes` have been produced.
///
/// The file ends on a line boundary, so the result is always valid NDJSON and
/// the size lands slightly over the target rather than truncating a line.
pub fn generate<W: Write>(
    out: &mut W,
    preset: Preset,
    target_bytes: u64,
    seed: u64,
) -> io::Result<GenStats> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 20);
    let mut written: u64 = 0;
    let mut lines: u64 = 0;
    let mut ts: u64 = 1_700_000_000;

    while written < target_bytes {
        let before = buf.len();
        match preset {
            Preset::Nginx => nginx_line(&mut buf, &mut rng, ts),
            Preset::CloudTrail => cloudtrail_line(&mut buf, &mut rng, ts),
            Preset::K8s => k8s_line(&mut buf, &mut rng, ts),
            Preset::Nested => nested_line(&mut buf, &mut rng, ts),
        }
        buf.push(b'\n');
        written += (buf.len() - before) as u64;
        lines += 1;
        // Timestamps advance monotonically with jitter, like a real log.
        ts += rng.gen_range(0..3);

        if buf.len() >= (1 << 20) {
            out.write_all(&buf)?;
            buf.clear();
        }
    }
    out.write_all(&buf)?;
    out.flush()?;
    Ok(GenStats {
        bytes: written,
        lines,
    })
}

#[derive(Copy, Clone, Debug)]
pub struct GenStats {
    pub bytes: u64,
    pub lines: u64,
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(s.as_bytes());
}

fn push_num(buf: &mut Vec<u8>, n: u64) {
    let mut b = itoa::Buffer::new();
    buf.extend_from_slice(b.format(n).as_bytes());
}

fn push_i(buf: &mut Vec<u8>, n: i64) {
    let mut b = itoa::Buffer::new();
    buf.extend_from_slice(b.format(n).as_bytes());
}

fn nginx_line(buf: &mut Vec<u8>, rng: &mut ChaCha8Rng, ts: u64) {
    push_str(buf, r#"{"ts":"#);
    push_num(buf, ts);
    push_str(buf, r#","remote":""#);
    push_num(buf, rng.gen_range(1..255));
    buf.push(b'.');
    push_num(buf, rng.gen_range(0..255));
    buf.push(b'.');
    push_num(buf, rng.gen_range(0..255));
    buf.push(b'.');
    push_num(buf, rng.gen_range(1..255));
    push_str(buf, r#"","method":""#);
    push_str(buf, pick(rng, METHODS));
    push_str(buf, r#"","path":""#);
    push_str(buf, pick(rng, PATHS));
    push_str(buf, r#"","status":"#);
    push_num(buf, status(rng) as u64);
    push_str(buf, r#","bytes":"#);
    push_num(buf, rng.gen_range(0..2_000_000));
    push_str(buf, r#","duration_ms":"#);
    push_num(buf, rng.gen_range(0..5000));
    push_str(buf, r#","host":""#);
    push_str(buf, pick(rng, HOSTS));
    push_str(buf, r#"","agent":""#);
    push_str(buf, pick(rng, AGENTS));
    push_str(buf, r#""}"#);
}

fn cloudtrail_line(buf: &mut Vec<u8>, rng: &mut ChaCha8Rng, ts: u64) {
    let has_error = rng.gen_range(0..100) < 7;
    push_str(buf, r#"{"eventVersion":"1.08","eventTime":"#);
    push_num(buf, ts);
    push_str(buf, r#","eventName":""#);
    push_str(buf, pick(rng, EVENTS));
    push_str(buf, r#"","awsRegion":""#);
    push_str(buf, pick(rng, REGIONS));
    push_str(buf, r#"","sourceIPAddress":"10."#);
    push_num(buf, rng.gen_range(0..255));
    buf.push(b'.');
    push_num(buf, rng.gen_range(0..255));
    buf.push(b'.');
    push_num(buf, rng.gen_range(1..255));
    push_str(
        buf,
        r#"","userIdentity":{"type":"AssumedRole","principalId":"AROA"#,
    );
    push_num(buf, rng.gen_range(100000..999999));
    push_str(buf, r#"","arn":"arn:aws:sts::"#);
    push_num(buf, rng.gen_range(100000000000u64..999999999999));
    push_str(buf, r#":assumed-role/svc","accountId":""#);
    push_num(buf, rng.gen_range(100000000000u64..999999999999));
    push_str(
        buf,
        r#"","sessionContext":{"attributes":{"mfaAuthenticated":""#,
    );
    push_str(buf, if rng.gen_bool(0.3) { "true" } else { "false" });
    push_str(buf, r#"","creationDate":"#);
    push_num(buf, ts - rng.gen_range(0..3600));
    push_str(buf, r#"}}},"requestParameters":{"bucketName":"bucket-"#);
    push_num(buf, rng.gen_range(0..64));
    push_str(buf, r#"","key":"data/part-"#);
    push_num(buf, rng.gen_range(0..100000));
    push_str(buf, r#".parquet"},"responseElements":null,"readOnly":"#);
    push_str(buf, if rng.gen_bool(0.7) { "true" } else { "false" });
    push_str(
        buf,
        r#","resources":[{"type":"AWS::S3::Object","ARN":"arn:aws:s3:::b/"#,
    );
    push_num(buf, rng.gen_range(0..100000));
    push_str(buf, r#""}],"errorCode":"#);
    if has_error {
        buf.push(b'"');
        push_str(buf, pick(rng, ERRORS));
        buf.push(b'"');
    } else {
        push_str(buf, "null");
    }
    push_str(buf, r#","bytesTransferred":"#);
    push_num(buf, rng.gen_range(0..50_000_000));
    buf.push(b'}');
}

fn k8s_line(buf: &mut Vec<u8>, rng: &mut ChaCha8Rng, ts: u64) {
    let ns = pick(rng, NAMESPACES);
    push_str(buf, r#"{"time":"#);
    push_num(buf, ts);
    push_str(buf, r#","level":""#);
    push_str(buf, pick(rng, LEVELS));
    push_str(buf, r#"","message":"handled request in "#);
    push_num(buf, rng.gen_range(0..3000));
    push_str(buf, r#"ms","status":"#);
    push_num(buf, status(rng) as u64);
    push_str(buf, r#","bytes":"#);
    push_num(buf, rng.gen_range(0..500_000));
    push_str(buf, r#","kubernetes":{"namespace":""#);
    push_str(buf, ns);
    push_str(buf, r#"","pod_name":""#);
    push_str(buf, ns);
    push_str(buf, "-");
    push_num(buf, rng.gen_range(0..40));
    push_str(buf, "-");
    push_num(buf, rng.gen_range(100000..999999));
    push_str(buf, r#"","container_name":"app","node":""#);
    push_str(buf, pick(rng, HOSTS));
    push_str(buf, r#"","labels":{"app":"svc-"#);
    push_num(buf, rng.gen_range(0..12));
    push_str(buf, r#"","tier":"backend"}},"trace_id":""#);
    for _ in 0..4 {
        push_num(buf, rng.gen_range(100000000u64..999999999));
    }
    push_str(buf, r#""}"#);
}

/// Every awkward case in one preset: 5 levels of nesting, non-ASCII keys and
/// values, embedded quotes and backslashes, an integer past 2^53, a float
/// written in exponent form, an empty object, an empty array, and `null`.
fn nested_line(buf: &mut Vec<u8>, rng: &mut ChaCha8Rng, ts: u64) {
    push_str(buf, r#"{"ts":"#);
    push_num(buf, ts);
    push_str(buf, r#","a":{"b":{"c":{"d":{"value":"#);
    push_num(buf, rng.gen_range(0..1000));
    push_str(buf, r#","tag":"café-"#);
    push_num(buf, rng.gen_range(0..100));
    push_str(buf, r#"","deep":{"e":[1,2,{"f":"#);
    push_num(buf, rng.gen_range(0..10));
    push_str(
        buf,
        r#"}]}}}}},"unicode":"日本語 🚀 ☃","escaped":"quote\" backslash\\ tab\t newline\n","big":"#,
    );
    // Past 2^53: any DOM-based tool that round-trips through f64 loses digits
    // here, which is exactly what we want to catch.
    push_num(buf, 9_007_199_254_740_993 + rng.gen_range(0..1000));
    push_str(buf, r#","sci":"#);
    push_str(
        buf,
        if rng.gen_bool(0.5) {
            "1.5e-7"
        } else {
            "-2.5E+10"
        },
    );
    push_str(
        buf,
        r#","empty_obj":{},"empty_arr":[],"nothing":null,"списки":["#,
    );
    for i in 0..rng.gen_range(0..5) {
        if i > 0 {
            buf.push(b',');
        }
        push_i(buf, rng.gen_range(-1000..1000));
    }
    push_str(buf, r#"],"host":""#);
    push_str(buf, pick(rng, HOSTS));
    push_str(buf, r#"","status":"#);
    push_num(buf, status(rng) as u64);
    buf.push(b'}');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    fn gen(preset: Preset, bytes: u64, seed: u64) -> Vec<u8> {
        let mut out = Vec::new();
        generate(&mut out, preset, bytes, seed).unwrap();
        out
    }

    #[test]
    fn every_preset_emits_valid_ndjson() {
        for &p in Preset::all() {
            let data = gen(p, 200_000, 42);
            let mut n = 0;
            for line in data.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                json::validate(line).unwrap_or_else(|e| {
                    panic!(
                        "{} produced invalid JSON: {e}\n{}",
                        p.name(),
                        String::from_utf8_lossy(line)
                    )
                });
                n += 1;
            }
            assert!(n > 10, "{} produced only {n} lines", p.name());
        }
    }

    #[test]
    fn output_ends_on_a_line_boundary() {
        for &p in Preset::all() {
            let data = gen(p, 50_000, 7);
            assert_eq!(*data.last().unwrap(), b'\n', "{}", p.name());
        }
    }

    #[test]
    fn the_same_seed_reproduces_the_same_bytes() {
        // This is the property the README's benchmark instructions depend on.
        for &p in Preset::all() {
            assert_eq!(gen(p, 100_000, 1234), gen(p, 100_000, 1234), "{}", p.name());
        }
    }

    #[test]
    fn different_seeds_produce_different_data() {
        assert_ne!(gen(Preset::Nginx, 50_000, 1), gen(Preset::Nginx, 50_000, 2));
    }

    #[test]
    fn generated_size_meets_the_target() {
        let data = gen(Preset::Nginx, 100_000, 9);
        assert!(data.len() as u64 >= 100_000);
        // Overshoot is bounded by one line.
        assert!((data.len() as u64) < 100_000 + 4096);
    }

    #[test]
    fn the_nested_preset_actually_contains_the_hard_cases() {
        let data = String::from_utf8(gen(Preset::Nested, 100_000, 3)).unwrap();
        assert!(
            data.contains("9007199254740"),
            "missing a past-2^53 integer"
        );
        assert!(data.contains(r#"\""#), "missing an escaped quote");
        assert!(data.contains("🚀"), "missing non-ASCII content");
        assert!(data.contains("{}"), "missing an empty object");
        assert!(data.contains("null"), "missing a null");
        assert!(data.contains("списки"), "missing a non-ASCII key");
    }

    #[test]
    fn example_queries_compile() {
        for &p in Preset::all() {
            crate::query::parse(p.example_query()).unwrap_or_else(|e| panic!("{}: {e}", p.name()));
        }
    }
}
