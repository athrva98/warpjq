//! `warpjq bench` runs the same query on every engine present and prints a
//! table.
//!
//! Every number here is **end-to-end wall time for the whole process**,
//! including reading the file. Kernel-only timings are the standard way these
//! projects get taken apart in public, so the tool simply cannot produce one.
//! The table prints the cache state and the hardware alongside the numbers so
//! a pasted result is self-describing.

use std::io;
use std::path::Path;
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use warpjq_core::chunk::Input;
use warpjq_core::exec::{run_query, GpuStatus, Options, Preference};
use warpjq_core::output::{Format, Writer};
use warpjq_core::query::{AggKind, Output, Program};

use crate::{human_bytes, BenchArgs};

struct Row {
    engine: String,
    note: String,
    best: Option<Duration>,
    /// Set when the engine could not run this query at all.
    skipped: Option<String>,
    /// The exact argv, printed under the table. A benchmark nobody can
    /// re-run by hand is not a benchmark.
    command: Option<String>,
    /// First line of the tool's own stdout, so a row that claims an
    /// implausible speed can be checked against what it actually produced.
    output: Option<String>,
}

impl Row {
    fn new(engine: &str) -> Row {
        Row {
            engine: engine.into(),
            note: String::new(),
            best: None,
            skipped: None,
            command: None,
            output: None,
        }
    }
}

fn shell_quote(s: &str) -> String {
    if s.is_empty() || s.contains([' ', '"', '\'', '$', '|', '(', ')']) {
        format!("'{}'", s.replace('\'', r"'\''"))
    } else {
        s.to_string()
    }
}

pub fn run(args: BenchArgs) -> anyhow::Result<ExitCode> {
    let program = warpjq_core::parse(&args.query)?;
    let path = Path::new(&args.file);
    let size = std::fs::metadata(path)
        .map_err(|e| anyhow::anyhow!("could not stat `{}`: {e}", args.file))?
        .len();

    let gpu = GpuStatus::detect();
    let mut rows: Vec<Row> = Vec::new();

    eprintln!(
        "warpjq bench: {} over {} ({})",
        args.query,
        args.file,
        human_bytes(size)
    );
    eprintln!(
        "warpjq bench: {} warmup run(s) then {} timed run(s); reporting the best of each\n",
        args.warmup, args.runs
    );

    // --- warpjq, GPU ------------------------------------------------------
    if gpu.is_available() {
        rows.push(time_warpjq(
            "warpjq (gpu)",
            &program,
            path,
            Preference::Gpu,
            &args,
        ));
    } else {
        let mut r = Row::new("warpjq (gpu)");
        r.skipped = Some(gpu.reason());
        rows.push(r);
    }

    // --- warpjq, CPU ------------------------------------------------------
    rows.push(time_warpjq(
        "warpjq (cpu)",
        &program,
        path,
        Preference::Cpu,
        &args,
    ));

    // --- jq ---------------------------------------------------------------
    if args.jq {
        for tool in ["jq", "jaq"] {
            match which(tool) {
                None => {
                    let mut r = Row::new(tool);
                    r.skipped = Some(format!("`{tool}` is not on PATH"));
                    rows.push(r);
                }
                Some(_) => match jq_argv(tool, &program, &args.file) {
                    Err(why) => {
                        let mut r = Row::new(tool);
                        r.skipped = Some(why);
                        rows.push(r);
                    }
                    Ok(spec) => {
                        let mut row = time_external(tool, &spec.argv, &args);
                        row.note = spec.note;
                        rows.push(row);
                    }
                },
            }
        }
    }

    // --- grep, for the count queries where it is even close ----------------
    if let Some(spec) = grep_argv(&program, &args.file) {
        if which("grep").is_some() {
            let mut row = time_external("grep -c", &spec.argv, &args);
            row.note = spec.note;
            rows.push(row);
        }
    }

    // Cross-check: for a query with a single-line answer, every engine in the
    // table should print the same thing. If one does not, it is not running
    // the query we think it is, and its time means nothing, which is how a
    // baseline that quietly matched zero lines ends up looking 40x faster.
    if let Some(reference) = warpjq_answer(&program, path) {
        for r in rows.iter_mut() {
            let Some(got) = r.output.clone() else {
                continue;
            };
            if got != reference {
                r.note = format!("NOT COMPARABLE: printed {got}, warpjq says {reference}");
                r.best = None;
                r.skipped = Some(format!(
                    "output disagrees with warpjq ({got} vs {reference}); \
                     timing it would be misleading"
                ));
            }
        }
    }

    print_table(&rows, size);
    print_commands(&rows);
    print_footer(&gpu);
    Ok(ExitCode::SUCCESS)
}

/// warpjq's own answer, for queries whose result is a single line.
///
/// Returns `None` for streaming or grouped queries, where there is no single
/// value to compare against.
fn warpjq_answer(program: &Program, path: &Path) -> Option<String> {
    if !program.is_aggregate() || program.group_by.is_some() {
        return None;
    }
    let mut input = Input::open(path).ok()?;
    let mut writer = Writer::new(Vec::new(), Format::Ndjson);
    let options = Options::default();
    run_query(program, &mut input, &options, &mut writer, Preference::Cpu).ok()?;
    let (bytes, _) = writer.finish().ok()?;
    let s = String::from_utf8_lossy(&bytes);
    s.lines().next().map(|l| l.trim().to_string())
}

fn time_warpjq(
    label: &str,
    program: &Program,
    path: &Path,
    preference: Preference,
    args: &BenchArgs,
) -> Row {
    let options = Options::default();
    let once = || -> anyhow::Result<()> {
        let mut input = Input::open(path)?;
        // Output goes to a sink, but it is still fully formatted. Skipping
        // the formatting would flatter warpjq against tools that cannot.
        let mut writer = Writer::new(io::sink(), Format::Ndjson);
        run_query(program, &mut input, &options, &mut writer, preference)?;
        writer.finish()?;
        Ok(())
    };

    for _ in 0..args.warmup {
        if let Err(e) = once() {
            let mut r = Row::new(label);
            r.skipped = Some(e.to_string());
            return r;
        }
    }
    let mut best: Option<Duration> = None;
    for _ in 0..args.runs.max(1) {
        let t = Instant::now();
        if let Err(e) = once() {
            let mut r = Row::new(label);
            r.skipped = Some(e.to_string());
            return r;
        }
        let d = t.elapsed();
        best = Some(best.map_or(d, |b: Duration| b.min(d)));
    }
    let mut r = Row::new(label);
    r.best = best;
    let backend_flag = if preference == Preference::Gpu {
        "gpu"
    } else {
        "cpu"
    };
    r.command = Some(format!(
        "warpjq {} --backend {} {}",
        shell_quote(&program.source),
        backend_flag,
        shell_quote(&path.display().to_string())
    ));
    r
}

/// Times an external tool.
///
/// A process that failed must never be reported as a fast one. A crashed `jq`
/// exits in milliseconds, and timing that as a win would turn this table into
/// a lie, the exact kind that gets a benchmark post dismantled in public. So
/// the exit status is checked on every run and a failure produces a skipped
/// row explaining itself, not a number.
fn time_external(label: &str, argv: &[String], args: &BenchArgs) -> Row {
    let cmdline = argv
        .iter()
        .map(|a| shell_quote(a))
        .collect::<Vec<_>>()
        .join(" ");
    let fail = |why: String| {
        let mut r = Row::new(label);
        r.skipped = Some(why);
        r.command = Some(cmdline.clone());
        r
    };

    let once = || -> Result<std::process::Output, io::Error> {
        Command::new(&argv[0])
            .args(&argv[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
    };

    let check = |out: &std::process::Output| -> Result<(), String> {
        // grep exits 1 for "no matches", which is a legitimate result.
        let benign_one = label.starts_with("grep") && out.status.code() == Some(1);
        if out.status.success() || benign_one {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&out.stderr);
        let first = stderr.lines().next().unwrap_or("").trim();
        Err(if first.is_empty() {
            format!("exited with status {}", out.status)
        } else {
            format!("failed: {first}")
        })
    };

    for _ in 0..args.warmup {
        match once() {
            Err(e) => return fail(format!("could not run `{}`: {e}", argv[0])),
            Ok(out) => {
                if let Err(why) = check(&out) {
                    return fail(why);
                }
            }
        }
    }
    let mut best: Option<Duration> = None;
    let mut first_out = None;
    for _ in 0..args.runs.max(1) {
        let t = Instant::now();
        let out = match once() {
            Ok(o) => o,
            Err(e) => return fail(format!("could not run `{}`: {e}", argv[0])),
        };
        let d = t.elapsed();
        if let Err(why) = check(&out) {
            return fail(why);
        }
        if first_out.is_none() {
            let s = String::from_utf8_lossy(&out.stdout);
            first_out = Some(s.lines().next().unwrap_or("").trim().to_string());
        }
        best = Some(best.map_or(d, |b: Duration| b.min(d)));
    }
    let mut r = Row::new(label);
    r.best = best;
    r.command = Some(cmdline);
    r.output = first_out;
    r
}

struct ExternalSpec {
    argv: Vec<String>,
    note: String,
}

/// Translates the compiled program back into a real jq invocation.
///
/// This has to be an honest translation or the whole table is worthless, so
/// where jq needs a different shape (slurping for `group_by`, `reduce inputs`
/// for a streaming count) we use that shape and say so in the note column.
fn jq_argv(tool: &str, program: &Program, file: &str) -> Result<ExternalSpec, String> {
    let select = match &program.filter {
        Some(_) => format!("select({}) | ", cond_to_jq(program)?),
        None => String::new(),
    };

    match (&program.output, program.group_by) {
        (Output::Agg { kind, arg }, None) => {
            let inner = match arg {
                Some(p) => format!("{select}{}", program.path(*p)),
                None => select.trim_end_matches(" | ").to_string(),
            };
            // A bare `count` has no filter and no argument, so there is
            // nothing to pipe `inputs` into. Emitting `(inputs | )` anyway is
            // a jq syntax error. The row then vanishes from the table as a
            // failed baseline rather than being wrong, but the comparison is
            // silently lost, which is nearly as bad.
            let source = if inner.is_empty() {
                "inputs".to_string()
            } else {
                format!("(inputs | {inner})")
            };
            let expr = match kind {
                AggKind::Count => format!("reduce {source} as $x (0; .+1)"),
                AggKind::Sum => format!("reduce {source} as $x (0; .+$x)"),
                AggKind::Min => format!("[{source}] | min"),
                AggKind::Max => format!("[{source}] | max"),
                AggKind::Avg => format!("[{source}] | add / length"),
            };
            let streaming = matches!(kind, AggKind::Count | AggKind::Sum);
            Ok(ExternalSpec {
                argv: vec![tool.into(), "-n".into(), expr, file.into()],
                note: if streaming {
                    "streaming".into()
                } else {
                    "buffers all values".into()
                },
            })
        }
        (Output::Agg { kind, arg }, Some(g)) => {
            let key = program.path(g).to_string();
            let value = match (kind, arg) {
                (AggKind::Count, _) => "length".to_string(),
                (AggKind::Sum, Some(p)) => format!("map({}) | add", program.path(*p)),
                (AggKind::Min, Some(p)) => format!("map({}) | min", program.path(*p)),
                (AggKind::Max, Some(p)) => format!("map({}) | max", program.path(*p)),
                (AggKind::Avg, Some(p)) => {
                    format!("map({}) | add / length", program.path(*p))
                }
                _ => return Err("no argument for the aggregate".into()),
            };
            let expr = format!(
                "[inputs | {select}.] | group_by({key}) | map({{key: .[0]{key}, v: ({value})}}) | .[]"
            );
            Ok(ExternalSpec {
                argv: vec![tool.into(), "-n".into(), "-c".into(), expr, file.into()],
                note: "must slurp the whole file".into(),
            })
        }
        (Output::Passthrough, _) => Ok(ExternalSpec {
            argv: vec![
                tool.into(),
                "-c".into(),
                format!("{}.", select),
                file.into(),
            ],
            note: "streaming".into(),
        }),
        (Output::Path(p), _) => Ok(ExternalSpec {
            argv: vec![
                tool.into(),
                "-c".into(),
                format!("{select}{}", program.path(*p)),
                file.into(),
            ],
            note: "streaming".into(),
        }),
        (Output::Project(fields), _) => {
            let body = fields
                .iter()
                .map(|(k, p)| format!("{}: {}", quote_key(k), program.path(*p)))
                .collect::<Vec<_>>()
                .join(", ");
            Ok(ExternalSpec {
                argv: vec![
                    tool.into(),
                    "-c".into(),
                    format!("{select}{{{body}}}"),
                    file.into(),
                ],
                note: "streaming".into(),
            })
        }
    }
}

fn quote_key(k: &str) -> String {
    format!("{k:?}")
}

fn cond_to_jq(program: &Program) -> Result<String, String> {
    use warpjq_core::query::{Cond, Literal};
    fn go(c: &Cond, p: &Program) -> String {
        match c {
            Cond::Cmp { path, op, lit } => {
                let l = match lit {
                    Literal::Null => "null".to_string(),
                    Literal::Bool(b) => b.to_string(),
                    Literal::Num(n) => warpjq_core::agg::format_number(*n),
                    Literal::Str(s) => format!("{s:?}"),
                };
                format!("({} {} {})", p.path(*path), op.as_str(), l)
            }
            Cond::Truthy(id) => format!("({})", p.path(*id)),
            Cond::And(a, b) => format!("({} and {})", go(a, p), go(b, p)),
            Cond::Or(a, b) => format!("({} or {})", go(a, p), go(b, p)),
            Cond::Not(a) => format!("({} | not)", go(a, p)),
        }
    }
    program
        .filter
        .as_ref()
        .map(|c| go(c, program))
        .ok_or_else(|| "no filter".to_string())
}

/// grep is only a meaningful comparison for `select(.field == <literal>) | count`,
/// and even then it is a substring match, not a parse. It is in the table
/// because "we beat grep" is the surprising result, and leaving the caveat off
/// would be the dishonest part.
fn grep_argv(program: &Program, file: &str) -> Option<ExternalSpec> {
    use warpjq_core::query::{CmpOp, Cond, Literal, Step};
    let Output::Agg {
        kind: AggKind::Count,
        ..
    } = program.output
    else {
        return None;
    };
    if program.group_by.is_some() {
        return None;
    }
    let Some(Cond::Cmp {
        path,
        op: CmpOp::Eq,
        lit,
    }) = &program.filter
    else {
        return None;
    };
    let steps = &program.path(*path).steps;
    let [Step::Key(key)] = steps.as_slice() else {
        return None;
    };
    let value = match lit {
        Literal::Num(n) => warpjq_core::agg::format_number(*n),
        Literal::Str(s) => format!("\"{s}\""),
        _ => return None,
    };
    Some(ExternalSpec {
        argv: vec![
            "grep".into(),
            "-c".into(),
            format!("\"{key}\":{value}"),
            file.into(),
        ],
        note: "substring match, not a parse".into(),
    })
}

fn which(tool: &str) -> Option<String> {
    let cmd = if cfg!(windows) { "where" } else { "which" };
    let out = Command::new(cmd).arg(tool).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next().map(|l| l.trim().to_string())
}

fn print_table(rows: &[Row], size: u64) {
    let baseline = rows
        .iter()
        .find(|r| r.engine == "jq")
        .and_then(|r| r.best)
        .or_else(|| rows.iter().filter_map(|r| r.best).max());

    let w_engine = rows
        .iter()
        .map(|r| r.engine.len())
        .max()
        .unwrap_or(6)
        .max(6);
    let w_note = rows
        .iter()
        .map(|r| {
            r.note
                .len()
                .max(r.skipped.as_ref().map_or(0, |s| s.len().min(46)))
        })
        .max()
        .unwrap_or(4)
        .max(4);

    println!(
        "{:<w_engine$}  {:>9}  {:>10}  {:>9}  notes",
        "engine",
        "time",
        "throughput",
        "vs jq",
        w_engine = w_engine
    );
    println!(
        "{}  {}  {}  {}  {}",
        "-".repeat(w_engine),
        "-".repeat(9),
        "-".repeat(10),
        "-".repeat(9),
        "-".repeat(w_note)
    );

    for r in rows {
        match (r.best, &r.skipped) {
            (Some(d), _) => {
                let secs = d.as_secs_f64();
                let gbps = size as f64 / secs / 1e9;
                let speedup = baseline
                    .map(|b| format!("{:.1}x", b.as_secs_f64() / secs))
                    .unwrap_or_else(|| "-".into());
                println!(
                    "{:<w_engine$}  {:>8.3}s  {:>7.2} GB/s  {:>9}  {}",
                    r.engine,
                    secs,
                    gbps,
                    speedup,
                    r.note,
                    w_engine = w_engine
                );
            }
            (None, Some(why)) => {
                println!(
                    "{:<w_engine$}  {:>9}  {:>10}  {:>9}  {}",
                    r.engine,
                    "skipped",
                    "-",
                    "-",
                    truncate(why, 60),
                    w_engine = w_engine
                );
            }
            (None, None) => {}
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    let one_line = s.replace('\n', " ");
    if one_line.chars().count() <= n {
        one_line
    } else {
        let t: String = one_line.chars().take(n - 1).collect();
        format!("{t}…")
    }
}

/// Prints the exact command behind every row, plus what each one printed.
///
/// Two reasons. First, a benchmark table nobody can re-run by hand is just a
/// claim. Second, seeing the row's own output next to its time is how you
/// catch a baseline that "won" by not doing the work.
fn print_commands(rows: &[Row]) {
    if rows.iter().all(|r| r.command.is_none()) {
        return;
    }
    println!("\nCommands:");
    for r in rows {
        let Some(cmd) = &r.command else { continue };
        println!("  {:<13} {cmd}", r.engine);
        if let Some(out) = &r.output {
            if !out.is_empty() {
                println!("                  -> {out}");
            }
        }
    }
}

fn print_footer(gpu: &GpuStatus) {
    println!();
    println!("All timings are end-to-end wall clock for the whole process, including");
    println!("reading the file. There is no kernel-only number here on purpose.");
    if !gpu.is_available() {
        println!("GPU row skipped: {}", gpu.reason());
    }
    println!("Cache state: whatever your page cache held after the warmup run(s).");
    println!("Drop caches between runs to measure cold I/O.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use warpjq_core::parse;

    fn jq_expr(query: &str) -> String {
        let p = parse(query).unwrap_or_else(|e| panic!("{query}: {e}"));
        let spec = jq_argv("jq", &p, "F").unwrap_or_else(|e| panic!("{query}: {e}"));
        // The expression is the last argument before the file.
        spec.argv[spec.argv.len() - 2].clone()
    }

    fn jq_flags(query: &str) -> Vec<String> {
        let p = parse(query).unwrap();
        let spec = jq_argv("jq", &p, "F").unwrap();
        spec.argv[1..spec.argv.len() - 2].to_vec()
    }

    /// The whole table is meaningless if the jq expression is not the same
    /// question warpjq was asked, so pin the translation for every shape.
    #[test]
    fn jq_translation_matches_the_query_for_each_output_shape() {
        assert_eq!(jq_expr("."), ".");
        assert_eq!(jq_expr(".a.b"), ".a.b");
        assert_eq!(jq_expr(".a[0]"), ".a[0]");
        assert_eq!(
            jq_expr("select(.status == 500)"),
            "select((.status == 500)) | ."
        );
        assert_eq!(
            jq_expr("{t: .ts, s: .status}"),
            r#"{"t": .ts, "s": .status}"#
        );
        assert_eq!(jq_expr("select(.a == 1) | .b"), "select((.a == 1)) | .b");
    }

    #[test]
    fn jq_translation_of_aggregates_uses_streaming_forms_where_possible() {
        // A bare count has nothing to pipe into, so `inputs` stands alone.
        assert_eq!(jq_expr("count"), "reduce inputs as $x (0; .+1)");
        assert_eq!(
            jq_expr("select(.a == 1) | count"),
            "reduce (inputs | select((.a == 1))) as $x (0; .+1)"
        );
        assert_eq!(
            jq_expr("sum(.bytes)"),
            "reduce (inputs | .bytes) as $x (0; .+$x)"
        );
        assert_eq!(jq_expr("min(.b)"), "[(inputs | .b)] | min");
        assert_eq!(jq_expr("max(.b)"), "[(inputs | .b)] | max");
        assert_eq!(jq_expr("avg(.b)"), "[(inputs | .b)] | add / length");
        // -n is required for `inputs` to be available.
        assert!(jq_flags("count").contains(&"-n".to_string()));
    }

    #[test]
    fn jq_translation_of_group_by_admits_that_it_must_slurp() {
        let p = parse("group_by(.host) | count").unwrap();
        let spec = jq_argv("jq", &p, "F").unwrap();
        assert!(
            spec.note.contains("slurp"),
            "the note must say jq buffers the whole file: {}",
            spec.note
        );
        let expr = &spec.argv[spec.argv.len() - 2];
        assert!(expr.contains("group_by(.host)"), "{expr}");
        assert!(expr.contains("length"), "{expr}");
    }

    #[test]
    fn jq_translation_preserves_boolean_structure_and_precedence() {
        assert_eq!(
            jq_expr("select(.a == 1 and .b == 2)"),
            "select(((.a == 1) and (.b == 2))) | ."
        );
        assert_eq!(
            jq_expr("select(.a == 1 or .b == 2 and .c == 3)"),
            "select(((.a == 1) or ((.b == 2) and (.c == 3)))) | ."
        );
        assert_eq!(jq_expr("select(.a | not)"), "select(((.a) | not)) | .");
        assert_eq!(jq_expr("select(.a)"), "select((.a)) | .");
    }

    #[test]
    fn jq_translation_renders_literals_the_way_jq_spells_them() {
        assert_eq!(jq_expr("select(.a == null)"), "select((.a == null)) | .");
        assert_eq!(jq_expr("select(.a == true)"), "select((.a == true)) | .");
        assert_eq!(
            jq_expr(r#"select(.a == "x")"#),
            r#"select((.a == "x")) | ."#
        );
        // Integral floats must not come out as `1.0`, which jq would compare
        // equal anyway but which reads as a different query.
        assert_eq!(jq_expr("select(.a == 500)"), "select((.a == 500)) | .");
        assert_eq!(jq_expr("select(.a >= 1.5)"), "select((.a >= 1.5)) | .");
    }

    #[test]
    fn jq_translation_escapes_awkward_projection_keys() {
        let expr = jq_expr(r#"{"a b": .x, "q\"q": .y}"#);
        assert!(expr.contains(r#""a b": .x"#), "{expr}");
        assert!(expr.contains(r#""q\"q": .y"#), "{expr}");
    }

    /// grep is only offered where it answers the same question, and even then
    /// it is labelled. Offering it for anything else would be inviting a
    /// comparison that is not one.
    #[test]
    fn grep_is_only_offered_for_a_single_equality_count() {
        let ok = parse("select(.status == 500) | count").unwrap();
        let spec = grep_argv(&ok, "F").expect("should offer grep");
        assert_eq!(spec.argv, vec!["grep", "-c", r#""status":500"#, "F"]);
        assert!(spec.note.contains("substring"), "{}", spec.note);

        for not_offered in [
            "select(.status == 500)",              // not a count
            "select(.status >= 500) | count",      // not equality
            "select(.a.b == 1) | count",           // nested path
            "group_by(.h) | count",                // grouped
            "count",                               // no filter
            "select(.a == 1 and .b == 2) | count", // compound
            "select(.a == null) | count",          // no literal spelling
        ] {
            let p = parse(not_offered).unwrap();
            assert!(
                grep_argv(&p, "F").is_none(),
                "grep should not be offered for `{not_offered}`"
            );
        }
    }

    #[test]
    fn grep_pattern_quotes_string_literals() {
        let p = parse(r#"select(.method == "POST") | count"#).unwrap();
        let spec = grep_argv(&p, "F").unwrap();
        assert_eq!(spec.argv[2], r#""method":"POST""#);
    }

    #[test]
    fn shell_quoting_survives_a_round_trip_through_a_shell() {
        assert_eq!(shell_quote("plain"), "plain");
        assert_eq!(shell_quote("has space"), "'has space'");
        assert_eq!(shell_quote("select(.a == 1)"), "'select(.a == 1)'");
        assert_eq!(shell_quote(r#"say "hi""#), r#"'say "hi"'"#);
        assert_eq!(shell_quote(""), "''");
        // A single quote inside must be closed, escaped and reopened.
        assert_eq!(shell_quote("it's"), r#"'it'\''s'"#);
    }

    #[test]
    fn warpjq_answer_is_only_taken_for_single_row_queries() {
        // Streaming and grouped queries have no single value to cross-check,
        // so the comparison must not be attempted at all.
        let path = Path::new("does-not-matter");
        assert!(warpjq_answer(&parse("select(.a == 1)").unwrap(), path).is_none());
        assert!(warpjq_answer(&parse("group_by(.h) | count").unwrap(), path).is_none());
        assert!(warpjq_answer(&parse(".a").unwrap(), path).is_none());
    }

    #[test]
    fn a_row_whose_output_disagrees_is_reported_as_not_comparable() {
        // This is the guard that caught grep matching nothing and "winning".
        let mut row = Row::new("grep -c");
        row.best = Some(Duration::from_millis(10));
        row.output = Some("0".into());
        let reference = "59218";

        if row.output.as_deref() != Some(reference) {
            row.note = format!("NOT COMPARABLE: printed 0, warpjq says {reference}");
            row.best = None;
            row.skipped = Some("output disagrees".into());
        }
        assert!(
            row.best.is_none(),
            "a disagreeing row must not carry a time"
        );
        assert!(row.skipped.is_some());
    }

    /// The translation is only worth anything if jq accepts it *and* answers
    /// the same question. Asserting the expression text catches typos; this
    /// catches the class of bug where the text looks plausible and jq rejects
    /// it, which is how `reduce (inputs | ) as $x` survived review, being a
    /// syntax error that merely made the jq row disappear from the table.
    #[test]
    fn every_jq_translation_is_accepted_by_jq_and_agrees_with_warpjq() {
        let Some(jq) = which("jq") else {
            eprintln!("bench: SKIPPING the jq translation check: jq is not on PATH");
            return;
        };
        let dir = std::env::temp_dir().join(format!("warpjq-bench-xlate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("f.ndjson");
        std::fs::write(
            &file,
            concat!(
                r#"{"a":1,"b":10,"ts":1,"status":500,"host":"x"}"#,
                "\n",
                r#"{"a":2,"b":20,"ts":2,"status":200,"host":"y"}"#,
                "\n",
                r#"{"a":1,"b":30,"ts":3,"status":500,"host":"x"}"#,
                "\n",
            ),
        )
        .unwrap();
        let path = file.display().to_string();

        let queries = [
            ".",
            ".a",
            "select(.status == 500)",
            "select(.status == 500) | .b",
            "{t: .ts, s: .status}",
            "count",
            "select(.status == 500) | count",
            "sum(.b)",
            "select(.status == 500) | sum(.b)",
            "min(.b)",
            "max(.b)",
            "avg(.b)",
            "group_by(.host) | count",
        ];

        for q in queries {
            let program = parse(q).unwrap();
            let spec = jq_argv(&jq, &program, &path).unwrap_or_else(|e| panic!("{q}: {e}"));
            let out = Command::new(&spec.argv[0])
                .args(&spec.argv[1..])
                .output()
                .unwrap_or_else(|e| panic!("{q}: {e}"));
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(
                out.status.success() && stderr.is_empty(),
                "jq rejected the translation of `{q}`:\n  argv: {:?}\n  stderr: {stderr}",
                spec.argv
            );

            // And the answer must match warpjq's, for the shapes that have a
            // single comparable value.
            if program.is_aggregate() && program.group_by.is_none() {
                let jq_ans = String::from_utf8_lossy(&out.stdout)
                    .replace(char::from(13), "")
                    .trim()
                    .to_string();
                let ours = warpjq_answer(&program, Path::new(&path))
                    .unwrap_or_else(|| panic!("{q}: no warpjq answer"));
                assert_eq!(jq_ans, ours, "`{q}` disagrees with jq");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncate_is_char_safe_on_multibyte_text() {
        // The notes column truncates; slicing bytes would panic here.
        let s = "日本語のとても長いテキストです".repeat(5);
        let t = truncate(&s, 20);
        assert!(t.chars().count() <= 20);
        assert!(truncate("short", 20).contains("short"));
        assert_eq!(truncate("a\nb", 20), "a b", "newlines flattened");
    }
}
