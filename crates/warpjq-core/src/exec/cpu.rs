//! The CPU backend.
//!
//! Two jobs, and they pull in slightly different directions:
//!
//!  1. Be the fallback that makes `cargo install warpjq` work on a laptop with
//!     no GPU, at a speed that is respectable next to jq.
//!  2. Be the *oracle*. `tests/differential.rs` asserts the GPU produces these
//!     exact bytes, so anywhere the two could disagree, this one is right.
//!
//! Job 2 is why every line is validated end to end before any path is
//! resolved, even though skipping validation would be measurably faster. A
//! fallback that quietly accepts input the GPU rejects would make the
//! differential tests meaningless.

use std::io::Write;

use rayon::prelude::*;

use crate::agg::{AggState, GroupKey};
use crate::chunk::{Chunk, Input, Lines};
use crate::error::{Result, WarpError};
use crate::exec::{finish_totals, Backend, BackendKind, OnInvalid, Options, RunStats, Totals};
use crate::json::{self, Lookup, Slot, MISSING};
use crate::output::Writer;
use crate::query::{Cond, Output, PathId, Program};

#[derive(Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

/// A resolved value, stored as offsets rather than a borrow.
///
/// The obvious representation is `Slot<'line>`, but then the reusable slot
/// table would be tied to one line's lifetime and have to be reallocated per
/// line. Offsets let one `Vec` serve every line in a slice, and let the GPU
/// fallback path call the same evaluator.
#[derive(Copy, Clone, Debug)]
pub(crate) struct RawSlot {
    /// Byte offset within the line. `usize`, not `i32`: a narrower field
    /// silently wraps on a line past 2 GB, which `--max-line-bytes` permits,
    /// and the wrapped value then reads as "missing", dropping real data on
    /// exactly the oversized inputs this tool exists for.
    off: usize,
    len: usize,
    kind: crate::json::Kind,
}

/// `kind == Missing` is the only marker; there is no out-of-band offset.
const RAW_MISSING: RawSlot = RawSlot {
    off: 0,
    len: 0,
    kind: crate::json::Kind::Missing,
};

impl RawSlot {
    fn of(line: &[u8], s: Slot<'_>) -> RawSlot {
        if s.kind == crate::json::Kind::Missing {
            return RAW_MISSING;
        }
        let base = line.as_ptr() as usize;
        let off = s.raw.as_ptr() as usize - base;
        RawSlot {
            off,
            len: s.raw.len(),
            kind: s.kind,
        }
    }

    fn to_slot<'a>(self, line: &'a [u8]) -> Slot<'a> {
        if self.kind == crate::json::Kind::Missing {
            return MISSING;
        }
        Slot {
            kind: self.kind,
            raw: &line[self.off..self.off + self.len],
        }
    }
}

/// Precomputed per-query scheduling: which paths must be resolved before the
/// filter can be decided, and which only matter for lines that survive it.
pub(crate) struct Plan<'p> {
    program: &'p Program,
    filter_paths: Vec<PathId>,
    post_paths: Vec<PathId>,
    proj_keys: Vec<String>,
    proj_paths: Vec<PathId>,
    n_paths: usize,
}

impl<'p> Plan<'p> {
    /// A fresh slot table for this query, for callers that drive
    /// [`eval_line`] themselves.
    #[cfg(feature = "cuda")]
    pub(crate) fn fresh_slots(&self) -> Vec<RawSlot> {
        vec![RAW_MISSING; self.n_paths]
    }

    pub(crate) fn new(program: &'p Program) -> Self {
        let mut filter_paths = Vec::new();
        if let Some(c) = &program.filter {
            c.visit_paths(&mut |id| {
                if !filter_paths.contains(&id) {
                    filter_paths.push(id);
                }
            });
        }
        // Everything else is deferred: on a `select(.status==500)` over a log
        // where 0.1% of lines match, this skips the projection work for
        // 99.9% of the input.
        let post_paths: Vec<PathId> = program
            .required_paths()
            .into_iter()
            .filter(|p| !filter_paths.contains(p))
            .collect();

        let (proj_keys, proj_paths) = match &program.output {
            Output::Project(fields) => (
                fields.iter().map(|(k, _)| k.clone()).collect(),
                fields.iter().map(|(_, p)| *p).collect(),
            ),
            _ => (Vec::new(), Vec::new()),
        };

        Plan {
            program,
            filter_paths,
            post_paths,
            proj_keys,
            proj_paths,
            n_paths: program.paths.len(),
        }
    }
}

/// Per-worker output and counters for one slice of a chunk.
struct Partial {
    bytes: Vec<u8>,
    rows: u64,
    /// Physical lines in this slice, used to rebase error line numbers.
    physical_lines: u64,
    lines_in: u64,
    totals: Totals,
    /// (slice-relative line index, message), rebased by the caller.
    malformed: Vec<(u64, String)>,
    type_errors: Vec<(u64, String)>,
    abort: Option<(u64, String)>,
}

impl Backend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn run<W: Write>(
        &mut self,
        program: &Program,
        input: &mut Input,
        options: &Options,
        writer: &mut Writer<W>,
    ) -> Result<RunStats> {
        let plan = Plan::new(program);
        let mut stats = RunStats {
            backend: Some(BackendKind::Cpu),
            ..Default::default()
        };
        let mut totals = Totals::for_program(program);
        let file_name = input.name().to_string();

        let threads = if options.threads > 0 {
            options.threads
        } else {
            rayon::current_num_threads()
        };

        writer.write_header(program)?;

        let mut abort: Option<WarpError> = None;
        let mut warnings_shown = 0usize;

        input.for_each_chunk(options.chunk_bytes, options.max_line_bytes, |chunk| {
            if abort.is_some() {
                return Ok(0);
            }
            stats.bytes_in += chunk.data.len() as u64;

            let slices = split_chunk(&chunk, threads);
            let partials: Vec<Partial> = slices
                .par_iter()
                .map(|slice| process_slice(slice, &plan, options))
                .collect();

            // Rebase relative line numbers and emit in input order.
            let mut line_base = chunk.first_line;
            let chunk_lines: u64 = partials.iter().map(|p| p.physical_lines).sum();
            for p in partials {
                if let Some((rel, msg)) = &p.abort {
                    if abort.is_none() {
                        abort = Some(WarpError::MalformedLine {
                            file: file_name.clone(),
                            line: line_base + rel,
                            detail: msg.clone(),
                        });
                    }
                }
                stats.lines_in += p.lines_in;
                stats.lines_out += p.rows;
                stats.malformed += p.malformed.len() as u64;
                stats.type_errors += p.type_errors.len() as u64;
                // The cap is per run, not per slice: a file split across
                // eight workers used to print eight times the intended number
                // of warnings.
                for (rel, msg) in p.malformed.iter().chain(p.type_errors.iter()) {
                    if warnings_shown >= MAX_WARNINGS {
                        break;
                    }
                    warnings_shown += 1;
                    warn_once(&file_name, line_base + rel, msg, options);
                }
                writer.write_raw(&p.bytes, p.rows)?;
                totals.merge(p.totals);
                line_base += p.physical_lines;
            }
            Ok(chunk_lines)
        })?;

        if let Some(e) = abort {
            return Err(e);
        }

        finish_totals(program, totals, writer)?;
        stats.lines_out = writer.rows();
        Ok(stats)
    }
}

const MAX_WARNINGS: usize = 5;

fn warn_once(file: &str, line: u64, msg: &str, options: &Options) {
    if options.on_invalid == OnInvalid::Skip {
        return;
    }
    eprintln!("warpjq: {file}:{line}: {msg}");
}

/// Splits a chunk into roughly `n` newline-aligned slices.
///
/// Every slice starts just after a `\n`, so no worker ever sees half a line
/// and the slices can be processed in any order while still being *emitted*
/// in input order.
fn split_chunk<'a>(chunk: &Chunk<'a>, n: usize) -> Vec<&'a [u8]> {
    let data = chunk.data;
    // Below this, thread hand-off costs more than the work.
    const MIN_SLICE: usize = 64 << 10;
    let n = n.max(1).min((data.len() / MIN_SLICE).max(1));
    if n == 1 {
        return vec![data];
    }

    let mut bounds = Vec::with_capacity(n + 1);
    bounds.push(0usize);
    for i in 1..n {
        let want = data.len() * i / n;
        let prev = *bounds.last().unwrap();
        let cut = match data[want..].iter().position(|&b| b == b'\n') {
            Some(rel) => want + rel + 1,
            None => data.len(),
        };
        if cut > prev && cut < data.len() {
            bounds.push(cut);
        }
    }
    bounds.push(data.len());
    bounds.dedup();

    bounds.windows(2).map(|w| &data[w[0]..w[1]]).collect()
}

/// What happened to one line.
pub(crate) enum LineOutcome {
    Emitted,
    /// Filtered out by `select(...)`, or folded into an aggregate.
    NoRow,
    Malformed(String),
    TypeError(String),
}

/// Evaluates one line: the single source of truth for what a query *means*.
///
/// Both the parallel CPU loop below and the GPU fallback path in `gpu::` call
/// this, so a line the kernel declines gets exactly the treatment it would
/// have had on the CPU, which is what makes the merged output byte-identical
/// either way.
pub(crate) fn eval_line<W: std::io::Write>(
    line: &[u8],
    plan: &Plan<'_>,
    slots: &mut [RawSlot],
    out: &mut Writer<W>,
    totals: &mut Totals,
) -> LineOutcome {
    let program = plan.program;

    if let Err(e) = json::validate(line) {
        return LineOutcome::Malformed(e.to_string());
    }

    for s in slots.iter_mut() {
        *s = RAW_MISSING;
    }

    // Phase 1: only what the filter needs.
    for &id in &plan.filter_paths {
        match json::lookup(line, &program.path(id).steps) {
            Lookup::Found(s) => slots[id as usize] = RawSlot::of(line, s),
            Lookup::TypeError(msg) => return LineOutcome::TypeError(msg),
            Lookup::Invalid(e) => return LineOutcome::Malformed(e.to_string()),
        }
    }

    if let Some(cond) = &program.filter {
        if !eval_cond(cond, line, slots) {
            return LineOutcome::NoRow;
        }
    }

    // Phase 2: only for lines that survived the filter.
    for &id in &plan.post_paths {
        match json::lookup(line, &program.path(id).steps) {
            Lookup::Found(s) => slots[id as usize] = RawSlot::of(line, s),
            Lookup::TypeError(msg) => return LineOutcome::TypeError(msg),
            Lookup::Invalid(e) => return LineOutcome::Malformed(e.to_string()),
        }
    }

    match &program.output {
        Output::Passthrough => {
            let _ = out.passthrough(line);
            LineOutcome::Emitted
        }
        Output::Path(p) => {
            let _ = out.value(&slots[*p as usize].to_slot(line));
            LineOutcome::Emitted
        }
        Output::Project(_) => {
            let vals: Vec<Slot<'_>> = plan
                .proj_paths
                .iter()
                .map(|p| slots[*p as usize].to_slot(line))
                .collect();
            let _ = out.projection(&plan.proj_keys, &vals);
            LineOutcome::Emitted
        }
        Output::Agg { arg, .. } => {
            let value = arg.map(|p| slots[p as usize].to_slot(line));
            match (totals, program.group_by) {
                (Totals::Grouped(acc), Some(g)) => {
                    let key = GroupKey::from_slot(&slots[g as usize].to_slot(line));
                    push(acc.entry(key), value.as_ref());
                }
                (Totals::Scalar(state), _) => push(state, value.as_ref()),
                _ => {}
            }
            LineOutcome::NoRow
        }
    }
}

fn process_slice(slice: &[u8], plan: &Plan<'_>, options: &Options) -> Partial {
    let program = plan.program;
    let mut out = Writer::new(Vec::new(), options.format);
    let mut totals = Totals::for_program(program);
    let mut malformed = Vec::new();
    let mut type_errors = Vec::new();
    let mut abort = None;
    let mut lines_in = 0u64;

    let mut slots: Vec<RawSlot> = vec![RAW_MISSING; plan.n_paths];

    let chunk = Chunk {
        data: slice,
        // Zero-based within the slice; rebased by the caller.
        first_line: 0,
    };

    for line in Lines::new(&chunk) {
        lines_in += 1;
        match eval_line(line.bytes, plan, &mut slots, &mut out, &mut totals) {
            LineOutcome::Emitted | LineOutcome::NoRow => {}
            LineOutcome::Malformed(msg) => {
                if options.on_invalid == OnInvalid::Abort {
                    abort = Some((line.number, msg));
                    break;
                }
                malformed.push((line.number, msg));
            }
            LineOutcome::TypeError(msg) => {
                if options.on_invalid == OnInvalid::Abort {
                    abort = Some((line.number, msg));
                    break;
                }
                type_errors.push((line.number, msg));
            }
        }
    }

    let physical_lines = count_physical_lines(slice);
    let (bytes, rows) = out.into_inner().expect("Vec write cannot fail");

    Partial {
        bytes,
        rows,
        physical_lines,
        lines_in,
        totals,
        malformed,
        type_errors,
        abort,
    }
}

fn push(state: &mut AggState, value: Option<&Slot<'_>>) {
    match value {
        Some(s) => state.push_value(s),
        None => state.push_count(),
    }
}

fn count_physical_lines(slice: &[u8]) -> u64 {
    let n = slice.iter().filter(|&&b| b == b'\n').count() as u64;
    // A slice that does not end in a newline still holds one more line.
    if slice.last().is_some_and(|&b| b != b'\n') {
        n + 1
    } else {
        n
    }
}

fn eval_cond(cond: &Cond, line: &[u8], slots: &[RawSlot]) -> bool {
    match cond {
        Cond::Cmp { path, op, lit } => {
            json::eval_cmp(&slots[*path as usize].to_slot(line), *op, lit)
        }
        Cond::Truthy(p) => slots[*p as usize].to_slot(line).is_truthy(),
        Cond::And(a, b) => eval_cond(a, line, slots) && eval_cond(b, line, slots),
        Cond::Or(a, b) => eval_cond(a, line, slots) || eval_cond(b, line, slots),
        Cond::Not(a) => !eval_cond(a, line, slots),
    }
}

/// One line's worth of fallback output.
#[cfg(feature = "cuda")]
pub(crate) struct FallbackRow {
    pub bytes: Vec<u8>,
    /// 1 when this line produced a row, 0 otherwise.
    ///
    /// Kept separately from `bytes` because they are not the same question:
    /// under `--count` a row is produced but formats to zero bytes, and
    /// inferring "no row" from "no bytes" silently dropped every
    /// CPU-finished line from the tally.
    pub rows: u64,
}

/// Runs a set of individual lines through the CPU evaluator.
///
/// The GPU backend uses this for the lines its kernel declined, so the caller
/// can splice them into the GPU's output at the right positions.
#[cfg(feature = "cuda")]
pub(crate) fn eval_lines_for_fallback<'a>(
    plan: &Plan<'_>,
    lines: impl Iterator<Item = &'a [u8]>,
    format: crate::output::Format,
    totals: &mut Totals,
) -> (Vec<FallbackRow>, u64, u64) {
    let mut rows = Vec::new();
    let mut malformed = 0u64;
    let mut type_errors = 0u64;
    let mut slots: Vec<RawSlot> = vec![RAW_MISSING; plan.n_paths];
    for line in lines {
        let mut out = Writer::new(Vec::new(), format);
        match eval_line(line, plan, &mut slots, &mut out, totals) {
            LineOutcome::Emitted | LineOutcome::NoRow => {}
            LineOutcome::Malformed(_) => malformed += 1,
            LineOutcome::TypeError(_) => type_errors += 1,
        }
        let (bytes, n) = out.into_inner().expect("Vec write cannot fail");
        rows.push(FallbackRow { bytes, rows: n });
    }
    (rows, malformed, type_errors)
}

/// Convenience for tests and for the differential harness: run a query over an
/// in-memory buffer and return the output bytes.
pub fn run_bytes(program: &Program, data: &[u8], options: &Options) -> Result<(Vec<u8>, RunStats)> {
    use std::io::Cursor;
    let mut input = Input::Streamed {
        reader: Box::new(Cursor::new(data.to_vec())),
        name: "<memory>".to_string(),
    };
    let mut writer = Writer::new(Vec::new(), options.format);
    let mut backend = CpuBackend::new();
    let stats = backend.run(program, &mut input, options, &mut writer)?;
    let (bytes, _rows) = writer.finish()?;
    Ok((bytes, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::Format;
    use crate::query::parse;

    fn run(query: &str, input: &str) -> String {
        run_fmt(query, input, Format::Ndjson)
    }

    fn run_fmt(query: &str, input: &str, format: Format) -> String {
        let p = parse(query).unwrap_or_else(|e| panic!("{query}: {e}"));
        let opts = Options {
            format,
            // Force multi-slice execution so ordering bugs surface in tests.
            chunk_bytes: 64,
            threads: 4,
            ..Default::default()
        };
        let (bytes, _) = run_bytes(&p, input.as_bytes(), &opts).unwrap();
        String::from_utf8(bytes).unwrap()
    }

    const LOG: &str = concat!(
        r#"{"status":200,"bytes":10,"host":"a","msg":"ok"}"#,
        "\n",
        r#"{"status":500,"bytes":20,"host":"b","msg":"boom"}"#,
        "\n",
        r#"{"status":500,"bytes":30,"host":"a","msg":"boom again"}"#,
        "\n",
        r#"{"status":404,"bytes":40,"host":"c"}"#,
        "\n",
    );

    #[test]
    fn identity_passes_lines_through_byte_for_byte() {
        assert_eq!(run(".", LOG), LOG);
    }

    #[test]
    fn select_filters_and_preserves_order() {
        let out = run("select(.status == 500)", LOG);
        assert_eq!(
            out,
            concat!(
                r#"{"status":500,"bytes":20,"host":"b","msg":"boom"}"#,
                "\n",
                r#"{"status":500,"bytes":30,"host":"a","msg":"boom again"}"#,
                "\n"
            )
        );
    }

    #[test]
    fn count_counts_survivors() {
        assert_eq!(run("select(.status == 500) | count", LOG), "2\n");
        assert_eq!(run("count", LOG), "4\n");
    }

    #[test]
    fn projection_emits_objects_in_query_key_order() {
        assert_eq!(
            run("select(.status==404) | {h: .host, s: .status}", LOG),
            "{\"h\":\"c\",\"s\":404}\n"
        );
    }

    #[test]
    fn missing_fields_project_as_null() {
        assert_eq!(
            run("select(.status==404) | {m: .msg}", LOG),
            "{\"m\":null}\n"
        );
    }

    #[test]
    fn aggregates_over_a_field() {
        assert_eq!(run("sum(.bytes)", LOG), "100\n");
        assert_eq!(run("min(.bytes)", LOG), "10\n");
        assert_eq!(run("max(.bytes)", LOG), "40\n");
        assert_eq!(run("avg(.bytes)", LOG), "25\n");
        assert_eq!(run("select(.status==500) | sum(.bytes)", LOG), "50\n");
    }

    #[test]
    fn group_by_is_sorted_and_stable() {
        assert_eq!(
            run("group_by(.host) | count", LOG),
            "{\"host\":\"a\",\"count\":2}\n{\"host\":\"b\",\"count\":1}\n{\"host\":\"c\",\"count\":1}\n"
        );
        assert_eq!(
            run("group_by(.host) | sum(.bytes)", LOG),
            "{\"host\":\"a\",\"sum\":40}\n{\"host\":\"b\",\"sum\":20}\n{\"host\":\"c\",\"sum\":40}\n"
        );
    }

    #[test]
    fn csv_output_has_a_header() {
        assert_eq!(
            run_fmt("{h: .host, s: .status}", LOG, Format::Csv),
            "h,s\na,200\nb,500\na,500\nc,404\n"
        );
    }

    #[test]
    fn count_only_format_prints_a_single_number() {
        assert_eq!(
            run_fmt("select(.status == 500)", LOG, Format::CountOnly),
            "2\n"
        );
    }

    #[test]
    fn output_order_is_input_order_regardless_of_slicing() {
        let mut input = String::new();
        for i in 0..2000 {
            input.push_str(&format!("{{\"i\":{i}}}\n"));
        }
        let out = run(".i", &input);
        let got: Vec<&str> = out.lines().collect();
        assert_eq!(got.len(), 2000);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(*v, i.to_string(), "row {i} out of order");
        }
    }

    #[test]
    fn malformed_lines_are_skipped_not_fatal_by_default() {
        let input = concat!(r#"{"a":1}"#, "\n", "{not json\n", r#"{"a":2}"#, "\n");
        let p = parse(".a").unwrap();
        let opts = Options {
            on_invalid: OnInvalid::Skip,
            ..Default::default()
        };
        let (bytes, stats) = run_bytes(&p, input.as_bytes(), &opts).unwrap();
        assert_eq!(String::from_utf8(bytes).unwrap(), "1\n2\n");
        assert_eq!(stats.malformed, 1);
        assert_eq!(stats.lines_in, 3);
    }

    #[test]
    fn strict_mode_aborts_on_the_first_bad_line() {
        let input = "{\"a\":1}\n{oops\n";
        let p = parse(".a").unwrap();
        let opts = Options {
            on_invalid: OnInvalid::Abort,
            ..Default::default()
        };
        let err = run_bytes(&p, input.as_bytes(), &opts).unwrap_err();
        assert!(
            matches!(err, WarpError::MalformedLine { line: 2, .. }),
            "{err}"
        );
    }

    #[test]
    fn type_errors_skip_the_line_and_are_counted() {
        // `.a` is a number, so `.a.b` is a type error, exactly as in jq.
        let input = "{\"a\":1}\n{\"a\":{\"b\":2}}\n";
        let p = parse(".a.b").unwrap();
        let opts = Options {
            on_invalid: OnInvalid::Skip,
            ..Default::default()
        };
        let (bytes, stats) = run_bytes(&p, input.as_bytes(), &opts).unwrap();
        assert_eq!(String::from_utf8(bytes).unwrap(), "2\n");
        assert_eq!(stats.type_errors, 1);
    }

    #[test]
    fn boolean_operators_compose() {
        let out = run("select(.status >= 400 and .host == \"a\")", LOG);
        assert!(out.contains("boom again"));
        assert_eq!(out.lines().count(), 1);

        let out = run("select(.status == 200 or .status == 404) | count", LOG);
        assert_eq!(out, "2\n");

        let out = run("select(.status == 500 | not) | count", LOG);
        assert_eq!(out, "2\n");
    }

    #[test]
    fn empty_input_produces_empty_output_and_zero_counts() {
        assert_eq!(run(".", ""), "");
        assert_eq!(run("count", ""), "0\n");
        assert_eq!(run("sum(.x)", ""), "0\n");
        assert_eq!(run("min(.x)", ""), "null\n");
        assert_eq!(run("avg(.x)", ""), "null\n");
        assert_eq!(run("group_by(.h) | count", ""), "");
    }

    #[test]
    fn crlf_and_blank_lines_are_tolerated() {
        let input = "{\"a\":1}\r\n\r\n{\"a\":2}\r\n";
        assert_eq!(run(".a", input), "1\n2\n");
    }
}
