//! Execution: the contract both backends implement, and the shared run state.

pub mod cpu;

use std::io::Write;
use std::time::Duration;

use crate::agg::{AggState, GroupAccumulator};
use crate::chunk::{Input, DEFAULT_CHUNK_BYTES, DEFAULT_MAX_LINE_BYTES};
use crate::error::Result;
use crate::output::{Format, Writer};
use crate::query::Program;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Gpu,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Gpu => "gpu",
        }
    }
}

/// How a malformed line is handled.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum OnInvalid {
    /// Count it, keep going, warn once at the end. jq's default is to abort;
    /// ours is to survive, because a 10 GB log with three truncated lines is
    /// the normal case, not the exceptional one.
    #[default]
    Warn,
    /// Drop it and say nothing.
    Skip,
    /// Stop immediately with a non-zero exit.
    Abort,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub format: Format,
    pub chunk_bytes: usize,
    pub max_line_bytes: usize,
    pub on_invalid: OnInvalid,
    /// 0 means "one per core".
    pub threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            format: Format::Ndjson,
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            max_line_bytes: DEFAULT_MAX_LINE_BYTES,
            on_invalid: OnInvalid::Warn,
            threads: 0,
        }
    }
}

/// What a run did, for `--stats` and `--bench`.
#[derive(Clone, Debug, Default)]
pub struct RunStats {
    pub backend: Option<BackendKind>,
    pub bytes_in: u64,
    pub lines_in: u64,
    pub lines_out: u64,
    pub malformed: u64,
    /// Lines where a path hit a type error, e.g. `.a.b` with `.a` a number.
    pub type_errors: u64,
    /// Lines the kernel declined individually, for the CPU to finish. Normal
    /// and cheap in small numbers. Always 0 on the CPU backend.
    pub gpu_fallback_lines: u64,
    /// Chunks the device could not represent at all, and the lines in them.
    ///
    /// Separate from the count above because it means something different: not
    /// a handful of awkward lines, but the GPU doing nothing for that stretch
    /// of input while still reporting itself as the backend. Short lines are
    /// the usual cause, since the index buffers are sized assuming at least 24
    /// bytes a line.
    pub gpu_redone_chunks: u64,
    pub gpu_redone_lines: u64,
    pub elapsed: Duration,
}

impl RunStats {
    pub fn throughput_gbps(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        self.bytes_in as f64 / secs / 1e9
    }

    pub fn merge(&mut self, other: &RunStats) {
        self.bytes_in += other.bytes_in;
        self.lines_in += other.lines_in;
        self.lines_out += other.lines_out;
        self.malformed += other.malformed;
        self.type_errors += other.type_errors;
        self.gpu_fallback_lines += other.gpu_fallback_lines;
        self.gpu_redone_chunks += other.gpu_redone_chunks;
        self.gpu_redone_lines += other.gpu_redone_lines;
    }
}

/// Accumulated aggregate state for a run, in whichever shape the query needs.
#[derive(Default, Debug)]
pub enum Totals {
    #[default]
    None,
    Scalar(AggState),
    Grouped(GroupAccumulator),
}

impl Totals {
    pub fn for_program(p: &Program) -> Totals {
        if !p.is_aggregate() {
            Totals::None
        } else if p.group_by.is_some() {
            Totals::Grouped(GroupAccumulator::default())
        } else {
            Totals::Scalar(AggState::default())
        }
    }

    pub fn merge(&mut self, other: Totals) {
        match (self, other) {
            (Totals::Scalar(a), Totals::Scalar(b)) => a.merge(&b),
            (Totals::Grouped(a), Totals::Grouped(b)) => a.merge(b),
            _ => {}
        }
    }
}

/// Runs `program` over `input`, writing to `writer`.
///
/// Both backends go through this signature, which is what makes
/// `tests/differential.rs` able to swap them and diff the bytes.
pub trait Backend {
    fn kind(&self) -> BackendKind;

    fn run<W: Write>(
        &mut self,
        program: &Program,
        input: &mut Input,
        options: &Options,
        writer: &mut Writer<W>,
    ) -> Result<RunStats>;
}

/// Which engine the caller wants.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Preference {
    /// GPU when one is usable, CPU otherwise. Never fails over hardware.
    #[default]
    Auto,
    /// Force the GPU. Fails loudly if it is unavailable, the right choice
    /// in CI, where a silent fallback would turn a broken kernel into a
    /// green build.
    Gpu,
    Cpu,
}

/// Why the GPU was not used, when it wasn't.
#[derive(Clone, Debug)]
pub enum GpuStatus {
    Available,
    /// Built without `--features cuda`.
    NotCompiledIn,
    Unavailable(String),
}

impl GpuStatus {
    pub fn detect() -> GpuStatus {
        #[cfg(feature = "cuda")]
        {
            match crate::gpu::probe() {
                Ok(()) => GpuStatus::Available,
                Err(e) => GpuStatus::Unavailable(e.to_string()),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            GpuStatus::NotCompiledIn
        }
    }

    pub fn reason(&self) -> String {
        match self {
            GpuStatus::Available => "available".into(),
            GpuStatus::NotCompiledIn => {
                "this binary was built without CUDA support (--features cuda)".into()
            }
            GpuStatus::Unavailable(why) => why.clone(),
        }
    }

    pub fn is_available(&self) -> bool {
        matches!(self, GpuStatus::Available)
    }
}

/// Runs `program` on the preferred engine, falling back where allowed.
pub fn run_query<W: Write>(
    program: &Program,
    input: &mut Input,
    options: &Options,
    writer: &mut Writer<W>,
    preference: Preference,
) -> Result<RunStats> {
    validate(program, options)?;
    let want_gpu = matches!(preference, Preference::Auto | Preference::Gpu);

    if want_gpu {
        let status = GpuStatus::detect();
        if status.is_available() {
            #[cfg(feature = "cuda")]
            {
                // Set the device context up *before* running, so a query the
                // kernel cannot host (more distinct field paths than its slot
                // table holds, or not enough device memory) can still fall
                // back on `auto` instead of failing the whole run. Nothing has
                // been written at this point, so falling back is safe.
                match crate::gpu::GpuBackend::prepared(program, options, input.len_hint()) {
                    Ok(mut backend) => return backend.run(program, input, options, writer),
                    Err(e) if preference == Preference::Gpu => return Err(e),
                    Err(e) => {
                        eprintln!(
                            "warpjq: falling back to the CPU engine: {e}\n\
                             warpjq: pass --backend gpu to make this fatal instead"
                        );
                    }
                }
            }
        }
        if preference == Preference::Gpu {
            return Err(crate::error::WarpError::GpuUnavailable(status.reason()));
        }
    }

    CpuBackend::new().run(program, input, options, writer)
}

use cpu::CpuBackend;

/// Rejects query/option combinations that have no sensible answer.
///
/// This lives here rather than in the CLI so that both backends refuse the
/// same things. A combination that one backend guesses at and the other
/// guesses at differently is worse than an error message.
pub fn validate(program: &Program, options: &Options) -> Result<()> {
    if options.format == Format::Csv && !crate::output::csv_is_meaningful(program) {
        return Err(crate::error::WarpError::Other(
            "--csv needs a query with named columns, but this one emits whole \
             lines\n  help: project the fields you want, e.g. \
             `{ts: .ts, status: .status}`, or use an aggregate"
                .to_string(),
        ));
    }
    Ok(())
}

/// Runs a query over an in-memory buffer and returns the output bytes.
///
/// This is what the differential tests drive: same program, same options, two
/// backends, compare the bytes.
pub fn run_bytes(
    program: &Program,
    data: &[u8],
    options: &Options,
    preference: Preference,
) -> Result<(Vec<u8>, RunStats)> {
    // A temp file, not a cursor. This comment used to say exactly that while
    // the code below passed a cursor, and the difference is not cosmetic: a
    // cursor is a stream, and the GPU backend serves streams with the chunker
    // rather than with the reader it uses for real files. The entire
    // differential suite was therefore testing a path no real invocation
    // takes. Writing the bytes out first costs a little and puts both backends
    // on the same input path a user gets.
    let path = scratch_path();
    let mut input = match std::fs::write(&path, data).and_then(|_| Input::open(&path)) {
        Ok(i) => i,
        // A read-only or missing temp dir should not turn every test into a
        // failure about the filesystem; fall back to the stream path.
        Err(_) => Input::from_bytes(data),
    };
    let _cleanup = ScratchFile(path);
    let mut writer = Writer::new(Vec::new(), options.format);
    let stats = run_query(program, &mut input, options, &mut writer, preference)?;
    let (bytes, _) = writer.finish()?;
    Ok((bytes, stats))
}

/// A counter, not the data address: addresses repeat once the allocator reuses
/// them, and every empty slice shares one, so two of these running in parallel
/// would collide on a filename and read each other's bytes.
fn scratch_path() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "warpjq-run-{}-{}.ndjson",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

struct ScratchFile(std::path::PathBuf);

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Writes the terminal rows for an aggregate query and returns the row count.
pub fn finish_totals<W: Write>(
    program: &Program,
    totals: Totals,
    writer: &mut Writer<W>,
) -> Result<()> {
    let crate::query::Output::Agg { kind, .. } = program.output else {
        return Ok(());
    };
    match totals {
        Totals::Scalar(state) => writer.aggregate(kind, &state)?,
        Totals::Grouped(acc) => {
            let name = crate::output::group_key_name(program);
            let groups = acc.sorted();
            writer.groups(&name, kind, &groups)?;
        }
        Totals::None => {}
    }
    Ok(())
}
