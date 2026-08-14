//! The CUDA backend: program lowering, the double-buffered chunk pipeline,
//! and the merge that puts GPU rows and CPU fallback rows back in input order.

pub mod ffi;
mod lower;

use std::io::Write;
use std::sync::OnceLock;

use crate::agg::{AggState, GroupKey};
use crate::chunk::{Chunk, Input, Lines};
use crate::error::{Result, WarpError};
use crate::exec::{finish_totals, Backend, BackendKind, OnInvalid, Options, RunStats, Totals};
use crate::json::{Kind, Slot};
use crate::output::Writer;
use crate::query::Program;

pub use lower::LoweredProgram;

/// Double buffering, exactly as described in docs/ARCHITECTURE.md: while one
/// chunk is uploading and computing, the previous one is being drained.
const N_SLOTS: u32 = 2;

/// GPU chunks are smaller than the CPU default. The index arrays are sized off
/// this, and smaller chunks also overlap better, since there is nothing to hide
/// the first upload behind.
pub const DEFAULT_GPU_CHUNK_BYTES: usize = 64 << 20;

/// Result of the one-time ABI check. `OnceLock` rather than a `static mut`
/// behind a `Once`: the latter was sound only by inspection, and a future
/// refactor could have made it unsound without any compiler complaint.
static ABI_RESULT: OnceLock<Option<String>> = OnceLock::new();

/// Cheap "is there a usable GPU" test, safe to call before opening any input.
pub fn probe() -> std::result::Result<(), String> {
    // The ABI check runs once per process; a mismatch means the .cu and the
    // .rs were built from different sources and every result would be suspect.
    if let Some(e) = ABI_RESULT.get_or_init(|| ffi::abi_check().err()) {
        return Err(e.clone());
    }

    let mut err = ffi::ErrBuf::new();
    let rc = unsafe { ffi::warpjq_probe(err.as_mut_ptr(), err.len()) };
    if rc == ffi::OK {
        Ok(())
    } else {
        Err(ffi::status_message(rc, &err))
    }
}

/// e.g. "NVIDIA GeForce RTX 4070 (sm_89, 12282 MiB)". Used in `--stats` and in
/// the `bench` header, so a pasted benchmark says what it ran on.
pub fn device_name() -> Option<String> {
    let mut buf = vec![0u8; 256];
    let rc = unsafe { ffi::warpjq_device_name(buf.as_mut_ptr() as *mut i8, buf.len()) };
    if rc != ffi::OK {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Some(String::from_utf8_lossy(&buf[..end]).into_owned())
}

/// Owns the device context for one run.
pub struct GpuBackend {
    ctx: *mut ffi::Ctx,
    /// Kept alive because the device program holds pointers into it.
    _lowered: Option<Box<LoweredProgram>>,
    chunk_cap: usize,
}

// The context is only ever touched from the thread that created it; the
// pointer is not shared. `run` takes &mut self, which enforces that.
unsafe impl Send for GpuBackend {}

impl GpuBackend {
    pub fn new() -> Result<GpuBackend> {
        probe().map_err(WarpError::GpuUnavailable)?;
        Ok(GpuBackend {
            ctx: std::ptr::null_mut(),
            _lowered: None,
            chunk_cap: 0,
        })
    }

    /// A backend with its device context already built.
    ///
    /// Context creation is where a query gets rejected for being wider than
    /// the kernel's slot table, or where device allocation fails. Doing it up
    /// front lets the caller fall back to the CPU while no output has been
    /// written yet.
    pub fn prepared(program: &Program, options: &Options) -> Result<GpuBackend> {
        let mut b = GpuBackend::new()?;
        b.ensure_ctx(program, options)?;
        Ok(b)
    }

    fn ensure_ctx(&mut self, program: &Program, options: &Options) -> Result<()> {
        if !self.ctx.is_null() {
            return Ok(());
        }
        let lowered = Box::new(LoweredProgram::build(program, options.format));
        let cap = options
            .chunk_bytes
            .min(DEFAULT_GPU_CHUNK_BYTES.max(1 << 20));
        let mut ctx: *mut ffi::Ctx = std::ptr::null_mut();
        let mut err = ffi::ErrBuf::new();
        let rc = unsafe {
            ffi::warpjq_ctx_create(
                &lowered.ffi_program(),
                cap as u64,
                N_SLOTS,
                &mut ctx,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != ffi::OK {
            return Err(WarpError::Cuda {
                op: "creating the device context",
                detail: ffi::status_message(rc, &err),
            });
        }
        self.ctx = ctx;
        self.chunk_cap = cap;
        self._lowered = Some(lowered);
        Ok(())
    }
}

impl Drop for GpuBackend {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { ffi::warpjq_ctx_destroy(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

/// A chunk that has been handed to the device and is waiting to be drained.
///
/// Note there is no `first_line`: the pipeline runs a chunk behind, so at
/// submit time we do not yet know how many lines came before. We do know it at
/// *drain* time, because every earlier chunk has been drained by then, so the
/// running total lives in the loop instead.
struct InFlight {
    slot: u32,
    n_bytes: usize,
}

/// Where the wall clock actually went, printed when `WARPJQ_PROFILE=1`.
///
/// This exists because the first honest measurement of this pipeline showed
/// the GPU merely matching the CPU, and the only way to find out why was to
/// stop guessing. Keeping it in the binary means anyone reproducing a
/// benchmark can see the same breakdown.
#[derive(Default)]
struct Profile {
    enabled: bool,
    read: std::time::Duration,
    stage_copy: std::time::Duration,
    submit: std::time::Duration,
    wait: std::time::Duration,
    merge: std::time::Duration,
    bytes: u64,
}

impl Profile {
    fn new() -> Profile {
        Profile {
            enabled: std::env::var_os("WARPJQ_PROFILE").is_some(),
            ..Default::default()
        }
    }

    fn report(&self) {
        if !self.enabled {
            return;
        }
        let gb = self.bytes as f64 / 1e9;
        let row = |name: &str, d: std::time::Duration| {
            let s = d.as_secs_f64();
            eprintln!(
                "warpjq profile: {name:<22} {s:7.3}s  {:6.2} GB/s",
                if s > 0.0 { gb / s } else { 0.0 }
            );
        };
        eprintln!("warpjq profile: {:.2} GB through the pipeline", gb);
        row("read+chunk (host)", self.read);
        row("copy to pinned", self.stage_copy);
        row("submit (H2D+kernels)", self.submit);
        row("wait (sync+D2H)", self.wait);
        row("merge+write", self.merge);
    }
}

impl Backend for GpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Gpu
    }

    fn run<W: Write>(
        &mut self,
        program: &Program,
        input: &mut Input,
        options: &Options,
        writer: &mut Writer<W>,
    ) -> Result<RunStats> {
        self.ensure_ctx(program, options)?;

        let plan = crate::exec::cpu::Plan::new(program);
        let mut stats = RunStats {
            backend: Some(BackendKind::Gpu),
            ..Default::default()
        };
        let mut totals = Totals::for_program(program);
        let file_name = input.name().to_string();

        writer.write_header(program)?;

        let mut in_flight: Option<InFlight> = None;
        let mut chunk_no: u64 = 0;
        let mut abort: Option<WarpError> = None;
        let mut prof = Profile::new();
        let mut lines_seen: u64 = 0;

        let ctx = self.ctx;
        let cap = self.chunk_cap;
        let mut last_return = std::time::Instant::now();

        let drain = |f: &InFlight,
                     writer: &mut Writer<W>,
                     totals: &mut Totals,
                     stats: &mut RunStats,
                     prof: &mut Profile,
                     lines_seen: &mut u64|
         -> Result<()> {
            drain_slot(
                ctx, f, &plan, program, options, writer, totals, stats, &file_name, prof,
                lines_seen,
            )
        };

        // A real file is read straight into the pinned pool. Everything else
        // (stdin, a chain of files) has no stable extent to read positionally,
        // and keeps the chunker.
        if let Some((file, len)) = input.file() {
            let mut rd = PinnedReader::new(file, len, options.max_line_bytes);
            let mut oversized: Option<u64> = None;
            while !rd.done() && abort.is_none() {
                let slot = (chunk_no % N_SLOTS as u64) as u32;
                let buf = unsafe { ffi::warpjq_slot_buffer(ctx, slot) };
                if buf.is_null() {
                    abort = Some(WarpError::other("device buffer is unavailable"));
                    break;
                }
                let t = std::time::Instant::now();
                let got = match rd.next_chunk(buf, cap) {
                    Ok(g) => g,
                    Err(e) => {
                        abort = Some(e);
                        break;
                    }
                };
                prof.read += t.elapsed();
                let Some(n_bytes) = got else {
                    // One line longer than the whole buffer. Rare enough to be
                    // worth handling by rereading that stretch through the
                    // mapping rather than complicating the fast path.
                    oversized = Some(rd.off);
                    break;
                };
                if n_bytes == 0 {
                    break;
                }
                stats.bytes_in += n_bytes as u64;
                prof.bytes += n_bytes as u64;

                if let Err(e) = submit(ctx, slot, n_bytes, &mut prof) {
                    abort = Some(e);
                    break;
                }
                let queued = InFlight { slot, n_bytes };
                if let Some(prev) = in_flight.take() {
                    if let Err(e) = drain(
                        &prev,
                        writer,
                        &mut totals,
                        &mut stats,
                        &mut prof,
                        &mut lines_seen,
                    ) {
                        abort = Some(e);
                        break;
                    }
                }
                in_flight = Some(queued);
                chunk_no += 1;
            }

            if let Some(prev) = in_flight.take() {
                if abort.is_none() {
                    if let Err(e) = drain(
                        &prev,
                        writer,
                        &mut totals,
                        &mut stats,
                        &mut prof,
                        &mut lines_seen,
                    ) {
                        abort = Some(e);
                    }
                }
            }

            if let (Some(from), None) = (oversized, abort.as_ref()) {
                if let Err(e) = run_tail_on_cpu(
                    input, from, &plan, options, writer, &mut totals, &mut stats,
                    &mut lines_seen,
                ) {
                    abort = Some(e);
                }
            }

            prof.report();
            if let Some(e) = abort {
                return Err(e);
            }
            finish_totals(program, totals, writer)?;
            stats.lines_out = writer.rows();
            return Ok(stats);
        }

        input.for_each_chunk(cap, options.max_line_bytes, |chunk| {
            if abort.is_some() {
                return Ok(0);
            }
            // Time between our returning and being called again is the
            // chunker finding the next newline boundary.
            prof.read += last_return.elapsed();
            stats.bytes_in += chunk.data.len() as u64;
            prof.bytes += chunk.data.len() as u64;

            // A chunk bigger than the staging buffer can only happen when a
            // single line exceeds it. Hand the whole thing to the CPU rather
            // than splitting a line.
            if chunk.data.len() > cap {
                // Drain first. The CPU path writes straight to the shared
                // writer, so doing it while an earlier chunk is still on the
                // device would emit this chunk's rows ahead of that one's --
                // output order is input order, no exceptions.
                if let Some(prev) = in_flight.take() {
                    if let Err(e) = drain(
                        &prev,
                        writer,
                        &mut totals,
                        &mut stats,
                        &mut prof,
                        &mut lines_seen,
                    ) {
                        abort = Some(e);
                        return Ok(0);
                    }
                }
                match run_chunk_on_cpu(
                    &chunk,
                    &plan,
                    options,
                    writer,
                    &mut totals,
                    &mut stats,
                    &file_name,
                ) {
                    Ok(n) => lines_seen += n,
                    Err(e) => abort = Some(e),
                }
                return Ok(0);
            }

            let slot = (chunk_no % N_SLOTS as u64) as u32;
            // Safe because the slot we are about to reuse was drained on the
            // previous iteration: with two slots, chunk n and chunk n-2 share
            // a slot and chunk n-2 was waited on while chunk n-1 was queued.
            if let Err(e) = fill_and_submit(ctx, slot, chunk.data, &mut prof) {
                abort = Some(e);
                return Ok(0);
            }
            let queued = InFlight {
                slot,
                n_bytes: chunk.data.len(),
            };

            if let Some(prev) = in_flight.take() {
                if let Err(e) = drain(
                    &prev,
                    writer,
                    &mut totals,
                    &mut stats,
                    &mut prof,
                    &mut lines_seen,
                ) {
                    abort = Some(e);
                    return Ok(0);
                }
            }
            in_flight = Some(queued);
            chunk_no += 1;
            last_return = std::time::Instant::now();
            // The chunker's own line numbering is unused on this path; we
            // keep an exact count from the device instead.
            Ok(0)
        })?;

        if let Some(prev) = in_flight.take() {
            if abort.is_none() {
                if let Err(e) = drain(
                    &prev,
                    writer,
                    &mut totals,
                    &mut stats,
                    &mut prof,
                    &mut lines_seen,
                ) {
                    abort = Some(e);
                }
            }
        }
        prof.report();

        if let Some(e) = abort {
            return Err(e);
        }

        finish_totals(program, totals, writer)?;
        stats.lines_out = writer.rows();
        Ok(stats)
    }
}

fn fill_and_submit(ctx: *mut ffi::Ctx, slot: u32, data: &[u8], prof: &mut Profile) -> Result<()> {
    let buf = unsafe { ffi::warpjq_slot_buffer(ctx, slot) };
    if buf.is_null() {
        return Err(WarpError::other("device staging buffer is unavailable"));
    }
    // Copy into pinned memory, in parallel. A single-threaded memcpy of a
    // 64 MB chunk runs at roughly one memory channel's worth of bandwidth,
    // which is the same order as the entire rest of the pipeline. Splitting
    // it across cores makes it disappear behind the DMA instead.
    let t = std::time::Instant::now();
    copy_to_pinned(data, buf);
    prof.stage_copy += t.elapsed();

    let t = std::time::Instant::now();
    let mut err = ffi::ErrBuf::new();
    let rc =
        unsafe { ffi::warpjq_submit(ctx, slot, data.len() as u64, err.as_mut_ptr(), err.len()) };
    if rc != ffi::OK {
        return Err(WarpError::Cuda {
            op: "submitting a chunk",
            detail: ffi::status_message(rc, &err),
        });
    }
    prof.submit += t.elapsed();
    Ok(())
}

/// Charges everything from construction to scope exit to `acc`, so the merge
/// cost is measured across every early return in `drain_slot`.
struct MergeTimer<'a> {
    start: std::time::Instant,
    acc: &'a mut std::time::Duration,
}

impl Drop for MergeTimer<'_> {
    fn drop(&mut self) {
        *self.acc += self.start.elapsed();
    }
}

/// Queues H2D and the kernels for bytes already sitting in the slot's buffer.
fn submit(ctx: *mut ffi::Ctx, slot: u32, n_bytes: usize, prof: &mut Profile) -> Result<()> {
    let t = std::time::Instant::now();
    let mut err = ffi::ErrBuf::new();
    let rc = unsafe { ffi::warpjq_submit(ctx, slot, n_bytes as u64, err.as_mut_ptr(), err.len()) };
    if rc != ffi::OK {
        return Err(WarpError::Cuda {
            op: "submitting a chunk",
            detail: ffi::status_message(rc, &err),
        });
    }
    prof.submit += t.elapsed();
    Ok(())
}

/// Finishes a file on the CPU from `from` onward.
///
/// Only reachable when a single line is longer than the whole buffer, which
/// the pinned reader cannot split. Reading it through the mapping keeps that
/// case out of the path every other chunk takes.
#[allow(clippy::too_many_arguments)]
fn run_tail_on_cpu<W: Write>(
    input: &Input,
    from: u64,
    plan: &crate::exec::cpu::Plan,
    options: &Options,
    writer: &mut Writer<W>,
    totals: &mut Totals,
    stats: &mut RunStats,
    lines_seen: &mut u64,
) -> Result<()> {
    let Some(map) = input.mapping() else {
        return Err(WarpError::other(
            "a line is longer than the chunk buffer and the input cannot be re-read",
        ));
    };
    let data = &map[from as usize..];
    stats.bytes_in += data.len() as u64;
    let chunk = Chunk {
        data,
        first_line: *lines_seen + 1,
    };
    let name = input.name().to_string();
    let n = run_chunk_on_cpu(&chunk, plan, options, writer, totals, stats, &name)?;
    *lines_seen += n;
    Ok(())
}

/// Reads the file directly into the pinned buffers the DMA engine reads from.
///
/// The bytes land once, in the only buffer that holds them. There is no
/// mapping to fault in and no staging copy, because there is nothing to stage
/// from: the destination is the pinned pool the context already allocated.
struct PinnedReader<'a> {
    file: &'a std::fs::File,
    len: u64,
    off: u64,
    max_line: usize,
    lines_before: u64,
}

impl<'a> PinnedReader<'a> {
    fn new(file: &'a std::fs::File, len: u64, max_line: usize) -> Self {
        PinnedReader {
            file,
            len,
            // A byte-order mark is not part of the first line. The mapped
            // path strips it with strip_bom; here it is simpler to start
            // reading past it.
            off: bom_len(file, len),
            max_line,
            lines_before: 0,
        }
    }

    fn done(&self) -> bool {
        self.off >= self.len
    }

    /// Fills `dst` with the next whole lines, returning how many bytes.
    ///
    /// Returns `Ok(None)` when the next line is longer than the buffer, which
    /// positional reads cannot serve without splitting it; the caller falls
    /// back for that chunk.
    fn next_chunk(&mut self, dst: *mut u8, cap: usize) -> Result<Option<usize>> {
        let want = cap.min((self.len - self.off) as usize);
        // SAFETY: `dst` is the slot's pinned buffer, `cap` bytes long, and this
        // slot is not in flight.
        let buf = unsafe { std::slice::from_raw_parts_mut(dst, want) };
        let n = read_parallel(self.file, buf, self.off)?;
        if n == 0 {
            self.off = self.len;
            return Ok(Some(0));
        }

        let at_eof = self.off + n as u64 >= self.len;
        let end = if at_eof {
            n
        } else {
            // Chunks end on a line boundary. The bytes past the last newline
            // are re-read as the head of the next chunk rather than moved
            // down, which keeps the buffer write-once.
            match buf[..n].iter().rposition(|&b| b == b'\n') {
                Some(p) => p + 1,
                None => return Ok(None),
            }
        };

        // Enforcing --max-line-bytes needs the longest run without a newline,
        // which is a second pass over the chunk. Skip it when the limit is at
        // least a whole buffer, because then no line inside one can exceed it
        // and a line that spans buffers is caught by the no-newline case above.
        // That covers the default, where the limit equals the chunk size, so
        // the common path never pays for the scan.
        if self.max_line < buf.len() {
            let mut start = 0usize;
            for (i, _) in buf[..end].iter().enumerate().filter(|(_, &b)| b == b'\n') {
                if i - start > self.max_line {
                    return Err(WarpError::LineTooLong {
                        line: self.lines_before + 1,
                        len: i - start,
                        limit: self.max_line,
                    });
                }
                start = i + 1;
                self.lines_before += 1;
            }
            if end - start > self.max_line {
                return Err(WarpError::LineTooLong {
                    line: self.lines_before + 1,
                    len: end - start,
                    limit: self.max_line,
                });
            }
        }

        self.off += end as u64;
        Ok(Some(end))
    }
}

/// Length of a leading UTF-8 byte-order mark, which Windows tooling prepends
/// and which would otherwise make the first line fail to parse.
fn bom_len(file: &std::fs::File, len: u64) -> u64 {
    if len < 3 {
        return 0;
    }
    let mut head = [0u8; 3];
    match read_fully(file, &mut head, 0) {
        Ok(3) if head == [0xEF, 0xBB, 0xBF] => 3,
        _ => 0,
    }
}

#[cfg(unix)]
fn read_at(f: &std::fs::File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(f, buf, off)
}

#[cfg(windows)]
fn read_at(f: &std::fs::File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(f, buf, off)
}

/// Fills `buf` from `off`, splitting the range across cores.
///
/// One thread reading 64 MB is a single core's worth of copy bandwidth, which
/// is the same mistake as the staging memcpy in a different costume. The
/// pieces are disjoint and positional, so they need no shared file cursor.
fn read_parallel(f: &std::fs::File, buf: &mut [u8], off: u64) -> Result<usize> {
    use rayon::prelude::*;
    const MIN_PARALLEL: usize = 1 << 20;
    if buf.len() < MIN_PARALLEL {
        return read_fully(f, buf, off).map_err(WarpError::from);
    }
    let threads = rayon::current_num_threads().clamp(1, 16);
    let piece = buf.len().div_ceil(threads);
    let counts: Vec<std::io::Result<usize>> = buf
        .par_chunks_mut(piece)
        .enumerate()
        .map(|(i, part)| read_fully(f, part, off + (i * piece) as u64))
        .collect();
    let mut total = 0;
    for c in counts {
        // EOF is monotonic in offset, so once a piece comes up short every
        // later piece read nothing and the sum is still the prefix length.
        total += c?;
    }
    Ok(total)
}

fn read_fully(f: &std::fs::File, mut buf: &mut [u8], mut off: u64) -> std::io::Result<usize> {
    let mut total = 0;
    while !buf.is_empty() {
        match read_at(f, buf, off) {
            Ok(0) => break,
            Ok(n) => {
                buf = &mut buf[n..];
                off += n as u64;
                total += n;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(total)
}

/// Parallel memcpy into the pinned staging buffer.
fn copy_to_pinned(data: &[u8], dst: *mut u8) {
    use rayon::prelude::*;
    const MIN_PARALLEL: usize = 1 << 20;
    if data.len() < MIN_PARALLEL {
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
        return;
    }
    let threads = rayon::current_num_threads().clamp(1, 16);
    let piece = data.len().div_ceil(threads);
    let dst_addr = dst as usize;
    data.par_chunks(piece).enumerate().for_each(|(i, src)| {
        // SAFETY: the pieces are disjoint by construction and the destination
        // is at least `data.len()` bytes, checked by the caller against the
        // slot capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                (dst_addr + i * piece) as *mut u8,
                src.len(),
            )
        };
    });
}

#[allow(clippy::too_many_arguments)]
fn drain_slot<W: Write>(
    ctx: *mut ffi::Ctx,
    f: &InFlight,
    plan: &crate::exec::cpu::Plan<'_>,
    program: &Program,
    options: &Options,
    writer: &mut Writer<W>,
    totals: &mut Totals,
    stats: &mut RunStats,
    file_name: &str,
    prof: &mut Profile,
    lines_seen: &mut u64,
) -> Result<()> {
    let t = std::time::Instant::now();
    let mut res = ffi::ChunkResult::default();
    let mut err = ffi::ErrBuf::new();
    let rc = unsafe { ffi::warpjq_wait(ctx, f.slot, &mut res, err.as_mut_ptr(), err.len()) };
    if rc != ffi::OK {
        return Err(WarpError::Cuda {
            op: "waiting for a chunk",
            detail: ffi::status_message(rc, &err),
        });
    }
    prof.wait += t.elapsed();
    let t_merge = std::time::Instant::now();
    let _guard = MergeTimer {
        start: t_merge,
        acc: &mut prof.merge,
    };

    // The staging buffer still holds this chunk's bytes and stays valid until
    // the slot is refilled, so it is the byte source for fallback lines.
    let data: &[u8] =
        unsafe { std::slice::from_raw_parts(ffi::warpjq_slot_buffer(ctx, f.slot), f.n_bytes) };

    let first_line = *lines_seen + 1;

    if res.chunk_overflow != 0 || res.group_overflow != 0 {
        // The device could not represent this chunk exactly (too many lines
        // for the index buffers, or a group cardinality past the table).
        // Redo it on the CPU: slower for this chunk, still correct.
        stats.gpu_fallback_lines += res.n_lines;
        let chunk = Chunk { data, first_line };
        let n = run_chunk_on_cpu(&chunk, plan, options, writer, totals, stats, file_name)?;
        *lines_seen += n;
        return Ok(());
    }

    *lines_seen += res.n_lines;
    stats.lines_in += res.n_lines - res.n_blank;
    stats.malformed += res.n_invalid;
    stats.type_errors += res.n_type_error;
    stats.gpu_fallback_lines += res.n_fallback;

    // The kernel records the lowest failing line index, so --strict names the
    // line that actually failed rather than the start of the chunk it landed
    // in, which is what the CPU backend does, and what jq does.
    let bad_line = if res.first_invalid == ffi::NONE {
        first_line
    } else {
        first_line + res.first_invalid as u64
    };

    if options.on_invalid == OnInvalid::Abort && res.n_invalid > 0 {
        return Err(WarpError::MalformedLine {
            file: file_name.to_string(),
            line: bad_line,
            detail: "line is not valid JSON".to_string(),
        });
    }
    if res.n_invalid > 0 && options.on_invalid != OnInvalid::Skip {
        eprintln!(
            "warpjq: {}:{}: not valid JSON ({} malformed line(s) in this chunk)",
            file_name, bad_line, res.n_invalid
        );
    }

    // Fallback lines get the CPU treatment, using the same evaluator.
    let n_fb = res.n_fallback as usize;
    let fb_idx: &[u32] = if n_fb > 0 {
        unsafe { std::slice::from_raw_parts(res.fallback_idx, n_fb) }
    } else {
        &[]
    };
    let fb_off: &[u32] = if n_fb > 0 {
        unsafe { std::slice::from_raw_parts(res.fallback_off, n_fb) }
    } else {
        &[]
    };
    let fb_len: &[u32] = if n_fb > 0 {
        unsafe { std::slice::from_raw_parts(res.fallback_len, n_fb) }
    } else {
        &[]
    };

    // The kernel appends fallback entries with an atomicAdd, so the three
    // arrays come back in whatever order the blocks happened to retire --
    // *not* in line order. Both the CPU re-evaluation below and the
    // two-pointer merge further down require ascending line indices, so sort
    // an index permutation first. This is cheap: falling back is rare by
    // design, and n_fb is a small fraction of the chunk.
    //
    // Getting this wrong emits real lines in the wrong positions rather than
    // producing garbage, which is exactly the kind of bug that hides until
    // some unrelated change perturbs the scheduling.
    let mut order: Vec<usize> = (0..n_fb).collect();
    order.sort_unstable_by_key(|&k| fb_idx[k]);

    let mut fb_totals = Totals::for_program(program);
    let (fb_rows, fb_bad, fb_type) = crate::exec::cpu::eval_lines_for_fallback(
        plan,
        order.iter().map(|&k| {
            let s = fb_off[k] as usize;
            &data[s..s + fb_len[k] as usize]
        }),
        options.format,
        &mut fb_totals,
    );
    stats.malformed += fb_bad;
    stats.type_errors += fb_type;

    if program.is_aggregate() {
        merge_agg(program, &res, data, totals)?;
        totals.merge(fb_totals);
        return Ok(());
    }

    totals.merge(fb_totals);

    // Merge the device's rows with the CPU's, by line index, so output order
    // is input order regardless of which engine produced each row.
    let n_sel = res.n_selected as usize;
    let sel_idx: &[u32] = if n_sel > 0 {
        unsafe { std::slice::from_raw_parts(res.out_line_idx, n_sel) }
    } else {
        &[]
    };
    let row_off: &[u64] = if n_sel > 0 {
        unsafe { std::slice::from_raw_parts(res.out_row_off, n_sel + 1) }
    } else {
        &[]
    };
    let out_bytes: &[u8] = if res.out_len > 0 {
        unsafe { std::slice::from_raw_parts(res.out_bytes, res.out_len as usize) }
    } else {
        &[]
    };

    // With nothing to interleave, the device's block is already exactly the
    // bytes to emit, in order. Walking it row by row would copy every byte
    // again to arrive at what it already is. This is the usual case: the
    // kernel decides nearly every line, and fallback is the exception.
    if n_fb == 0 {
        if n_sel > 0 {
            let s = row_off[0] as usize;
            let e = row_off[n_sel] as usize;
            writer.write_bulk(&out_bytes[s..e], n_sel as u64)?;
        }
        return Ok(());
    }

    let mut gi = 0usize;
    let mut fi = 0usize;
    while gi < n_sel || fi < n_fb {
        let take_gpu = if gi >= n_sel {
            false
        } else if fi >= n_fb {
            true
        } else {
            // fb_rows is in `order` sequence, so compare against the same.
            sel_idx[gi] < fb_idx[order[fi]]
        };
        if take_gpu {
            let s = row_off[gi] as usize;
            let e = row_off[gi + 1] as usize;
            writer.write_raw(&out_bytes[s..e], 1)?;
            gi += 1;
        } else {
            let row = &fb_rows[fi];
            writer.write_raw(&row.bytes, row.rows)?;
            fi += 1;
        }
    }

    Ok(())
}

/// Folds a chunk's device-side aggregate into the run totals.
fn merge_agg(
    program: &Program,
    res: &ffi::ChunkResult,
    data: &[u8],
    totals: &mut Totals,
) -> Result<()> {
    match totals {
        Totals::Scalar(state) => {
            state.merge(&AggState {
                count: res.agg.count,
                numeric: res.agg.numeric,
                sum: res.agg.sum,
                min: res.agg.min,
                max: res.agg.max,
                saw_non_numeric: res.agg.saw_non_numeric != 0,
            });
        }
        Totals::Grouped(acc) => {
            let n = res.n_groups as usize;
            if n > 0 {
                let groups: &[ffi::Group] = unsafe { std::slice::from_raw_parts(res.groups, n) };
                for g in groups {
                    let key = device_group_key(data, g);
                    acc.entry(key).merge(&AggState {
                        count: g.count,
                        numeric: g.numeric,
                        sum: g.sum,
                        min: g.min,
                        max: g.max,
                        saw_non_numeric: false,
                    });
                }
            }
            let _ = program;
        }
        Totals::None => {}
    }
    Ok(())
}

/// Rebuilds a `GroupKey` from a device table entry.
///
/// The device stores where the key was, not what it decoded to, so the
/// decoding happens here with the same code the CPU backend uses. There is
/// only one implementation of "what does this group key mean".
fn device_group_key(data: &[u8], g: &ffi::Group) -> GroupKey {
    let start = g.key_off as usize;
    let raw = &data[start..start + g.key_len as usize];
    match g.key_kind {
        ffi::JK_NULL | ffi::JK_MISSING => GroupKey::Null,
        ffi::JK_BOOL => GroupKey::Bool(raw == b"true"),
        ffi::JK_NUM => GroupKey::from_slot(&Slot {
            kind: Kind::Num,
            raw,
        }),
        ffi::JK_STR => {
            // The device stored the *body*, without quotes; from_slot expects
            // the quoted form, so decode directly instead.
            if raw.contains(&b'\\') {
                let mut buf = Vec::with_capacity(raw.len());
                if crate::json::unescape_into(raw, &mut buf).is_ok() {
                    return GroupKey::Str(buf);
                }
            }
            GroupKey::Str(raw.to_vec())
        }
        _ => GroupKey::Composite(raw.to_vec()),
    }
}

/// Runs one whole chunk through the CPU backend, used when the device declines
/// the chunk as a unit.
#[allow(clippy::too_many_arguments)]
fn run_chunk_on_cpu<W: Write>(
    chunk: &Chunk<'_>,
    plan: &crate::exec::cpu::Plan<'_>,
    options: &Options,
    writer: &mut Writer<W>,
    totals: &mut Totals,
    stats: &mut RunStats,
    file_name: &str,
) -> Result<u64> {
    let mut out = Writer::new(Vec::new(), options.format);
    let mut slots = plan.fresh_slots();
    let mut lines = 0u64;
    for line in Lines::new(chunk) {
        stats.lines_in += 1;
        lines += 1;
        match crate::exec::cpu::eval_line(line.bytes, plan, &mut slots, &mut out, totals) {
            crate::exec::cpu::LineOutcome::Emitted | crate::exec::cpu::LineOutcome::NoRow => {}
            crate::exec::cpu::LineOutcome::Malformed(msg) => {
                stats.malformed += 1;
                if options.on_invalid == OnInvalid::Abort {
                    return Err(WarpError::MalformedLine {
                        file: file_name.to_string(),
                        line: line.number,
                        detail: msg,
                    });
                }
            }
            crate::exec::cpu::LineOutcome::TypeError(_) => stats.type_errors += 1,
        }
    }
    let (bytes, rows) = out.into_inner()?;
    writer.write_raw(&bytes, rows)?;
    Ok(lines)
}
