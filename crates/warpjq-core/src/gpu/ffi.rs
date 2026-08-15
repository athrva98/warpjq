//! Raw bindings to `cuda/warpjq_kernels.h`.
//!
//! Every struct here is a hand-written mirror of a C struct. That is a
//! standing hazard, so `abi_check` compares `size_of` on both sides at startup
//! and refuses to run on a mismatch. A silently misaligned struct would
//! produce plausible wrong answers rather than a crash, which is the worst
//! possible failure mode for a tool whose pitch is "same output as jq".

use std::os::raw::c_char;

pub const OK: i32 = 0;
pub const ERR_CUDA: i32 = 1;
pub const ERR_NO_DEVICE: i32 = 2;
pub const ERR_OOM: i32 = 3;
pub const ERR_ABI: i32 = 4;
pub const ERR_INVALID_ARG: i32 = 5;

pub const LINE_OK: u8 = 0;
pub const LINE_INVALID: u8 = 1;
pub const LINE_TYPE_ERROR: u8 = 2;
pub const LINE_FALLBACK: u8 = 3;
pub const LINE_BLANK: u8 = 4;

pub const OUT_PASSTHROUGH: u32 = 0;
pub const OUT_PATH: u32 = 1;
pub const OUT_PROJECT: u32 = 2;
pub const OUT_AGG: u32 = 3;

pub const AGG_COUNT: u32 = 0;
pub const AGG_SUM: u32 = 1;
pub const AGG_MIN: u32 = 2;
pub const AGG_MAX: u32 = 3;
pub const AGG_AVG: u32 = 4;

pub const NONE: u32 = 0xFFFF_FFFF;

/// Device-side JSON kinds. Must match `json::Kind` and the `JK_*` enum in
/// the `.cu` file.
pub const JK_MISSING: u32 = 0;
pub const JK_NULL: u32 = 1;
pub const JK_BOOL: u32 = 2;
pub const JK_NUM: u32 = 3;
pub const JK_STR: u32 = 4;
pub const JK_ARR: u32 = 5;
pub const JK_OBJ: u32 = 6;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Program {
    pub steps: *const crate::query::GpuStep,
    pub n_steps: u32,
    pub paths: *const crate::query::GpuPath,
    pub n_paths: u32,
    pub cmps: *const crate::query::GpuCmp,
    pub n_cmps: u32,
    pub cond_rpn: *const crate::query::GpuCondOp,
    pub n_cond: u32,
    pub blob: *const u8,
    pub n_blob: u32,

    pub needed_paths: *const u32,
    pub n_needed: u32,
    pub slot_of_path: *const u32,

    pub output_kind: u32,
    pub output_path: u32,
    pub agg_kind: u32,
    pub agg_path: u32,
    pub group_path: u32,
    pub has_filter: u32,

    pub n_fields: u32,
    pub field_paths: *const u32,
    pub prefix_off: *const u32,
    pub prefix_len: *const u32,
    pub suffix_off: u32,
    pub suffix_len: u32,
    pub csv_mode: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct AggPartial {
    pub count: u64,
    pub numeric: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub saw_non_numeric: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Group {
    pub key_off: u32,
    pub key_len: u32,
    pub key_kind: u32,
    pub _pad: u32,
    pub count: u64,
    pub numeric: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ChunkResult {
    pub n_lines: u64,
    pub n_blank: u64,
    pub n_invalid: u64,
    pub n_type_error: u64,
    pub n_fallback: u64,
    pub n_selected: u64,

    pub out_bytes: *const u8,
    pub out_len: u64,
    pub out_line_idx: *const u32,
    pub out_row_off: *const u64,

    pub fallback_idx: *const u32,
    pub fallback_off: *const u32,
    pub fallback_len: *const u32,
    pub chunk_overflow: u32,
    /// Index within the chunk of the first unparseable line, or [`NONE`].
    pub first_invalid: u32,

    pub agg: AggPartial,

    pub groups: *const Group,
    pub n_groups: u32,
    pub group_overflow: u32,
}

impl Default for ChunkResult {
    fn default() -> Self {
        // Zeroed is a valid "nothing happened" result; the C side memsets it
        // before filling anything in.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
pub struct Ctx {
    _private: [u8; 0],
}

extern "C" {
    pub fn warpjq_probe(err: *mut c_char, err_len: usize) -> i32;
    pub fn warpjq_abi_check(sizes: *const u64, n: usize) -> i32;
    pub fn warpjq_device_name(buf: *mut c_char, len: usize) -> i32;

    pub fn warpjq_ctx_create(
        prog: *const Program,
        max_chunk_bytes: u64,
        n_slots: u32,
        out: *mut *mut Ctx,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;

    pub fn warpjq_ctx_destroy(ctx: *mut Ctx);

    pub fn warpjq_slot_buffer(ctx: *mut Ctx, slot: u32) -> *mut u8;
    pub fn warpjq_slot_capacity(ctx: *const Ctx) -> u64;
    pub fn warpjq_max_lines(ctx: *const Ctx) -> u64;

    pub fn warpjq_submit(
        ctx: *mut Ctx,
        slot: u32,
        n_bytes: u64,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;

    pub fn warpjq_wait(
        ctx: *mut Ctx,
        slot: u32,
        out: *mut ChunkResult,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;
}

/// Scratch for the `char err[]` out-parameters.
pub struct ErrBuf(Vec<u8>);

impl ErrBuf {
    pub fn new() -> ErrBuf {
        ErrBuf(vec![0u8; 512])
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_char {
        self.0.as_mut_ptr() as *mut c_char
    }

    /// Capacity of the buffer, for the `size_t err_len` parameter.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Present only to satisfy the `len`-without-`is_empty` lint; the buffer
    /// is fixed-size and never empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn take(&self) -> String {
        let end = self.0.iter().position(|&b| b == 0).unwrap_or(self.0.len());
        String::from_utf8_lossy(&self.0[..end]).into_owned()
    }
}

impl Default for ErrBuf {
    fn default() -> Self {
        ErrBuf::new()
    }
}

/// Asserts the Rust and C++ views of every shared struct agree.
pub fn abi_check() -> Result<(), String> {
    use crate::query::{GpuCmp, GpuCondOp, GpuPath, GpuStep};
    let sizes: [u64; 6] = [
        std::mem::size_of::<GpuStep>() as u64,
        std::mem::size_of::<GpuPath>() as u64,
        std::mem::size_of::<GpuCmp>() as u64,
        std::mem::size_of::<GpuCondOp>() as u64,
        std::mem::size_of::<AggPartial>() as u64,
        std::mem::size_of::<Group>() as u64,
    ];
    let rc = unsafe { warpjq_abi_check(sizes.as_ptr(), sizes.len()) };
    if rc == OK {
        Ok(())
    } else {
        Err(format!(
            "the Rust and CUDA views of the shared structs disagree \
             (rust sizes: {sizes:?}); this build is inconsistent; \
             run `cargo clean` and rebuild"
        ))
    }
}

pub fn status_message(rc: i32, err: &ErrBuf) -> String {
    let detail = err.take();
    let kind = match rc {
        ERR_CUDA => "CUDA error",
        ERR_NO_DEVICE => "no usable GPU",
        ERR_OOM => "out of memory",
        ERR_ABI => "ABI mismatch",
        ERR_INVALID_ARG => "invalid argument",
        _ => "unknown error",
    };
    if detail.is_empty() {
        kind.to_string()
    } else {
        format!("{kind}: {detail}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_matches_the_compiled_kernels() {
        // If this fails the two sides of the FFI disagree about struct
        // layout, which would corrupt results rather than crash.
        abi_check().expect("ABI mismatch between Rust and CUDA");
    }

    #[test]
    fn json_kind_constants_match_the_rust_enum() {
        use crate::json::Kind;
        assert_eq!(Kind::Missing as u32, JK_MISSING);
        assert_eq!(Kind::Null as u32, JK_NULL);
        assert_eq!(Kind::Bool as u32, JK_BOOL);
        assert_eq!(Kind::Num as u32, JK_NUM);
        assert_eq!(Kind::Str as u32, JK_STR);
        assert_eq!(Kind::Arr as u32, JK_ARR);
        assert_eq!(Kind::Obj as u32, JK_OBJ);
    }
}
