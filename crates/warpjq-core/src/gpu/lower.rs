//! Lowers a compiled [`Program`] into the flat, pointer-free tables the
//! kernel reads.
//!
//! The interesting part is the projection: the kernel never formats a key
//! name. Instead the host precomputes the literal bytes that surround each
//! value (`{"a":` then `,"b":` then `}`) and the kernel just copies them.
//! That keeps JSON escaping of key names on the CPU, where it is already
//! implemented once and tested, and out of a kernel where getting it subtly
//! wrong would be hard to notice.

use super::ffi;
use crate::output::{write_json_string, Format};
use crate::query::{
    AggKind, FlatProgram, GpuCmp, GpuCondOp, GpuPath, GpuStep, Output, PathId, Program,
};

/// Owns every buffer the device program points into.
pub struct LoweredProgram {
    flat: FlatProgram,
    blob: Vec<u8>,
    needed_paths: Vec<u32>,
    slot_of_path: Vec<u32>,
    field_paths: Vec<u32>,
    prefix_off: Vec<u32>,
    prefix_len: Vec<u32>,
    suffix_off: u32,
    suffix_len: u32,
    output_kind: u32,
    output_path: u32,
    agg_kind: u32,
    agg_path: u32,
    group_path: u32,
    has_filter: u32,
    csv_mode: u32,
}

impl LoweredProgram {
    pub fn build(program: &Program, format: Format) -> LoweredProgram {
        let flat = FlatProgram::build(program);
        let mut blob = flat.blob.clone();
        let csv = format == Format::Csv;

        // Slot assignment: only the paths the kernel actually resolves get a
        // slot, so a query with one field does not carry a 16-wide table.
        let needed = program.required_paths();
        let mut slot_of_path = vec![ffi::NONE; program.paths.len()];
        let mut needed_paths = Vec::with_capacity(needed.len());
        for (i, &p) in needed.iter().enumerate() {
            slot_of_path[p as usize] = i as u32;
            needed_paths.push(p as u32);
        }

        let mut field_paths = Vec::new();
        let mut prefix_off = Vec::new();
        let mut prefix_len = Vec::new();
        let mut suffix_off = 0u32;
        let mut suffix_len = 0u32;

        let output_kind = match &program.output {
            Output::Passthrough => ffi::OUT_PASSTHROUGH,
            Output::Path(_) => ffi::OUT_PATH,
            Output::Agg { .. } => ffi::OUT_AGG,
            Output::Project(fields) => {
                for (i, (key, path)) in fields.iter().enumerate() {
                    field_paths.push(*path as u32);
                    let mut prefix = Vec::new();
                    if csv {
                        // CSV: no key names, just a comma between cells.
                        if i > 0 {
                            prefix.push(b',');
                        }
                    } else {
                        prefix.push(if i == 0 { b'{' } else { b',' });
                        write_json_string(&mut prefix, key.as_bytes());
                        prefix.push(b':');
                    }
                    prefix_off.push(blob.len() as u32);
                    prefix_len.push(prefix.len() as u32);
                    blob.extend_from_slice(&prefix);
                }
                if !csv {
                    suffix_off = blob.len() as u32;
                    suffix_len = 1;
                    blob.push(b'}');
                }
                ffi::OUT_PROJECT
            }
        };

        let output_path = match &program.output {
            Output::Path(p) => *p as u32,
            _ => ffi::NONE,
        };

        let (agg_kind, agg_path) = match &program.output {
            Output::Agg { kind, arg } => (
                match kind {
                    AggKind::Count => ffi::AGG_COUNT,
                    AggKind::Sum => ffi::AGG_SUM,
                    AggKind::Min => ffi::AGG_MIN,
                    AggKind::Max => ffi::AGG_MAX,
                    AggKind::Avg => ffi::AGG_AVG,
                },
                arg.map(|p| p as u32).unwrap_or(ffi::NONE),
            ),
            _ => (ffi::AGG_COUNT, ffi::NONE),
        };

        LoweredProgram {
            flat,
            blob,
            needed_paths,
            slot_of_path,
            field_paths,
            prefix_off,
            prefix_len,
            suffix_off,
            suffix_len,
            output_kind,
            output_path,
            agg_kind,
            agg_path,
            group_path: program.group_by.map(|p| p as u32).unwrap_or(ffi::NONE),
            has_filter: program.filter.is_some() as u32,
            csv_mode: csv as u32,
        }
    }

    /// The C view. Only valid while `self` is alive and unmoved, so callers
    /// keep the `LoweredProgram` boxed for exactly this reason.
    pub fn ffi_program(&self) -> ffi::Program {
        ffi::Program {
            steps: ptr_or_dangling(&self.flat.steps),
            n_steps: self.flat.steps.len() as u32,
            paths: ptr_or_dangling(&self.flat.paths),
            n_paths: self.flat.paths.len() as u32,
            cmps: ptr_or_dangling(&self.flat.cmps),
            n_cmps: self.flat.cmps.len() as u32,
            cond_rpn: ptr_or_dangling(&self.flat.cond_rpn),
            n_cond: self.flat.cond_rpn.len() as u32,
            blob: ptr_or_dangling(&self.blob),
            n_blob: self.blob.len() as u32,
            needed_paths: ptr_or_dangling(&self.needed_paths),
            n_needed: self.needed_paths.len() as u32,
            slot_of_path: ptr_or_dangling(&self.slot_of_path),
            output_kind: self.output_kind,
            output_path: self.output_path,
            agg_kind: self.agg_kind,
            agg_path: self.agg_path,
            group_path: self.group_path,
            has_filter: self.has_filter,
            n_fields: self.field_paths.len() as u32,
            field_paths: ptr_or_dangling(&self.field_paths),
            prefix_off: ptr_or_dangling(&self.prefix_off),
            prefix_len: ptr_or_dangling(&self.prefix_len),
            suffix_off: self.suffix_off,
            suffix_len: self.suffix_len,
            csv_mode: self.csv_mode,
        }
    }

    pub fn n_needed(&self) -> usize {
        self.needed_paths.len()
    }

    pub fn slot_of(&self, p: PathId) -> Option<u32> {
        let s = self.slot_of_path[p as usize];
        (s != ffi::NONE).then_some(s)
    }
}

fn ptr_or_dangling<T>(v: &[T]) -> *const T {
    if v.is_empty() {
        std::ptr::NonNull::dangling().as_ptr()
    } else {
        v.as_ptr()
    }
}

// Silence "field never read" for the tables that exist purely to be pointed at.
#[allow(dead_code)]
fn _keep_alive(_: &GpuStep, _: &GpuPath, _: &GpuCmp, _: &GpuCondOp) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::parse;

    #[test]
    fn slots_are_assigned_only_to_paths_the_query_uses() {
        let p = parse("select(.a == 1) | {x: .b}").unwrap();
        let l = LoweredProgram::build(&p, Format::Ndjson);
        assert_eq!(l.n_needed(), 2);
        for id in p.required_paths() {
            assert!(l.slot_of(id).is_some(), "path {id} has no slot");
        }
    }

    #[test]
    fn projection_prefixes_render_the_json_object() {
        let p = parse("{a: .x, b: .y}").unwrap();
        let l = LoweredProgram::build(&p, Format::Ndjson);
        let f = l.ffi_program();
        assert_eq!(f.output_kind, ffi::OUT_PROJECT);
        assert_eq!(f.n_fields, 2);
        let pre = |i: usize| {
            let off = l.prefix_off[i] as usize;
            let len = l.prefix_len[i] as usize;
            String::from_utf8(l.blob[off..off + len].to_vec()).unwrap()
        };
        assert_eq!(pre(0), "{\"a\":");
        assert_eq!(pre(1), ",\"b\":");
        let s = l.suffix_off as usize;
        assert_eq!(l.blob[s], b'}');
    }

    #[test]
    fn projection_keys_needing_escapes_are_escaped_on_the_host() {
        let p = parse(r#"{"a\"b": .x}"#).unwrap();
        let l = LoweredProgram::build(&p, Format::Ndjson);
        let off = l.prefix_off[0] as usize;
        let len = l.prefix_len[0] as usize;
        assert_eq!(
            String::from_utf8(l.blob[off..off + len].to_vec()).unwrap(),
            "{\"a\\\"b\":"
        );
    }

    #[test]
    fn csv_projection_uses_commas_and_no_keys() {
        let p = parse("{a: .x, b: .y}").unwrap();
        let l = LoweredProgram::build(&p, Format::Csv);
        assert_eq!(l.prefix_len[0], 0);
        assert_eq!(l.prefix_len[1], 1);
        assert_eq!(l.suffix_len, 0);
        assert_eq!(l.csv_mode, 1);
    }

    #[test]
    fn aggregate_lowering_marks_count_as_argumentless() {
        let p = parse("count").unwrap();
        let l = LoweredProgram::build(&p, Format::Ndjson);
        assert_eq!(l.agg_kind, ffi::AGG_COUNT);
        assert_eq!(l.agg_path, ffi::NONE);
        assert_eq!(l.group_path, ffi::NONE);

        let p = parse("group_by(.h) | sum(.b)").unwrap();
        let l = LoweredProgram::build(&p, Format::Ndjson);
        assert_eq!(l.agg_kind, ffi::AGG_SUM);
        assert_ne!(l.agg_path, ffi::NONE);
        assert_ne!(l.group_path, ffi::NONE);
    }

    #[test]
    fn empty_tables_still_yield_non_null_pointers() {
        // A query with no filter has no cmps; passing a null pointer for an
        // empty array is fine in C but trips Rust's slice invariants if it
        // ever comes back, so we hand over a dangling-but-aligned pointer.
        let p = parse(".").unwrap();
        let l = LoweredProgram::build(&p, Format::Ndjson);
        let f = l.ffi_program();
        assert!(!f.cmps.is_null());
        assert_eq!(f.n_cmps, 0);
        assert_eq!(f.has_filter, 0);
    }
}
