//! The JSON scanner must never panic, slice out of bounds, or loop forever on
//! arbitrary bytes, including invalid UTF-8, truncated escapes, and nesting
//! deep enough to blow a recursive parser's stack.
//!
//! "GPU parsers that segfault on bad input get roasted publicly", and the CPU
//! scanner here is the exact algorithm the kernel implements, so a crash found
//! by this target is very likely a crash in the kernel too.

#![no_main]

use libfuzzer_sys::fuzz_target;
use warpjq_core::json::{self, Lookup};
use warpjq_core::query::Step;

fuzz_target!(|data: &[u8]| {
    // Whatever the verdict, it must be reached without panicking.
    let valid = json::validate(data).is_ok();

    let paths: [&[Step]; 5] = [
        &[],
        &[Step::Key(String::new())],
        &[Step::Index(0)],
        &[Step::Index(u32::MAX)],
        &[],
    ];

    for steps in paths {
        match json::lookup(data, steps) {
            Lookup::Found(slot) => {
                // A found slot must point inside the input, and a string slot
                // must really be quoted, because str_inner() slices on that promise.
                let raw = slot.raw;
                if !raw.is_empty() {
                    let base = data.as_ptr() as usize;
                    let off = raw.as_ptr() as usize;
                    assert!(
                        off >= base && off + raw.len() <= base + data.len()
                            || raw == b"null",
                        "slot escaped the input buffer"
                    );
                }
                if slot.kind == json::Kind::Str {
                    assert!(raw.len() >= 2 && raw[0] == b'"' && raw[raw.len() - 1] == b'"');
                    let _ = slot.str_inner();
                }
                let _ = slot.as_f64();
                let _ = slot.is_truthy();
            }
            Lookup::TypeError(_) | Lookup::Invalid(_) => {}
        }
    }

    // A key with awkward bytes must not break key matching.
    let dynamic = String::from_utf8_lossy(&data[..data.len().min(16)]).into_owned();
    let _ = json::lookup(data, &[Step::Key(dynamic)]);

    // Unescaping arbitrary bytes must terminate and stay in bounds.
    let mut buf = Vec::new();
    let _ = json::unescape_into(data, &mut buf);

    let _ = valid;
});
