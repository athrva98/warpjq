//! The query compiler must reject bad input with an error, never a panic.
//!
//! A CLI that panics on a typo prints a Rust backtrace at the user, which is
//! precisely the "error messages that look like a stack trace" failure the
//! project set out to avoid.

#![no_main]

use libfuzzer_sys::fuzz_target;
use warpjq_core::query::FlatProgram;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    match warpjq_core::parse(text) {
        Err(e) => {
            // The caret offset is used to slice the source for display, so it
            // must always land on a character boundary inside it.
            assert!(e.at <= e.source_text.len());
            assert!(e.source_text.is_char_boundary(e.at));
            // Rendering must not panic either.
            let _ = e.to_string();
        }
        Ok(program) => {
            // Anything that parses must also lower and flatten cleanly.
            let flat = FlatProgram::build(&program);
            for p in &flat.paths {
                assert!(
                    (p.step_off as usize + p.step_count as usize) <= flat.steps.len()
                );
            }
            for c in &flat.cmps {
                assert!((c.lit_off as usize + c.lit_len as usize) <= flat.blob.len());
                assert!((c.path as usize) < flat.paths.len());
            }
            for s in &flat.steps {
                assert!((s.key_off as usize + s.key_len as usize) <= flat.blob.len());
            }
            assert!(flat.cond_stack_depth as usize <= flat.cond_rpn.len() + 1);
            let _ = program.required_paths();
            let _ = program.csv_headers();
        }
    }
});
