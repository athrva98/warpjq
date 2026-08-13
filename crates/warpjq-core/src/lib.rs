//! warpjq core: query compiler plus CPU and CUDA execution backends.
//!
//! The crate is arranged so that query *meaning* lives in exactly one place
//! ([`query`]) and the backends are interchangeable implementations of
//! [`exec::Backend`]. That is what lets `tests/differential.rs` assert
//! GPU output == CPU output == jq output on fuzzed input.

pub mod agg;
pub mod chunk;
pub mod error;
pub mod exec;
pub mod gen;
pub mod json;
pub mod output;
pub mod query;

#[cfg(feature = "cuda")]
pub mod gpu;

pub use error::{Result, WarpError};
pub use query::{parse, Program};
