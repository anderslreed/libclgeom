//! Geometry computations using ``OpenCL``

#![warn(
    clippy::cargo,
    clippy::nursery,
    clippy::panic,
    clippy::pedantic,
    clippy::restriction
)]
#![allow(
    clippy::blanket_clippy_restriction_lints,
    clippy::implicit_return,
    clippy::missing_docs_in_private_items,
    clippy::module_name_repetitions,

    /*
        clippy::as_conversions,
        clippy::expect_used,
        clippy::multiple_crate_versions,
        clippy::pattern_type_mismatch,
        clippy::undocumented_unsafe_blocks,
    */
)]

mod compile;
mod context;
mod errors;
mod ffi;
mod mesh;
