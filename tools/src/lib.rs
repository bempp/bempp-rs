//! Tools for interfacing Rust via CFFI
#![cfg_attr(feature = "strict", deny(warnings))]

pub mod arrays;
pub mod containers;
pub mod types;

pub use arrays::*;
pub use containers::*;
pub use types::*;
