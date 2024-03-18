//! A Rust grid library
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod common;
pub mod flat_triangle_grid;
pub mod io;
pub mod mixed_grid;
pub mod shapes;
pub mod single_element_grid;
pub mod traits;
pub mod traits_impl;

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
