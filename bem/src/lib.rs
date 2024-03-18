//! A Rust grid library
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod assembly;
pub mod function_space;

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
