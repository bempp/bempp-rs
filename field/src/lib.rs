//! Functionality for computing the metadata required for field translations for the kernel independent fast multipole method.
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod array;
pub mod fft;
pub mod field;
pub mod helpers;
pub mod transfer_vector;
pub mod types;

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
