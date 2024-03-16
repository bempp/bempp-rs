//! Finite elements
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod element;
pub mod polynomials;
pub mod reference_cell;
//pub mod map;

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
