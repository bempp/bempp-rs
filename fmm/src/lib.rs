//! A a general framework for implementing Fast Multipole Methods.
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod builder;
pub mod constants;
pub mod fmm;
pub mod helpers;
pub mod pinv;
pub mod send_ptr;
pub mod tree;
pub mod types;

mod field_translation {
    pub mod matmul;
    pub mod source;
    pub mod source_to_target {
        pub mod blas;
        pub mod fft;
    }
    pub mod target;
}
