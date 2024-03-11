//! A a general framework for implementing Fast Multipole Methods.
pub mod builder;
pub mod constants;
pub mod fmm;
pub mod helpers;
pub mod pinv;
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
