//! A a general framework for implementing Fast Multipole Methods.
pub mod builder;
pub mod constants;
pub mod fmm;
pub mod helpers;
pub mod pinv;
pub mod traits;
pub mod tree;
pub mod types;

mod field_translation {
    pub mod hadamard;
    pub mod source;
    pub mod source_to_target {
        pub mod blas;
        pub mod fft;
    }
    pub mod target;
}
