//! A a general framework for implementing Fast Multipole Methods.
pub mod builder;
pub mod charge;
pub mod constants;
pub mod fmm;
pub mod helpers;
pub mod interaction_lists;
pub mod pinv;
pub mod types;

mod field_translation {
    pub mod hadamard;
    pub mod source;
    pub mod source_to_target {
        pub mod fft;
        pub mod svd;
    }
    pub mod target;
}


// Temporary files to be moved over onto main source
pub mod new_types;
pub mod tree;