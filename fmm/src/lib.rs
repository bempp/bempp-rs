//! A a general framework for implementing Fast Multipole Methods.
pub mod charge;
pub mod constants;
pub mod fmm;
pub mod interaction_lists;
pub mod pinv;
pub mod types;

mod field_translation {
    pub mod hadamard;
    pub mod source;
    pub mod source_to_target;
    pub mod target;
}
