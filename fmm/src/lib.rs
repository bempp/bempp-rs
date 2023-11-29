//! A a general framework for implementing Fast Multipole Methods.
pub mod charge;
pub mod constants;
pub mod interaction_lists;
pub mod pinv;
pub mod types;

mod field_translation {
    pub mod hashmap {
        pub mod source;
        pub mod source_to_target;
        pub mod target;
    }
    pub mod linear {
        pub mod source;
        pub mod source_to_target;
        pub mod target;
    }
}

mod fmm {
    pub mod hashmap;
    pub mod linear;
}
