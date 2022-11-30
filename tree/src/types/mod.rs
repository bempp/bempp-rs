//! # Type declaration
pub mod domain;
pub mod morton;
pub mod point;
pub mod single_node;
#[cfg(feature = "mpi")]
pub mod  multi_node;