//! Bindings for C

#![allow(missing_docs)]
#![allow(clippy::missing_safety_doc)]
use c_api_tools::cfuncs;

pub mod boundary_assemblers;
pub mod matrices;
pub mod ndelement;
pub mod ndgrid;
pub mod space;

#[cfuncs(name = "matrix_t", create, free, unwrap)]
pub struct MatrixT;
