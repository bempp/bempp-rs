//! Bempp
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

#[macro_use]
extern crate lazy_static;

pub mod assembly;
pub mod function;
pub mod quadrature;
pub mod traits;
