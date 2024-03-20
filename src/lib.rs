//! Bempp
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

#[macro_use]
extern crate lazy_static;

pub mod bem;
pub mod element;
pub mod grid;
pub mod quadrature;
pub mod traits;
