//! Bempp
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

#[macro_use]
extern crate lazy_static;

// pub mod assembly;
pub mod element;
// pub mod function;
pub mod grid;
pub mod quadrature;
pub mod traits;
pub mod types;
