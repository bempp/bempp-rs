//! rusty-quadrature - A simple boundary element quadrature library for Rust
#![cfg_attr(feature = "strict", deny(warnings))]

#[macro_use]
extern crate lazy_static;

pub mod duffy;
pub mod simplex_rule_definitions;
pub mod simplex_rules;
pub mod types;
