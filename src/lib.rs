//! Bempp
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

#[macro_use]
extern crate lazy_static;

pub mod assembly;
pub mod bindings;
pub mod function;
pub mod quadrature;
pub mod traits;

#[cfg(test)]
mod test {
    use criterion as _; // Hack to show that criterion is used, as cargo test does not see benches
}
