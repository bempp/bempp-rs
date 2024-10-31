//! Bempp
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

pub mod assembly;
pub mod bindings;
pub mod function;
pub mod traits;

#[cfg(test)]
mod test {
    use criterion as _; // Hack to show that criterion is used, as cargo test does not see benches
}
