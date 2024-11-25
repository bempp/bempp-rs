//! Bempp
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

pub mod bindings;
pub mod boundary_assemblers;
pub mod function;
pub mod helmholtz;
pub mod laplace;

#[cfg(test)]
mod test {
    use approx as _;
    use cauchy as _;
    use criterion as _; // Hack to show that criterion is used, as cargo test does not see benches
}
