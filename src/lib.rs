//! Bempp
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

//pub mod bindings;
pub mod boundary_assemblers;
pub mod boundary_evaluators;
pub mod evaluator_tools;
pub mod function;
pub mod greens_function_evaluators;
pub mod helmholtz;
pub mod laplace;
pub mod shapes;

#[cfg(test)]
mod test {
    use approx as _;
    use cauchy as _;
    use criterion as _; // Hack to show that criterion is used, as cargo test does not see benches
}
