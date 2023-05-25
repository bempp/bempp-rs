//! Trait for Green's function kernels
use bempp_traits::types::{c64, Scalar};
use rayon::ThreadPool;

pub trait Kernel {
    type T: Scalar;

    fn evaluate_st(
        &self,
        eval_type: EvalType,
        sources: &[f64],
        targets: &[f64],
        charges: &[Self::T],
        result: &mut [Self::T],
    );

    fn evaluate_mt(
        &self,
        eval_type: EvalType,
        sources: &[f64],
        targets: &[f64],
        charges: &[Self::T],
        result: &mut [Self::T],
        thread_pool: &ThreadPool,
    );

    fn kernel_type(&self) -> &KernelType;

    fn domain_component_count(&self) -> usize;

    fn space_dimension(&self) -> usize;

    fn range_component_count(&self, eval_type: EvalType) -> usize;
}

// Evaluation Mode.
//
// - `Value`: Declares that only values required.
// - `Deriv`: Declare that only derivative required.
// - `ValueDeriv` Both values and derivatives required.
#[derive(Clone, Copy)]
pub enum EvalType {
    Value,
    ValueDeriv,
}

/// This enum defines the type of the kernel.
#[derive(Clone, Copy)]
pub enum KernelType {
    /// The Laplace kernel defined as g(x, y) = 1 / (4 pi | x- y| )
    Laplace,
    /// The Helmholtz kernel defined as g(x, y) = exp( 1j * k * | x- y| ) / (4 pi | x- y| )
    Helmholtz(c64),
    /// The modified Helmholtz kernel defined as g(x, y) = exp( -omega * | x- y| ) / (4 * pi * | x- y |)
    ModifiedHelmholtz(f64),
}

pub(crate) fn check_dimensions_evaluate<K: Kernel, T: Scalar>(
    kernel: &K,
    eval_type: EvalType,
    sources: &[f64],
    targets: &[f64],
    charges: &[T],
    result: &[T],
) {
    assert!(
        sources.len() % kernel.space_dimension() == 0,
        "Length of sources {} is not a multiple of space dimension {}.",
        sources.len(),
        kernel.space_dimension()
    );

    assert!(
        targets.len() % kernel.space_dimension() == 0,
        "Length of targets {} is not a multiple of space dimension {}.",
        sources.len(),
        kernel.space_dimension()
    );

    let nsources = sources.len() / kernel.space_dimension();
    let ntargets = targets.len() / kernel.space_dimension();

    assert_eq!(
        charges.len(),
        kernel.domain_component_count() * nsources,
        "Wrong dimension for `charges`. {} != {} ",
        charges.len(),
        nsources * kernel.domain_component_count(),
    );

    assert_eq!(
        result.len(),
        kernel.range_component_count(eval_type) * ntargets,
        "Wrong dimension for `result`. {} != {} ",
        result.len(),
        ntargets * kernel.range_component_count(eval_type),
    );
}

pub(crate) fn check_dimensions_assemble<K: Kernel, T: Scalar>(
    kernel: &K,
    eval_type: EvalType,
    sources: &[f64],
    targets: &[f64],
    result: &[T],
) {
    assert!(
        sources.len() % kernel.space_dimension() == 0,
        "Length of sources {} is not a multiple of space dimension {}.",
        sources.len(),
        kernel.space_dimension()
    );

    assert!(
        targets.len() % kernel.space_dimension() == 0,
        "Length of targets {} is not a multiple of space dimension {}.",
        sources.len(),
        kernel.space_dimension()
    );

    let nsources = sources.len() / kernel.space_dimension();
    let ntargets = targets.len() / kernel.space_dimension();

    assert_eq!(
        result.len(),
        nsources * ntargets * kernel.range_component_count(eval_type),
        "Wrong dimension for `result`. {} != {} ",
        result.len(),
        nsources * ntargets * kernel.range_component_count(eval_type),
    );
}

pub(crate) fn diff(target: &[f64], sources: &[f64], dim: usize, result: &mut [f64]) {
    assert_eq!(
        target.len(),
        dim,
        "Target length {} is not identical to dimension {}",
        target.len(),
        dim
    );
    assert_eq!(
        sources.len() % dim,
        0,
        "Sources length {} is not a multiple of dimension {}",
        sources.len(),
        dim
    );

    let nsources = sources.len() / dim;

    assert_eq!(
        result.len(),
        nsources,
        "Result length {} is not equal to number of sources {}",
        result.len(),
        nsources
    );

    for index in 0..result.len() {
        result[index] = 0.0;
    }

    for index in 0..result.len() {
        result[index] = ((target[0] - sources[index]) * (target[0] - sources[index])
            + (target[1] - sources[nsources + index]) * (target[1] - sources[nsources + index])
            + (target[2] - sources[2 * nsources + index])
                * (target[2] - sources[2 * nsources + index]))
            .sqrt();
    }
}
