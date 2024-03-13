//! Helpers

use bempp_traits::{
    kernel::Kernel,
    types::{EvalType, RlstScalar},
};

pub(crate) fn check_dimensions_evaluate<K: Kernel, T: RlstScalar>(
    kernel: &K,
    eval_type: EvalType,
    sources: &[T::Real],
    targets: &[T::Real],
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
        nsources,
        "Wrong dimension for `charges`. {} != {} ",
        charges.len(),
        nsources,
    );

    assert_eq!(
        result.len(),
        kernel.range_component_count(eval_type) * ntargets,
        "Wrong dimension for `result`. {} != {} ",
        result.len(),
        ntargets * kernel.range_component_count(eval_type),
    );
}

pub(crate) fn check_dimensions_assemble<K: Kernel, T: RlstScalar>(
    kernel: &K,
    eval_type: EvalType,
    sources: &[T::Real],
    targets: &[T::Real],
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
        kernel.range_component_count(eval_type) * nsources * ntargets,
        "Wrong dimension for `result`. {} != {} ",
        result.len(),
        nsources * ntargets * kernel.range_component_count(eval_type),
    );
}

pub(crate) fn check_dimensions_assemble_diagonal<K: Kernel, T: RlstScalar>(
    kernel: &K,
    eval_type: EvalType,
    sources: &[T::Real],
    targets: &[T::Real],
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
        nsources, ntargets,
        "Lengths of sources and targets do not match. {} != {} ",
        nsources, ntargets,
    );

    assert_eq!(
        result.len(),
        kernel.range_component_count(eval_type) * nsources,
        "Wrong dimension for `result`. {} != {} ",
        result.len(),
        nsources * kernel.range_component_count(eval_type),
    );
}
