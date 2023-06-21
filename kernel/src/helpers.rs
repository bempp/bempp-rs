use crate::traits::Kernel;
use crate::types::EvalType;
use bempp_traits::types::Scalar;

pub(crate) fn check_dimensions_evaluate<K: Kernel, T: Scalar>(
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
