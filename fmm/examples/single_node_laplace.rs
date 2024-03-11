use bempp_field::types::{BlasFieldTranslationKiFmm, FftFieldTranslationKiFmm};
use bempp_fmm::builder::KiFmmBuilderSingleNode;
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::fmm::Fmm;
use bempp_tree::implementations::helpers::points_fixture;
use rlst_dense::{rlst_dynamic_array2, traits::RawAccessMut};

fn main() {
    // Setup random sources and targets
    let nsources = 50000;
    let ntargets = 10000;
    let sources = points_fixture::<f64>(nsources, None, None, Some(0));
    let targets = points_fixture::<f64>(ntargets, None, None, Some(3));

    // FMM parameters
    let n_crit = Some(10);
    let expansion_order = 7;
    let sparse = true;

    // FFT based M2L for a vector of charges
    {
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let fmm_fft = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, n_crit, sparse)
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                FftFieldTranslationKiFmm::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_fft.evaluate();
    }

    // BLAS based M2L
    {
        // Vector of charges
        let nvecs = 1;
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .enumerate()
            .for_each(|(i, chunk)| chunk.iter_mut().for_each(|elem| *elem += (1 + i) as f64));

        let singular_value_threshold = Some(1e-5);

        let fmm_vec = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, n_crit, sparse)
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                BlasFieldTranslationKiFmm::new(singular_value_threshold),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_vec.evaluate();

        // Matrix of charges
        let nvecs = 5;
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);

        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .enumerate()
            .for_each(|(i, chunk)| chunk.iter_mut().for_each(|elem| *elem += (1 + i) as f64));

        let fmm_mat = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, n_crit, sparse)
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                BlasFieldTranslationKiFmm::new(singular_value_threshold),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_mat.evaluate();
    }
}
