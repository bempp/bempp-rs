use std::time::Instant;

use itertools::Itertools;

use rlst_dense::rlst_array_from_slice2;
use rlst_dense::traits::RawAccess;

use bempp_field::types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm};
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_tree::implementations::helpers::{points_fixture, points_fixture_sphere};
use bempp_tree::{constants::ROOT, types::single_node::SingleNodeTree};
use bempp_fmm::{types::{KiFmmLinear, FmmDataUniform}, charge::build_charge_dict};
use bempp_traits::{
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, FmmLoop, KiFmm as KiFmmTrait, SourceTranslation, TargetTranslation, TimeDict},
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
    types::EvalType,
};


fn main () {

        let npoints = 100000;

        let points = points_fixture::<f32>(npoints, None, None);

        let global_idxs = (0..npoints).collect_vec();
        let charges = vec![1.0; npoints];

        let order = 6;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let depth = 5;

        let tree = SingleNodeTree::new(points.data(), false, None, Some(depth), &global_idxs, true);

        let kernel = Laplace3dKernel::default();
        let m2l_data: FftFieldTranslationKiFmm<f32, Laplace3dKernel<f32>> =
            FftFieldTranslationKiFmm::new(kernel.clone(), order, *tree.get_domain(), alpha_inner);
        let m2l_data = SvdFieldTranslationKiFmm::new(
            kernel.clone(),
            Some(1000),
            order,
            *tree.get_domain(),
            alpha_inner,
        );

        let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

        // Form charge dict, matching charges with their associated global indices
        let charge_dict = build_charge_dict(&global_idxs, &charges);

        let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

        let s = Instant::now();
        let times = datatree.run(true);
        println!("runtime {:?} operators {:?}", s.elapsed(), times);

}