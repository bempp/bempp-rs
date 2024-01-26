//! Single Node FMM using an SVD based M2L translation operator
use std::time::Instant;

use itertools::Itertools;

use rlst_dense::traits::RawAccess;

use bempp_field::types::SvdFieldTranslationKiFmm;
use bempp_fmm::{
    charge::build_charge_dict,
    types::{FmmDataUniform, KiFmmLinear},
};
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::{fmm::FmmLoop, tree::Tree};
use bempp_tree::implementations::helpers::points_fixture;
use bempp_tree::types::single_node::SingleNodeTree;

fn main() {
    let npoints = 10000;

    let points = points_fixture::<f32>(npoints, None, None);

    let global_idxs = (0..npoints).collect_vec();
    let charges = vec![1.0; npoints];

    let order = 6;
    let alpha_inner = 1.05;
    let alpha_outer = 2.95;
    let depth = 3;

    let tree = SingleNodeTree::new(points.data(), false, None, Some(depth), &global_idxs, true);

    let kernel = Laplace3dKernel::default();

    let m2l_data = SvdFieldTranslationKiFmm::new(
        kernel.clone(),
        Some(70),
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
