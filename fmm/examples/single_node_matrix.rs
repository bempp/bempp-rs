use std::time::Instant;

use bempp_fmm::types::FmmDataUniformMatrix;
use itertools::Itertools;

use rlst_dense::traits::RawAccess;

use bempp_field::types::{SvdFieldTranslationKiFmm, SvdFieldTranslationKiFmmIA};
use bempp_fmm::charge::build_charge_dict;
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::{fmm::FmmLoop, tree::Tree};
use bempp_tree::implementations::helpers::points_fixture;
use bempp_tree::types::single_node::SingleNodeTree;

fn main() {
    let npoints = 1000000;

    let global_idxs = (0..npoints).collect_vec();

    let order = 6;
    let alpha_inner = 1.05;
    let alpha_outer = 2.95;

    // Test matrix input
    let points = points_fixture::<f32>(npoints, None, None);
    let ncharge_vecs = 5;
    let depth = 4;

    let mut charge_mat = vec![vec![0.0; npoints]; ncharge_vecs];
    charge_mat
        .iter_mut()
        .enumerate()
        .for_each(|(i, charge_mat_i)| *charge_mat_i = vec![i as f32 + 1.0; npoints]);

    // SVD based field translations
    let kernel = Laplace3dKernel::default();

    // Create a tree
    let tree = SingleNodeTree::new(points.data(), false, None, Some(depth), &global_idxs, true);

    // Precompute the M2L data
    // let m2l_data = SvdFieldTranslationKiFmm::new(
    //     kernel.clone(),
    //     Some(80),
    //     order,
    //     *tree.get_domain(),
    //     alpha_inner,
    // );

    let m2l_data = SvdFieldTranslationKiFmmIA::new(
        kernel.clone(),
        0.9999,
        order,
        *tree.get_domain(),
        alpha_inner
    );

    let fmm = bempp_fmm::types::KiFmmLinearMatrix::new(
        order,
        alpha_inner,
        alpha_outer,
        kernel,
        tree,
        m2l_data,
    );

    // Form charge dict, matching charges with their associated global indices
    let charge_dicts: Vec<_> = (0..ncharge_vecs)
        .map(|i| build_charge_dict(&global_idxs, &charge_mat[i]))
        .collect();

    // Associate data with the FMM
    let datatree = FmmDataUniformMatrix::new(fmm, &charge_dicts).unwrap();

    let s = Instant::now();
    let times = datatree.run(true);
    println!("runtime {:?} operators {:?}", s.elapsed(), times);
}
