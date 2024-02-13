//! Single Node FMM using an SVD based M2L operator with individual SVDs found
//! for each M2L operator for an FMM taking matrix input.

use std::time::Instant;

use bempp_fmm::types::{FmmDataUniformMatrix, KiFmmLinearMatrix};
use itertools::Itertools;

use rlst_dense::traits::RawAccess;

use bempp_field::types::SvdFieldTranslationKiFmmRcmp;
use bempp_fmm::charge::build_charge_dict;
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::{fmm::FmmLoop, tree::Tree};
use bempp_tree::implementations::helpers::points_fixture;
use bempp_tree::types::single_node::SingleNodeTree;
use bempp_traits::fmm::Fmm;


fn time_f32(digits: usize, npoints: usize, depth: u64, sparse: bool, ncharge_vecs: usize) {

    let points = points_fixture::<f32>(npoints, None, None);

    let global_idxs = (0..npoints).collect_vec();
    let charges = vec![1.0; npoints];

    let alpha_inner = 1.05;
    let alpha_outer = 2.95;

    // 6 digits
    let order;
    let k;
    let threshold;

    if digits == 3 {
        order = 3;
        k = 10;
        threshold = 0.99967;
    } else if digits == 5 {
        order = 5;
        k = 47;
        threshold = 0.999999;
    } else {
        panic!("only 3 or 5 valid")
    }

    let mut charge_mat = vec![vec![0.0; npoints]; ncharge_vecs];
    charge_mat
        .iter_mut()
        .enumerate()
        .for_each(|(i, charge_mat_i)| *charge_mat_i = vec![i as f32 + 1.0; npoints]);


    let tree = SingleNodeTree::new(points.data(), false, None, Some(depth), &global_idxs, sparse);
    let kernel = Laplace3dKernel::default();
    let s = Instant::now();
    let m2l_data = SvdFieldTranslationKiFmmRcmp::new(
        kernel.clone(),
        Some(k),
        threshold,
        order,
        *tree.get_domain(),
        alpha_inner,
    );
    println!("precomputation time {:?}", s.elapsed());
    let fmm = KiFmmLinearMatrix::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

    // Form charge dict, matching charges with their associated global indices
    // Form charge dict, matching charges with their associated global indices
    let charge_dicts: Vec<_> = (0..ncharge_vecs)
    .map(|i| build_charge_dict(&global_idxs, &charge_mat[i]))
    .collect();

    // Associate data with the FMM
    let datatree = FmmDataUniformMatrix::new(fmm, &charge_dicts).unwrap();


    let s = Instant::now();
    let times = datatree.run(true);
    println!("runtime {:?} operators {:?} nkeys {:?}", s.elapsed(), times, datatree.fmm.tree().get_all_leaves().unwrap().len());
}


fn time_f64(digits: usize, npoints: usize, depth: u64, sparse: bool, ncharge_vecs: usize) {

    let points = points_fixture::<f64>(npoints, None, None);

    let global_idxs = (0..npoints).collect_vec();
    let charges = vec![1.0; npoints];

    let alpha_inner = 1.05;
    let alpha_outer = 2.95;

    // 6 digits
    let order;
    let k;
    let threshold;
    let test_val;

    if digits == 6 {
        order = 6;
        k = 68;
        threshold = 0.9999999;
        test_val = 1e-6;
    } else if digits == 8 {
        order = 8;
        k = 133;
        threshold = 0.999999999999;
        test_val = 1e-8;
    } else if digits == 10 {
        order = 10;
        k = 226;
        threshold = 0.999999999999998;
        test_val = 1e-10;
    } else {
        panic!("only 6, 8 or 10 valid")
    }

    let mut charge_mat = vec![vec![0.0; npoints]; ncharge_vecs];
    charge_mat
        .iter_mut()
        .enumerate()
        .for_each(|(i, charge_mat_i)| *charge_mat_i = vec![i as f64 + 1.0; npoints]);

    // SVD based field translations
    let kernel = Laplace3dKernel::default();

    // Create a tree
    let tree = SingleNodeTree::new(points.data(), false, None, Some(depth), &global_idxs, sparse);

    let m2l_data = SvdFieldTranslationKiFmmRcmp::new(
        kernel.clone(),
        Some(k),
        threshold,
        order,
        *tree.get_domain(),
        alpha_inner,
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
    println!("runtime {:?} operators {:?} nkeys {:?}", s.elapsed(), times, datatree.fmm.tree().get_all_leaves().unwrap().len());
}


fn main() {
    // time_f64(10, 1000000, 5, true, 10)
    time_f32(5, 1000000, 4, true, 10)
}
