//! Single Node FMM using an SVD based M2L translation operator
use std::time::Instant;

use itertools::Itertools;

use rlst_dense::traits::RawAccess;

use bempp_field::types::SvdFieldTranslationKiFmmRcmp;
use bempp_fmm::{
    charge::build_charge_dict,
    types::{FmmDataUniform, KiFmmLinear},
};
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::{fmm::{Fmm, FmmLoop}, tree::Tree};
use bempp_tree::implementations::helpers::points_fixture;
use bempp_tree::types::single_node::SingleNodeTree;

fn time_f32(digits: usize, npoints: usize, depth: u64, sparse: bool) {

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
    let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

    // Form charge dict, matching charges with their associated global indices
    let charge_dict = build_charge_dict(&global_idxs, &charges);

    let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

    let s = Instant::now();
    let times = datatree.run(true);
    println!("runtime {:?} operators {:?} nkeys {:?}", s.elapsed(), times, datatree.fmm.tree().get_all_leaves().unwrap().len());
}


fn time_f64(digits: usize, npoints: usize, depth: u64, sparse: bool) {

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
    let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

    // Form charge dict, matching charges with their associated global indices
    let charge_dict = build_charge_dict(&global_idxs, &charges);

    let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

    let s = Instant::now();
    let times = datatree.run(true);
    println!("runtime {:?} operators {:?} nkeys {:?}", s.elapsed(), times, datatree.fmm.tree().get_all_leaves().unwrap().len());
}



fn main() {

    let order = 10;
    let npoints = 1000000;
    let depth = 5;
    let sparse = true;

    time_f64(order, npoints, depth, sparse);

}
