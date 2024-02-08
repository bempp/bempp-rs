//! Single Node FMM using an SVD based M2L translation operator
use std::time::Instant;

use itertools::Itertools;

use rlst_dense::{array::{empty_array, Array}, base_array::BaseArray, data_container::VectorContainer, rlst_array_from_slice2, rlst_dynamic_array2, traits::{MultIntoResize, RawAccess}};

use bempp_field::types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm, SvdFieldTranslationKiFmmIA, SvdFieldTranslationKiFmmRcmp};
use bempp_fmm::{
    charge::build_charge_dict,
    types::{FmmDataUniform, KiFmmLinear},
};
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::{field::FieldTranslationData, fmm::{FmmLoop, InteractionLists}, tree::Tree, types::EvalType};
use bempp_tree::implementations::helpers::points_fixture;
use bempp_tree::types::single_node::SingleNodeTree;

use bempp_traits::kernel::Kernel;
use bempp_traits::fmm::Fmm;

use bempp_traits::field::FieldTranslation;
use bempp_traits::kernel::ScaleInvariantKernel;

fn test_uniform_f64_svd_rcmp(
    points: &Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    charges: &[f64],
    global_idxs: &[usize],
    order: usize,
    alpha_inner: f64,
    alpha_outer: f64,
    sparse: bool,
    depth: u64,
    k: usize,
    threshold: f64,
    test_val: f64
) -> f64 {
    // Test with SVD field translation
    let tree =
        SingleNodeTree::new(points.data(), false, None, Some(depth), global_idxs, sparse);

    let kernel = Laplace3dKernel::default();

    let m2l_data = SvdFieldTranslationKiFmmRcmp::new(
        kernel.clone(),
        Some(k),
        threshold,
        order,
        *tree.get_domain(),
        alpha_inner,
    );

    let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

    // Form charge dict, matching charges with their associated global indices
    let charge_dict = build_charge_dict(global_idxs, charges);

    let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

    datatree.run(false);

    // Test that direct computation is close to the FMM.
    let mut test_idx_vec = Vec::new();
    for (idx, index_pointer) in datatree.charge_index_pointer.iter().enumerate() {
        if index_pointer.1 - index_pointer.0 > 0 {
            test_idx_vec.push(idx);
        }
    }
    let leaf = &datatree.fmm.tree().get_all_leaves().unwrap()[test_idx_vec[0]];
    let leaf_check_surface = leaf.compute_surface(&datatree.fmm.tree.domain, order, alpha_inner);
    let leaf_equivalent_surface = leaf.compute_surface(&datatree.fmm.tree.domain, order, alpha_outer);

    let v_list = datatree.fmm.get_v_list(leaf).unwrap();
    let ncoeffs = datatree.fmm.m2l.ncoeffs(order);

    let mut direct = vec![0f64; ncoeffs];

    for source in v_list.iter() {
        let idx = datatree.level_index_pointer[leaf.level() as usize].get(source).unwrap();
        let source_multipole_ptr = datatree.level_multipoles[source.level() as usize][*idx];
        let source_multipole = unsafe {
            std::slice::from_raw_parts_mut(source_multipole_ptr.raw, ncoeffs)
        };
        let scaled_multipole = source_multipole
            .iter()
            .map(|d| *d*datatree.fmm.kernel.scale(leaf.level()) ).collect_vec();

        let source_equivalent_surface = source.compute_surface(&datatree.fmm.tree.domain, order, alpha_inner);

        // Evaluate check potential directly
        datatree.fmm.kernel.evaluate_st(EvalType::Value, &source_equivalent_surface, &leaf_check_surface, &scaled_multipole, &mut direct)
    }

    let check_potential = rlst_array_from_slice2!(f64, direct.as_slice(), [ncoeffs, 1]);
    // Compute local expansion
    let local_direct = empty_array::<f64, 2>().simple_mult_into_resize(
        datatree.fmm.dc2e_inv_1.view(),
        empty_array::<f64, 2>().simple_mult_into_resize(
            datatree.fmm.dc2e_inv_2.view(),
            check_potential.view()
        )
    );

    let idx = datatree.level_index_pointer[leaf.level() as usize].get(leaf).unwrap();
    let leaf_local_ptr = datatree.level_locals[leaf.level() as usize][*idx];
    let leaf_local = unsafe {
        std::slice::from_raw_parts_mut(leaf_local_ptr.raw, ncoeffs)
    };


    let point = leaf.centre(datatree.fmm.tree().get_domain());
    let point = rlst_array_from_slice2!(f64, point.as_slice(), [3, 1]);
    let mut r1 = vec![0.];
    datatree.fmm.kernel.evaluate_st(EvalType::Value, &leaf_equivalent_surface, point.data(), local_direct.data(), &mut r1);

    let mut r2 = vec![0.];
    datatree.fmm.kernel.evaluate_st(EvalType::Value, &leaf_equivalent_surface, point.data(), leaf_local, &mut r2);

    println!("{:?} {:?}", r1, r2);
    let err = (r1[0]-r2[0]).abs()/r1[0];
    err
}

fn test_uniform_f32_svd_rcmp(
    points: &Array<f32, BaseArray<f32, VectorContainer<f32>, 2>, 2>,
    charges: &[f32],
    global_idxs: &[usize],
    order: usize,
    alpha_inner: f32,
    alpha_outer: f32,
    sparse: bool,
    depth: u64,
    k: usize,
    threshold: f32,
    test_val: f32
) -> f32 {
    // Test with SVD field translation
    let tree =
        SingleNodeTree::new(points.data(), false, None, Some(depth), global_idxs, sparse);

    let kernel = Laplace3dKernel::default();

    let m2l_data = SvdFieldTranslationKiFmmRcmp::new(
        kernel.clone(),
        Some(k),
        threshold,
        order,
        *tree.get_domain(),
        alpha_inner,
    );

    // let m2l_data = SvdFieldTranslationKiFmm::new(
    //     kernel.clone(),
    //     Some(k),
    //     order,
    //     *tree.get_domain(),
    //     alpha_inner,
    // );

    // let m2l_data = FftFieldTranslationKiFmm::new(
    //     kernel.clone(),
    //     order,
    //     *tree.get_domain(),
    //     alpha_inner
    // );

    let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

    // Form charge dict, matching charges with their associated global indices
    let charge_dict = build_charge_dict(global_idxs, charges);

    let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

    datatree.run(false);

    // Test that direct computation is close to the FMM.
    let mut test_idx_vec = Vec::new();
    for (idx, index_pointer) in datatree.charge_index_pointer.iter().enumerate() {
        if index_pointer.1 - index_pointer.0 > 0 {
            test_idx_vec.push(idx);
        }
    }
    let leaf = &datatree.fmm.tree().get_all_leaves().unwrap()[test_idx_vec[0]];
    let leaf_check_surface = leaf.compute_surface(&datatree.fmm.tree.domain, order, alpha_inner);
    let leaf_equivalent_surface = leaf.compute_surface(&datatree.fmm.tree.domain, order, alpha_outer);

    let v_list = datatree.fmm.get_v_list(leaf).unwrap();
    let ncoeffs = datatree.fmm.m2l.ncoeffs(order);

    let mut direct = vec![0f32; ncoeffs];

    for source in v_list.iter() {
        let idx = datatree.level_index_pointer[leaf.level() as usize].get(source).unwrap();
        let source_multipole_ptr = datatree.level_multipoles[source.level() as usize][*idx];
        let source_multipole = unsafe {
            std::slice::from_raw_parts_mut(source_multipole_ptr.raw, ncoeffs)
        };
        let scaled_multipole = source_multipole
            .iter()
            .map(|d| *d*datatree.fmm.kernel.scale(leaf.level()) ).collect_vec();

        let source_equivalent_surface = source.compute_surface(&datatree.fmm.tree.domain, order, alpha_inner);

        // Evaluate check potential directly
        datatree.fmm.kernel.evaluate_st(EvalType::Value, &source_equivalent_surface, &leaf_check_surface, &scaled_multipole, &mut direct)
    }

    let check_potential = rlst_array_from_slice2!(f32, direct.as_slice(), [ncoeffs, 1]);
    // Compute local expansion
    let local_direct = empty_array::<f32, 2>().simple_mult_into_resize(
        datatree.fmm.dc2e_inv_1.view(),
        empty_array::<f32, 2>().simple_mult_into_resize(
            datatree.fmm.dc2e_inv_2.view(),
            check_potential.view()
        )
    );

    let idx = datatree.level_index_pointer[leaf.level() as usize].get(leaf).unwrap();
    let leaf_local_ptr = datatree.level_locals[leaf.level() as usize][*idx];
    let leaf_local = unsafe {
        std::slice::from_raw_parts_mut(leaf_local_ptr.raw, ncoeffs)
    };


    let point = leaf.centre(datatree.fmm.tree().get_domain());
    let point = rlst_array_from_slice2!(f32, point.as_slice(), [3, 1]);
    let mut r1 = vec![0.];
    datatree.fmm.kernel.evaluate_st(EvalType::Value, &leaf_equivalent_surface, point.data(), local_direct.data(), &mut r1);

    let mut r2 = vec![0.];
    datatree.fmm.kernel.evaluate_st(EvalType::Value, &leaf_equivalent_surface, point.data(), leaf_local, &mut r2);

    let err = (r1[0]-r2[0]).abs()/r1[0];
    println!("{:?} {:?} {:?}", r1, r2, err);

    err
}


fn main() {
    let npoints = 10000;

    let points = points_fixture::<f32>(npoints, None, None);
    // let points = points_fixture::<f32>(npoints, None, None);

    let global_idxs = (0..npoints).collect_vec();
    let charges = vec![1.0; npoints];

    let order = 3;
    let test_val = 1e-6;

    let alpha_inner = 1.05;
    let alpha_outer = 2.95;
    let depth = 2;

    let k = 10000;
    let threshold = 0.99999999999999;
    // let tree = SingleNodeTree::new(points.data(), false, None, Some(depth), &global_idxs, true);
    // let kernel = Laplace3dKernel::default();
    // let m2l_data = SvdFieldTranslationKiFmmRcmp::new(
    //     kernel.clone(),
    //     Some(k),
    //     threshold,
    //     order,
    //     *tree.get_domain(),
    //     alpha_inner,
    // );

    // let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

    // // Form charge dict, matching charges with their associated global indices
    // let charge_dict = build_charge_dict(&global_idxs, &charges);

    // let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

    // let s = Instant::now();
    // let times = datatree.run(true);
    // println!("runtime {:?} operators {:?}", s.elapsed(), times);
    // min to get to 10^-3
    // test_uniform_f32_svd_rcmp(&points, &charges, &global_idxs, order, alpha_inner, alpha_outer, true, depth, 10000, 0.99999, 1e-3);
    // test_uniform_f32_svd_rcmp(&points, &charges, &global_idxs, order, alpha_inner, alpha_outer, false, depth, 10000, 0.99999999, 1e-6);

    // Double precision
    let err = test_uniform_f32_svd_rcmp(&points, &charges, &global_idxs, order, alpha_inner, alpha_outer, false, depth, k, threshold, 1e-6); // 3 digits minimum
    println!("err {:?}", err);
    assert!(err < test_val);
}
