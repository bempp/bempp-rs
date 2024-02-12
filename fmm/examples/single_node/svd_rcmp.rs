//! Single Node FMM using an SVD based M2L translation operator
use std::time::Instant;

use itertools::Itertools;

use rlst_dense::{array::Array, base_array::BaseArray, data_container::VectorContainer, rlst_array_from_slice2, rlst_dynamic_array2, traits::{MultInto, MultIntoResize, RawAccess, Shape}};

use bempp_field::types::SvdFieldTranslationKiFmmRcmp;
use bempp_fmm::{
    charge::build_charge_dict,
    types::{FmmDataUniform, KiFmmLinear},
};
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::{field::FieldTranslationData, fmm::FmmLoop, tree::Tree, types::EvalType};
use bempp_tree::implementations::helpers::points_fixture;
use bempp_tree::types::single_node::SingleNodeTree;

use bempp_traits::kernel::Kernel;
use bempp_traits::fmm::Fmm;


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
let charge_dict = build_charge_dict(global_idxs, charges);
let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();
datatree.run(false);
let ncoeffs = datatree.fmm.m2l.ncoeffs(order);

// Test that direct computation is close to the FMM.
let mut test_idx_vec = Vec::new();
for (idx, index_pointer) in datatree.charge_index_pointer.iter().enumerate() {
  if index_pointer.1 - index_pointer.0 > 0 {
      test_idx_vec.push(idx);
  }
}
let leaf = &datatree.fmm.tree().get_all_leaves().unwrap()[test_idx_vec[0]];
let leaf_idx = datatree.fmm.tree().get_leaf_index(leaf).unwrap();
let (l, r) = datatree.charge_index_pointer[*leaf_idx];
let potentials = &datatree.potentials[l..r];

let idx = datatree.level_index_pointer[leaf.level() as usize].get(leaf).unwrap();
let leaf_local_ptr = datatree.level_locals[leaf.level() as usize][*idx];
let leaf_local = unsafe {
  std::slice::from_raw_parts_mut(leaf_local_ptr.raw, ncoeffs)
};

let coordinates = datatree.fmm.tree().get_all_coordinates().unwrap();
let (l, r) = datatree.charge_index_pointer[*leaf_idx];
let leaf_coordinates_row_major = &coordinates[l * 3..r * 3];

let dim = datatree.fmm.kernel.space_dimension();
let ntargets = leaf_coordinates_row_major.len() / dim;

let leaf_coordinates_row_major =
  rlst_array_from_slice2!(f64, leaf_coordinates_row_major, [ntargets, dim], [dim, 1]);
let mut leaf_coordinates_col_major = rlst_dynamic_array2!(f64, [ntargets, dim]);
leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

let mut direct = vec![0f64; ntargets];

let all_charges = charge_dict.into_values().collect_vec();

let kernel = Laplace3dKernel::default();

kernel.evaluate_st(
  EvalType::Value,
  points.data(),
  leaf_coordinates_col_major.data(),
  &all_charges,
  &mut direct,
);

let abs_error: f64 = potentials
  .iter()
  .zip(direct.iter())
  .map(|(a, b)| (a - b).abs())
  .sum();
let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

// println!("pot {:?}", &potentials[0..3]);
// println!("direct {:?}", &direct[0..3]);
rel_error
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

    let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);
    let charge_dict = build_charge_dict(global_idxs, charges);
    let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();
    datatree.run(false);
    let ncoeffs = datatree.fmm.m2l.ncoeffs(order);

    // Test that direct computation is close to the FMM.
    let mut test_idx_vec = Vec::new();
    for (idx, index_pointer) in datatree.charge_index_pointer.iter().enumerate() {
        if index_pointer.1 - index_pointer.0 > 0 {
            test_idx_vec.push(idx);
        }
    }
    let leaf = &datatree.fmm.tree().get_all_leaves().unwrap()[test_idx_vec[0]];
    let leaf_idx = datatree.fmm.tree().get_leaf_index(leaf).unwrap();
    let (l, r) = datatree.charge_index_pointer[*leaf_idx];
    let potentials = &datatree.potentials[l..r];

    let idx = datatree.level_index_pointer[leaf.level() as usize].get(leaf).unwrap();
    let leaf_local_ptr = datatree.level_locals[leaf.level() as usize][*idx];
    let leaf_local = unsafe {
        std::slice::from_raw_parts_mut(leaf_local_ptr.raw, ncoeffs)
    };

    let coordinates = datatree.fmm.tree().get_all_coordinates().unwrap();
    let (l, r) = datatree.charge_index_pointer[*leaf_idx];
    let leaf_coordinates_row_major = &coordinates[l * 3..r * 3];

    let dim = datatree.fmm.kernel.space_dimension();
    let ntargets = leaf_coordinates_row_major.len() / dim;

    let leaf_coordinates_row_major =
        rlst_array_from_slice2!(f32, leaf_coordinates_row_major, [ntargets, dim], [dim, 1]);
    let mut leaf_coordinates_col_major = rlst_dynamic_array2!(f32, [ntargets, dim]);
    leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

    let mut direct = vec![0f32; ntargets];

    let all_charges = charge_dict.into_values().collect_vec();

    let kernel = Laplace3dKernel::default();

    kernel.evaluate_st(
        EvalType::Value,
        points.data(),
        leaf_coordinates_col_major.data(),
        &all_charges,
        &mut direct,
    );

    let abs_error: f32 = potentials
        .iter()
        .zip(direct.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let rel_error: f32 = abs_error / (direct.iter().sum::<f32>());

    // println!("pot {:?}", &potentials[0..3]);
    // println!("direct {:?}", &direct[0..3]);
    rel_error
}


fn minimal_ranks_f32(digits: usize, timing: bool, depth: u64) {
    let npoints = 1000000;

    let points = points_fixture::<f32>(npoints, None, None);

    let global_idxs = (0..npoints).collect_vec();
    let charges = vec![1.0; npoints];

    let alpha_inner = 1.05;
    let alpha_outer = 2.95;

    // 6 digits
    let order;
    let k;
    let threshold;
    let test_val;

    if digits == 3 {
        order = 3;
        k = 10;
        threshold = 0.99967;
        test_val = 1e-3;
    } else if digits == 5 {
        order = 5;
        k = 47;
        threshold = 0.999999;
        test_val = 1.5e-5;
    } else {
        panic!("only 3 or 5 valid")
    }

    if timing {
        let tree = SingleNodeTree::new(points.data(), false, None, Some(depth), &global_idxs, true);
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
        let charge_dict = build_charge_dict(&global_idxs, &charges);

        let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

        let s = Instant::now();
        let times = datatree.run(true);
        println!("runtime {:?} operators {:?}", s.elapsed(), times);
    } else {
        let err = test_uniform_f32_svd_rcmp(&points, &charges, &global_idxs, order, alpha_inner, alpha_outer, true, depth, k, threshold, test_val); // 3 digits minimum
        // println!("err {:?}", err);
        assert!(err < test_val);
    }
}


fn minimal_ranks_f64(digits: usize, timing: bool, depth: u64) {
    let npoints = 1000000;

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

    if timing {
        let tree = SingleNodeTree::new(points.data(), false, None, Some(depth), &global_idxs, true);
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
        let charge_dict = build_charge_dict(&global_idxs, &charges);

        let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

        let s = Instant::now();
        let times = datatree.run(true);
        println!("runtime {:?} operators {:?}", s.elapsed(), times);
    } else {
        let err = test_uniform_f64_svd_rcmp(&points, &charges, &global_idxs, order, alpha_inner, alpha_outer, true, depth, k, threshold, test_val); // 3 digits minimum
        // println!("err {:?}", err);
        assert!(err < test_val);
    }
}


fn main() {

    let depth = 6;
    let order = 3;
    let time = true;

    minimal_ranks_f32(order, time, depth);
    // minimal_ranks_f64(order, time, depth);
}
