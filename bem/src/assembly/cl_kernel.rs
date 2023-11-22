use crate::function_space::SerialFunctionSpace;
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_quadrature::duffy::quadrilateral::quadrilateral_duffy;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::simplex_rules::simplex_rule;
use bempp_quadrature::types::{CellToCellConnectivity, TestTrialNumericalQuadratureDefinition};
use bempp_tools::arrays::{transpose_to_matrix, zero_matrix, Array4D, Mat};
use bempp_traits::arrays::{AdjacencyListAccess, Array4DAccess};
use bempp_traits::bem::{DofMap, FunctionSpace};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};
use bempp_traits::kernel::Kernel;
use bempp_traits::types::EvalType;
use bempp_traits::types::Scalar;
use rayon::prelude::*;
use rlst_common::traits::{RandomAccessByRef, RandomAccessMut, RawAccess, RawAccessMut, Shape};

fn get_corners<'a>(grid: &impl Grid<'a>, index: &usize, corners: &mut Vec<Vec<f64>>) {
    let v = grid.geometry().cell_vertices(*index).unwrap();
    for (i, j) in v.iter().enumerate() {
        for k in 0..3 {
            corners[k][i] = *grid.geometry().coordinate(*j, k).unwrap();
        }
    }
}

fn get_corners_vec<'a>(grid: &impl Grid<'a>, indices: &[usize], corners: &mut Vec<Vec<f64>>) {
    for (a, index) in indices.iter().enumerate() {
        let v = grid.geometry().cell_vertices(*index).unwrap();
        for (i, j) in v.iter().enumerate() {
            for k in 0..3 {
                corners[k][a * 3 + i] = *grid.geometry().coordinate(*j, k).unwrap();
            }
        }
    }
}

fn get_element<'a>(grid: &impl Grid<'a>, index: &usize, element: &mut [usize]) {
    let c = grid.topology().connectivity(2, 0).row(*index).unwrap();
    for (i, j) in element.iter_mut().zip(c) {
        *i = *j;
    }
}
fn get_element_vec<'a>(grid: &impl Grid<'a>, indices: &[usize], element: &mut Vec<Vec<usize>>) {
    for (a, index) in indices.iter().enumerate() {
        let c = grid.topology().connectivity(2, 0).row(*index).unwrap();
        for (i, j) in c.iter().enumerate() {
            element[a][i] = *j;
        }
    }
}

fn get_local2global(local2global: &[usize], index: &usize, result: &mut [usize]) {
    result[0] = local2global[*index];
}
fn get_local2global_vec(local2global: &[usize], indices: &[usize], result: &mut Vec<Vec<usize>>) {
    for (a, index) in indices.iter().enumerate() {
        result[a][0] = local2global[*index];
    }
}

fn get_jacobian(corners: &Vec<Vec<f64>>, jac: &mut [f64]) {
    jac[0] = corners[0][0];
}
fn get_jacobian_vec(corners: &Vec<Vec<f64>>, jac: &mut Vec<Vec<f64>>) {
    jac[0][0] = corners[0][0];
}

fn get_integration_element(jac: &[f64], int_elem: &mut f64) {
    *int_elem = jac[0];
}
fn get_integration_element_vec(jac: &Vec<Vec<f64>>, int_elem: &mut [f64]) {
    int_elem[0] = jac[0][0];
}

fn get_global_point(corners: &Vec<Vec<f64>>, point: &[f64], result: &mut [f64]) {
    for i in 0..3 {
        result[i] = corners[0][i] + point[0] * (corners[1][i] - corners[0][i]) + point[1] * (corners[2][i] - corners[0][i]);
    }
}
fn get_global_point_vec(corners: &Vec<Vec<f64>>, point: &[f64], result: &mut [f64]) {
    return;
    for index in 0.. corners[0].len() / 3 {
        for i in 0..3 {
            result[index * 3 + i] = corners[0][3 * index + i] + point[0] * (corners[1][3 * index + i] - corners[0][3 * index + i]) + point[1] * (corners[2][3 * index + i] - corners[2][3 * index + i]);
        }
    }
    
}

fn elements_are_adjacent<'a>(e0: &[usize], e1: &[usize]) -> bool {
    for i in e0 {
        for j in e1 {
            if i == j {
                return true;
            }
        }
    }
    false
}

fn lagrange_kernel<'a>(
    n_trial: usize,
    test_indices: &[usize],
    trial_indices: &[usize],
    test_grid: &impl Grid<'a>,
    trial_grid: &impl Grid<'a>,
    test_local2global: &[usize],
    trial_local2global: &[usize],
    quad_points: &[f64],
    quad_weights: &[f64],
    global_result: &mut [f64],
) {
    let kernel = Laplace3dKernel::new();

    let NUMBER_OF_QUAD_POINTS = quad_weights.len();
    let VEC_LENGTH = test_indices.len();
    let TODO = test_indices.len() * 3;

    let test_index = 0;
    let trial_index = trial_indices;

    let mut test_quad_index = 0;
    let mut trial_quad_index = 0;
    let mut i = 0;
    let mut j = 0;
    let mut global_row_index = 0;
    let mut global_col_index = 0;

    let mut test_global_point = vec![0.0; 3];
    let mut trial_global_point = vec![0.0; TODO * 3];
    let mut test_corners = vec![vec![0.0; 3]; 3];
    let mut trial_corners = vec![vec![0.0; TODO]; 3];
    let mut test_element = vec![0; 3];
    let mut trial_element = vec![vec![0; 3]; VEC_LENGTH];

    let mut my_test_local2global = vec![0; 1];
    let mut my_trial_local2global = vec![vec![0; 1]; VEC_LENGTH];

    let mut test_jac = vec![0.0; 2];
    let mut trial_jac = vec![vec![0.0; 3]; 2];

    let mut test_point = vec![0.0; 2];
    let mut trial_point = vec![0.0; 2];

    let mut test_int_elem = 0.0;
    let mut trial_int_elem = vec![0.0; TODO];

    let mut test_value = vec![0.0];
    let mut trial_value = vec![0.0];

    let mut kernel_value = vec![0.0; TODO];
    let mut temp_result = vec![0.0; TODO];
    let mut temp_factor = vec![0.0; TODO];
    let mut shape_integral = vec![vec![vec![0.0; TODO]]];

    get_corners(test_grid, &test_index, &mut test_corners);
    get_corners_vec(trial_grid, &trial_index, &mut trial_corners);

    get_element(test_grid, &test_index, &mut test_element);
    get_element_vec(trial_grid, &trial_index, &mut trial_element);

    get_local2global(test_local2global, &test_index, &mut my_test_local2global);
    get_local2global_vec(trial_local2global, &trial_index, &mut my_trial_local2global);

    get_jacobian(&test_corners, &mut test_jac);
    get_jacobian_vec(&trial_corners, &mut trial_jac);

    get_integration_element(&test_jac, &mut test_int_elem);
    get_integration_element_vec(&trial_jac, &mut trial_int_elem);

    for test_quad_index in 0..NUMBER_OF_QUAD_POINTS {
        test_point[0] = quad_points[2 * test_quad_index];
        test_point[1] = quad_points[2 * test_quad_index + 1];
        get_global_point(&test_corners, &test_point, &mut test_global_point);
        test_value[0] = 1.0;

        for i in temp_result.iter_mut() {
            *i = 0.0;
        }
        for trial_quad_index in 0..NUMBER_OF_QUAD_POINTS {
            trial_point[0] = quad_points[2 * trial_quad_index];
            trial_point[1] = quad_points[2 * trial_quad_index + 1];
            get_global_point_vec(&trial_corners, &trial_point, &mut trial_global_point);
            trial_value[0] = 1.0;
            kernel.assemble_st(
                EvalType::Value,
                &test_global_point,
                &trial_global_point,
                &mut kernel_value,
            );
            for (i, j) in temp_factor.iter_mut().zip(&kernel_value) {
                *i = quad_weights[trial_quad_index] * j;
            }
            for ((i, j), k) in temp_result.iter_mut().zip(&trial_value).zip(&temp_factor) {
                *i += j * k
            }
        }

        for vec_index in 0..VEC_LENGTH {
            shape_integral[0][0][vec_index] *= test_int_elem * trial_int_elem[vec_index];
        }

        for vec_index in 0..VEC_LENGTH {
            if !elements_are_adjacent(&test_element, &trial_element[vec_index]) {
                global_row_index = my_test_local2global[0];
                global_col_index = my_trial_local2global[vec_index][0];
                global_result[global_row_index * n_trial + global_col_index] += shape_integral[0][0][vec_index];
            }
        }
    }
}

pub fn assemble<'a>(
    output: &mut Mat<f64>,
    kernel: &impl Kernel<T = f64>,
    needs_trial_normal: bool,
    needs_test_normal: bool,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) {
    const NPTS: usize = 16;

    let qrule_test = simplex_rule(ReferenceCellType::Triangle, NPTS).unwrap();
    let qpoints = qrule_test.points;
    let qweights = qrule_test.weights;
    let test_colouring = test_space.compute_cell_colouring();
    let trial_colouring = trial_space.compute_cell_colouring();


    let mut test_local2global = vec![0];
    for trial_c in &trial_colouring {
        let mut trial_local2global = vec![];
        for c in trial_c {
            for d in trial_space.dofmap().cell_dofs(*c).unwrap() {
                trial_local2global.push(*d);
            }
        }
        for test_c in &test_colouring {
            for c in test_c {
                for d in test_space.dofmap().cell_dofs(*c).unwrap() {
                    test_local2global[0] = *d;
                }

                lagrange_kernel(
                    trial_space.dofmap().global_size(),
                    test_c,
                    trial_c,
                    test_space.grid(),
                    trial_space.grid(),
                    &test_local2global,
                    &trial_local2global,
                    &qpoints,
                    &qweights,
                    output.data_mut(),
                );
            }
        }
    }
}
