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
use rlst_dense::{RandomAccessByRef, RandomAccessMut, RawAccess, RawAccessMut, Shape};

fn get_corners<'a>(grid: &impl Grid<'a>, index: &usize, corners: &mut Vec<Vec<f64>>) {}

fn get_corners_vec<'a>(grid: &impl Grid<'a>, indices: &[usize], corners: &mut Vec<Vec<f64>>) {}

fn get_element(connectivity: &[usize], index: &usize, element: &mut Vec<usize>) {}
fn get_element_vec(connectivity: &[usize], indices: &[usize], element: &mut Vec<Vec<usize>>) {}

fn get_local2global(local2global: &[usize], index: &usize, result: &mut Vec<usize>) {}
fn get_local2global_vec(local2global: &[usize], indices: &[usize], result: &mut Vec<Vec<usize>>) {}

fn get_jacobian(corners: &Vec<Vec<f64>>, jac: &mut [f64]) {}
fn get_jacobian_vec(corners: &Vec<Vec<f64>>, jac: &mut Vec<Vec<f64>>) {}

fn get_integration_element(jac: &Vec<f64>, int_elem: &f64) {}
fn get_integration_element_vec(jac: &Vec<Vec<f64>>, int_elem: &[f64]) {}

fn get_global_point(corners: &Vec<Vec<f64>>, point: &Vec<f64>, result: &mut [f64]) {}
fn get_global_point_vec(corners: &Vec<Vec<f64>>, point: &Vec<f64>, result: &mut [f64]) {}

fn lagrange_kernel<'a>(
    test_indices: &[usize],
    trial_indices: &[usize],
    test_grid: &impl Grid<'a>,
    trial_grid: &impl Grid<'a>,
    test_connectivity: &[usize],
    trial_connectivity: &[usize],
    test_local2global: &[usize],
    trial_local2global: &[usize],
    quad_points: &[f64],
    quad_weights: &[f64],
    global_result: &Mat<f64>,
) {
    let kernel = Laplace3dKernel::new();

    let NUMBER_OF_QUAD_POINTS = quad_weights.len();
    let VEC_LENGTH = 10;
    let TODO = 10;

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
    let mut temp_result = vec![vec![0.0]; TODO];
    let mut temp_factor = vec![0.0; TODO];
    let mut shape_integral = vec![vec![vec![0.0; TODO]]];

    get_corners(test_grid, &test_index, &mut test_corners);
    get_corners_vec(trial_grid, &trial_index, &mut trial_corners);

    get_element(test_connectivity, &test_index, &mut test_element);
    get_element_vec(trial_connectivity, &trial_index, &mut trial_element);

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

        temp_result[0][0] = 0.0;
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
        }
    }
    /*

      for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
           ++testQuadIndex) {
        testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex], quadPoints[2 * testQuadIndex + 1]);
        testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
        BASIS(TEST, evaluate)(&testPoint, &testValue[0]);

        for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
          tempResult[j] = M_ZERO;
        }

        for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
             ++trialQuadIndex) {
          trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex], quadPoints[2 * trialQuadIndex + 1]);
          getGlobalPointVec(trialCorners, &trialPoint, trialGlobalPoint);
          BASIS(TRIAL, evaluate)(&trialPoint, &trialValue[0]);
          KERNEL(VEC_STRING)
          (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters,
           &kernelValue);
          tempFactor = quadWeights[trialQuadIndex] * kernelValue;
          for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
            tempResult[j] += trialValue[j] * tempFactor;
        }

        for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
          for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
            shapeIntegral[i][j] +=
                tempResult[j] * quadWeights[testQuadIndex] * testValue[i];
          }
      }

      for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
        for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
          shapeIntegral[i][j] *=
              (testIntElem * myTestLocalMultipliers[i]) * trialIntElem;
        }

      for (int vecIndex = 0; vecIndex < VEC_LENGTH; ++vecIndex)
        if (!elementsAreAdjacent(testElement, trialElement[vecIndex],
                                 gridsAreDisjoint)) {
          for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
              globalRowIndex = myTestLocal2Global[i];
              globalColIndex = myTrialLocal2Global[vecIndex][j];
              globalResult[globalRowIndex * nTrial + globalColIndex] +=
                  ((REALTYPE*)(&shapeIntegral[i][j]))[vecIndex] *
                  myTrialLocalMultipliers[vecIndex][j];
            }
        }
    }
    */
}

pub fn assemble<'a>(
    output: &mut Mat<f64>,
    kernel: &impl Kernel<T = f64>,
    needs_trial_normal: bool,
    needs_test_normal: bool,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) {
    *output.get_mut(0, 0).unwrap() = 0.5;
    lagrange_kernel(
        &[],
        &[],
        test_space.grid(),
        trial_space.grid(),
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        output,
    );
}
