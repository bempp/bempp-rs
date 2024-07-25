//! Batched dense assembly of boundary operators
use crate::assembly::common::{equal_grids, RawData2D, SparseMatrixData};
use crate::quadrature::duffy::{
    quadrilateral_duffy, quadrilateral_triangle_duffy, triangle_duffy, triangle_quadrilateral_duffy,
};
use crate::quadrature::simplex_rules::simplex_rule;
use crate::quadrature::types::{CellToCellConnectivity, TestTrialNumericalQuadratureDefinition};
use crate::traits::function::FunctionSpace;
#[cfg(feature = "mpi")]
use crate::traits::function::FunctionSpaceInParallel;
use ndgrid::traits::{GeometryMap, Entity, Grid, Topology};
use ndgrid::types::Ownership;
use ndelement::reference_cell;
use ndelement::traits::FiniteElement;
use ndelement::types::ReferenceCellType;
use rayon::prelude::*;
use rlst::{
    rlst_dynamic_array2, rlst_dynamic_array3, rlst_dynamic_array4, CsrMatrix, RandomAccessMut,
    RawAccess, RawAccessMut, RlstScalar, Shape, UnsafeRandomAccessByRef, MatrixInverse
};
use std::collections::HashMap;

use super::RlstArray;

fn neighbours<TestGrid: Grid, TrialGrid: Grid>(
    test_grid: &TestGrid,
    trial_grid: &TrialGrid,
    test_cell: usize,
    trial_cell: usize,
) -> bool {
    if !equal_grids(test_grid, trial_grid) {
        false
    } else {
        let test_vertices = trial_grid
            .entity(2, test_cell).unwrap()
            .topology()
            .sub_entity_iter(0)
            .collect::<Vec<_>>();
        for v in trial_grid
            .entity(2, trial_cell).unwrap()
            .topology()
            .sub_entity_iter(0)
        {
            if test_vertices.contains(&v) {
                return true;
            }
        }
        false
    }
}

fn get_singular_quadrature_rule(
    test_celltype: ReferenceCellType,
    trial_celltype: ReferenceCellType,
    pairs: &[(usize, usize)],
    npoints: usize,
) -> TestTrialNumericalQuadratureDefinition {
    if pairs.is_empty() {
        panic!("Non-singular rule requested.");
    }
    let con = CellToCellConnectivity {
        connectivity_dimension: match pairs.len() {
            1 => 0,
            2 => 1,
            _ => 2,
        },
        local_indices: pairs.to_vec(),
    };
    match test_celltype {
        ReferenceCellType::Triangle => match trial_celltype {
            ReferenceCellType::Triangle => triangle_duffy(&con, npoints).unwrap(),
            ReferenceCellType::Quadrilateral => {
                triangle_quadrilateral_duffy(&con, npoints).unwrap()
            }
            _ => {
                unimplemented!("Only triangles and quadrilaterals are currently supported");
            }
        },
        ReferenceCellType::Quadrilateral => match trial_celltype {
            ReferenceCellType::Triangle => quadrilateral_triangle_duffy(&con, npoints).unwrap(),
            ReferenceCellType::Quadrilateral => quadrilateral_duffy(&con, npoints).unwrap(),
            _ => {
                unimplemented!("Only triangles and quadrilaterals are currently supported");
            }
        },
        _ => {
            unimplemented!("Only triangles and quadrilaterals are currently supported");
        }
    }
}

/// Assemble the contribution to the terms of a matrix for a batch of pairs of adjacent cells
#[allow(clippy::too_many_arguments)]
fn assemble_batch_singular<
    T: RlstScalar + MatrixInverse,
    TestGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    TrialGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    Element: FiniteElement<T = T> + Sync,
>(
    assembler: &impl BatchedAssembler<T = T>,
    deriv_size: usize,
    shape: [usize; 2],
    trial_cell_type: ReferenceCellType,
    test_cell_type: ReferenceCellType,
    trial_space: &impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element>,
    test_space: &impl FunctionSpace<Grid = TestGrid, FiniteElement = Element>,
    cell_pairs: &[(usize, usize)],
    trial_points: &RlstArray<T::Real, 2>,
    test_points: &RlstArray<T::Real, 2>,
    weights: &[T::Real],
    trial_table: &RlstArray<T, 4>,
    test_table: &RlstArray<T, 4>,
) -> SparseMatrixData<T> {
    let mut output = SparseMatrixData::<T>::new_known_size(
        shape,
        cell_pairs.len()
            * trial_space.element(trial_cell_type).dim()
            * test_space.element(test_cell_type).dim(),
    );
    let npts = weights.len();
    debug_assert!(weights.len() == npts);
    debug_assert!(test_points.shape()[0] == npts);
    debug_assert!(trial_points.shape()[0] == npts);

    let grid = test_space.grid();
    assert_eq!(grid.geometry_dim(), 3);
    assert_eq!(grid.topology_dim(), 2);

    // Memory assignment to be moved elsewhere as passed into here mutable?
    let mut k = rlst_dynamic_array2!(T, [deriv_size, npts]);
    let zero = num::cast::<f64, T::Real>(0.0).unwrap();
    let mut test_jdet = vec![zero; npts];
    let mut test_mapped_pts = rlst_dynamic_array2!(T::Real, [npts, 3]);
    let mut test_jacobians = rlst_dynamic_array2!(T::Real, [npts, 6]);
    let mut test_normals = rlst_dynamic_array2!(T::Real, [npts, 3]);

    let mut trial_jdet = vec![zero; npts];
    let mut trial_mapped_pts = rlst_dynamic_array2!(T::Real, [npts, 3]);
    let mut trial_jacobians = rlst_dynamic_array2!(T::Real, [npts, 6]);
    let mut trial_normals = rlst_dynamic_array2!(T::Real, [npts, 3]);

    let test_evaluator = grid.geometry_map(test_cell_type, test_points.data());
    let trial_evaluator = grid.geometry_map(trial_cell_type, trial_points.data());

    for (test_cell, trial_cell) in cell_pairs {
        test_evaluator.points(*test_cell, test_mapped_pts.data_mut());
        test_evaluator.jacobians_dets_normals(*test_cell, test_jacobians.data_mut(), &mut test_jdet, test_normals.data_mut());

        trial_evaluator.points(*trial_cell, trial_mapped_pts.data_mut());
        trial_evaluator.jacobians_dets_normals(*trial_cell, trial_jacobians.data_mut(), &mut trial_jdet, trial_normals.data_mut());

        assembler.kernel_assemble_pairwise_st(
            test_mapped_pts.data(),
            trial_mapped_pts.data(),
            k.data_mut(),
        );

        let test_dofs = test_space.cell_dofs(*test_cell).unwrap();
        let trial_dofs = trial_space.cell_dofs(*trial_cell).unwrap();
        for (test_i, test_dof) in test_dofs.iter().enumerate() {
            for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                let mut sum = num::cast::<f64, T>(0.0).unwrap();

                for (index, wt) in weights.iter().enumerate() {
                    unsafe {
                        sum += assembler.singular_kernel_value(
                            &k,
                            &test_normals,
                            &trial_normals,
                            index,
                        ) * assembler.test_trial_product(
                            test_table,
                            trial_table,
                            &test_jacobians,
                            &trial_jacobians,
                            &test_jdet,
                            &trial_jdet,
                            index,
                            index,
                            test_i,
                            trial_i,
                        ) * num::cast::<T::Real, T>(
                            *wt * *test_jdet.get_unchecked(index)
                                * *trial_jdet.get_unchecked(index),
                        )
                        .unwrap();
                    }
                }
                output.rows.push(test_space.global_dof_index(*test_dof));
                output.cols.push(trial_space.global_dof_index(*trial_dof));
                output.data.push(sum);
            }
        }
    }
    output
}

/// Assemble the contribution to the terms of a matrix for a batch of non-adjacent cells
#[allow(clippy::too_many_arguments)]
fn assemble_batch_nonadjacent<
    T: RlstScalar + MatrixInverse,
    TestGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    TrialGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    Element: FiniteElement<T = T> + Sync,
>(
    assembler: &impl BatchedAssembler<T = T>,
    deriv_size: usize,
    output: &RawData2D<T>,
    trial_cell_type: ReferenceCellType,
    test_cell_type: ReferenceCellType,
    trial_space: &impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element>,
    trial_cells: &[usize],
    test_space: &impl FunctionSpace<Grid = TestGrid, FiniteElement = Element>,
    test_cells: &[usize],
    trial_points: &RlstArray<T::Real, 2>,
    trial_weights: &[T::Real],
    test_points: &RlstArray<T::Real, 2>,
    test_weights: &[T::Real],
    trial_table: &RlstArray<T, 4>,
    test_table: &RlstArray<T, 4>,
) -> usize {
    let npts_test = test_weights.len();
    let npts_trial = trial_weights.len();
    debug_assert!(test_points.shape()[0] == npts_test);
    debug_assert!(trial_points.shape()[0] == npts_trial);

    let test_grid = test_space.grid();
    let trial_grid = trial_space.grid();

    assert_eq!(test_grid.geometry_dim(), 3);
    assert_eq!(test_grid.topology_dim(), 2);
    assert_eq!(trial_grid.geometry_dim(), 3);
    assert_eq!(trial_grid.topology_dim(), 2);

    let mut k = rlst_dynamic_array3!(T, [npts_test, deriv_size, npts_trial]);
    let zero = num::cast::<f64, T::Real>(0.0).unwrap();
    let mut test_jdet = vec![zero; npts_test];
    let mut test_mapped_pts = rlst_dynamic_array2!(T::Real, [npts_test, 3]);
    let mut test_normals = rlst_dynamic_array2!(T::Real, [npts_test, 3]);
    let mut test_jacobians = rlst_dynamic_array2!(T::Real, [npts_test, 6]);

    let test_evaluator = test_grid.geometry_map(test_cell_type, test_points.data());
    let trial_evaluator = trial_grid.geometry_map(trial_cell_type, trial_points.data());

    let mut trial_jdet = vec![vec![zero; npts_trial]; trial_cells.len()];
    let mut trial_mapped_pts = vec![];
    let mut trial_normals = vec![];
    let mut trial_jacobians = vec![];
    for _i in 0..trial_cells.len() {
        trial_mapped_pts.push(rlst_dynamic_array2!(T::Real, [npts_trial, 3]));
        trial_normals.push(rlst_dynamic_array2!(T::Real, [npts_trial, 3]));
        trial_jacobians.push(rlst_dynamic_array2!(T::Real, [npts_trial, 6]));
    }

    for (trial_cell_i, trial_cell) in trial_cells.iter().enumerate() {
        trial_evaluator.points(*trial_cell, trial_mapped_pts[trial_cell_i].data_mut());
        trial_evaluator.jacobians_dets_normals(*trial_cell, trial_jacobians[trial_cell_i].data_mut(), &mut trial_jdet[trial_cell_i], trial_normals[trial_cell_i].data_mut());
    }

    let mut sum: T;
    let mut trial_integrands = vec![num::cast::<f64, T>(0.0).unwrap(); npts_trial];

    for test_cell in test_cells {
        test_evaluator.points(*test_cell, test_mapped_pts.data_mut());
        test_evaluator.jacobians_dets_normals(*test_cell, test_jacobians.data_mut(), &mut test_jdet, test_normals.data_mut());

        for (trial_cell_i, trial_cell) in trial_cells.iter().enumerate() {
            if neighbours(test_grid, trial_grid, *test_cell, *trial_cell) {
                continue;
            }

            assembler.kernel_assemble_st(
                test_mapped_pts.data(),
                trial_mapped_pts[trial_cell_i].data(),
                k.data_mut(),
            );

            let test_dofs = test_space.cell_dofs(*test_cell).unwrap();
            let trial_dofs = trial_space.cell_dofs(*trial_cell).unwrap();

            for (test_i, test_dof) in test_dofs.iter().enumerate() {
                for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                    for (trial_index, trial_wt) in trial_weights.iter().enumerate() {
                        trial_integrands[trial_index] = num::cast::<T::Real, T>(
                            *trial_wt * trial_jdet[trial_cell_i][trial_index],
                        )
                        .unwrap();
                    }
                    sum = num::cast::<f64, T>(0.0).unwrap();
                    for (test_index, test_wt) in test_weights.iter().enumerate() {
                        let test_integrand =
                            num::cast::<T::Real, T>(*test_wt * test_jdet[test_index]).unwrap();
                        for trial_index in 0..npts_trial {
                            sum += unsafe {
                                assembler.nonsingular_kernel_value(
                                    &k,
                                    &test_normals,
                                    &trial_normals[trial_cell_i],
                                    test_index,
                                    trial_index,
                                ) * test_integrand
                                    * *trial_integrands.get_unchecked(trial_index)
                                    * assembler.test_trial_product(
                                        test_table,
                                        trial_table,
                                        &test_jacobians,
                                        &trial_jacobians[trial_cell_i],
                                        &test_jdet,
                                        &trial_jdet[trial_cell_i],
                                        test_index,
                                        trial_index,
                                        test_i,
                                        trial_i,
                                    )
                            };
                        }
                    }
                    unsafe {
                        *output.data.add(*test_dof + output.shape[0] * *trial_dof) += sum;
                    }
                }
            }
        }
    }
    1
}

/// Assemble the contribution to the terms of a matrix for a batch of pairs of adjacent cells if an (incorrect) non-singular quadrature rule was used
#[allow(clippy::too_many_arguments)]
fn assemble_batch_singular_correction<
    T: RlstScalar + MatrixInverse,
    TestGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    TrialGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    Element: FiniteElement<T = T> + Sync,
>(
    assembler: &impl BatchedAssembler<T = T>,
    deriv_size: usize,
    shape: [usize; 2],
    trial_cell_type: ReferenceCellType,
    test_cell_type: ReferenceCellType,
    trial_space: &impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element>,
    test_space: &impl FunctionSpace<Grid = TestGrid, FiniteElement = Element>,
    cell_pairs: &[(usize, usize)],
    trial_points: &RlstArray<T::Real, 2>,
    trial_weights: &[T::Real],
    test_points: &RlstArray<T::Real, 2>,
    test_weights: &[T::Real],
    trial_table: &RlstArray<T, 4>,
    test_table: &RlstArray<T, 4>,
) -> SparseMatrixData<T> {
    let mut output = SparseMatrixData::<T>::new_known_size(
        shape,
        cell_pairs.len()
            * trial_space.element(trial_cell_type).dim()
            * test_space.element(test_cell_type).dim(),
    );
    let npts_test = test_weights.len();
    let npts_trial = trial_weights.len();
    debug_assert!(test_points.shape()[0] == npts_test);
    debug_assert!(trial_points.shape()[0] == npts_trial);

    let grid = test_space.grid();
    assert_eq!(grid.geometry_dim(), 3);
    assert_eq!(grid.topology_dim(), 2);

    let mut k = rlst_dynamic_array3!(T, [npts_test, deriv_size, npts_trial]);

    let zero = num::cast::<f64, T::Real>(0.0).unwrap();

    let mut test_jdet = vec![zero; npts_test];
    let mut test_mapped_pts = rlst_dynamic_array2!(T::Real, [npts_test, 3]);
    let mut test_normals = rlst_dynamic_array2!(T::Real, [npts_test, 3]);
    let mut test_jacobians = rlst_dynamic_array2!(T::Real, [npts_test, 6]);

    let mut trial_jdet = vec![zero; npts_trial];
    let mut trial_mapped_pts = rlst_dynamic_array2!(T::Real, [npts_trial, 3]);
    let mut trial_normals = rlst_dynamic_array2!(T::Real, [npts_trial, 3]);
    let mut trial_jacobians = rlst_dynamic_array2!(T::Real, [npts_trial, 6]);

    let test_evaluator = grid.geometry_map(test_cell_type, test_points.data());
    let trial_evaluator = grid.geometry_map(trial_cell_type, trial_points.data());

    let mut sum: T;
    let mut trial_integrands = vec![num::cast::<f64, T>(0.0).unwrap(); npts_trial];

    for (test_cell, trial_cell) in cell_pairs {
        test_evaluator.points(*test_cell, test_mapped_pts.data_mut());
        test_evaluator.jacobians_dets_normals(*test_cell, test_jacobians.data_mut(), &mut test_jdet, test_normals.data_mut());

        trial_evaluator.points(*trial_cell, trial_mapped_pts.data_mut());
        trial_evaluator.jacobians_dets_normals(*trial_cell, trial_jacobians.data_mut(), &mut trial_jdet, trial_normals.data_mut());

        assembler.kernel_assemble_st(
            test_mapped_pts.data(),
            trial_mapped_pts.data(),
            k.data_mut(),
        );

        let test_dofs = test_space.cell_dofs(*test_cell).unwrap();
        let trial_dofs = trial_space.cell_dofs(*trial_cell).unwrap();
        for (test_i, test_dof) in test_dofs.iter().enumerate() {
            for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                for (trial_index, trial_wt) in trial_weights.iter().enumerate() {
                    trial_integrands[trial_index] =
                        num::cast::<T::Real, T>(*trial_wt * trial_jdet[trial_index]).unwrap();
                }
                sum = num::cast::<f64, T>(0.0).unwrap();
                for (test_index, test_wt) in test_weights.iter().enumerate() {
                    let test_integrand =
                        num::cast::<T::Real, T>(*test_wt * test_jdet[test_index]).unwrap();
                    for trial_index in 0..npts_trial {
                        sum += unsafe {
                            assembler.nonsingular_kernel_value(
                                &k,
                                &test_normals,
                                &trial_normals,
                                test_index,
                                trial_index,
                            ) * test_integrand
                                * *trial_integrands.get_unchecked(trial_index)
                                * assembler.test_trial_product(
                                    test_table,
                                    trial_table,
                                    &test_jacobians,
                                    &trial_jacobians,
                                    &test_jdet,
                                    &trial_jdet,
                                    test_index,
                                    trial_index,
                                    test_i,
                                    trial_i,
                                )
                        };
                    }
                }
                output.rows.push(test_space.global_dof_index(*test_dof));
                output.cols.push(trial_space.global_dof_index(*trial_dof));
                output.data.push(sum);
            }
        }
    }
    output
}

fn get_pairs_if_smallest(
    test_cell: &impl Entity,
    trial_cell: &impl Entity,
    vertex: usize,
) -> Option<Vec<(usize, usize)>> {
    let mut pairs = vec![];
    for (trial_i, trial_v) in trial_cell.topology().sub_entity_iter(0).enumerate() {
        for (test_i, test_v) in test_cell.topology().sub_entity_iter(0).enumerate() {
            if test_v == trial_v {
                if test_v < vertex {
                    return None;
                }
                pairs.push((test_i, trial_i));
            }
        }
    }
    Some(pairs)
}

/// Options for a batched assembler
pub struct BatchedAssemblerOptions {
    /// Number of points used in quadrature for non-singular integrals
    quadrature_degrees: HashMap<ReferenceCellType, usize>,
    /// Quadrature degrees to be used for singular integrals
    singular_quadrature_degrees: HashMap<(ReferenceCellType, ReferenceCellType), usize>,
    /// Maximum size of each batch of cells to send to an assembly function
    batch_size: usize,
}

impl Default for BatchedAssemblerOptions {
    fn default() -> Self {
        use ReferenceCellType::{Quadrilateral, Triangle};
        Self {
            quadrature_degrees: HashMap::from([(Triangle, 37), (Quadrilateral, 37)]),
            singular_quadrature_degrees: HashMap::from([
                ((Triangle, Triangle), 4),
                ((Quadrilateral, Quadrilateral), 4),
                ((Quadrilateral, Triangle), 4),
                ((Triangle, Quadrilateral), 4),
            ]),
            batch_size: 128,
        }
    }
}

pub trait BatchedAssembler: Sync + Sized {
    //! Batched assembler
    //!
    //! Assemble operators by processing batches of cells in parallel

    /// Scalar type
    type T: RlstScalar + MatrixInverse;
    /// Number of derivatives
    const DERIV_SIZE: usize;
    /// Number of derivatives needed in basis function tables
    const TABLE_DERIVS: usize;

    /// Get assembler options
    fn options(&self) -> &BatchedAssemblerOptions;

    /// Get mutable assembler options
    fn options_mut(&mut self) -> &mut BatchedAssemblerOptions;

    /// Set (non-singular) quadrature degree for a cell type
    fn quadrature_degree(&mut self, cell: ReferenceCellType, degree: usize) {
        *self
            .options_mut()
            .quadrature_degrees
            .get_mut(&cell)
            .unwrap() = degree;
    }

    /// Set singular quadrature degree for a pair of cell types
    fn singular_quadrature_degree(
        &mut self,
        cells: (ReferenceCellType, ReferenceCellType),
        degree: usize,
    ) {
        *self
            .options_mut()
            .singular_quadrature_degrees
            .get_mut(&cells)
            .unwrap() = degree;
    }

    /// Set the maximum size of a batch of cells to send to an assembly function
    fn batch_size(&mut self, size: usize) {
        self.options_mut().batch_size = size;
    }

    /// Return the kernel value to use in the integrand when using a singular quadrature rule
    ///
    /// # Safety
    /// This method is unsafe to allow `get_unchecked` to be used
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 2>,
        test_normals: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        trial_normals: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        index: usize,
    ) -> Self::T;

    /// Return the kernel value to use in the integrand when using a non-singular quadrature rule
    ///
    /// # Safety
    /// This method is unsafe to allow `get_unchecked` to be used
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 3>,
        test_normals: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        trial_normals: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> Self::T;

    /// Evaluate the kernel values for all source and target pairs
    ///
    /// For each source, the kernel is evaluated for exactly one target. This is equivalent to taking the diagonal of the matrix assembled by `kernel_assemble_st`
    fn kernel_assemble_pairwise_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    );

    /// Evaluate the kernel values for all sources and all targets
    ///
    /// For every source, the kernel is evaluated for every target.
    fn kernel_assemble_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    );

    /// The product of a test and trial function
    ///
    /// # Safety
    /// This method is unsafe to allow `get_unchecked` to be used
    #[allow(clippy::too_many_arguments)]
    unsafe fn test_trial_product(
        &self,
        test_table: &RlstArray<Self::T, 4>,
        trial_table: &RlstArray<Self::T, 4>,
        _test_jacobians: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        _trial_jacobians: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        _test_jdets: &[<Self::T as RlstScalar>::Real],
        _trial_jdets: &[<Self::T as RlstScalar>::Real],
        test_point_index: usize,
        trial_point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
    ) -> Self::T {
        *test_table.get_unchecked([0, test_point_index, test_basis_index, 0])
            * *trial_table.get_unchecked([0, trial_point_index, trial_basis_index, 0])
    }

    /// Assemble the singular contributions
    fn assemble_singular<
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        shape: [usize; 2],
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) -> SparseMatrixData<Self::T> {
        if !equal_grids(test_space.grid(), trial_space.grid()) {
            // If the test and trial grids are different, there are no neighbouring triangles
            return SparseMatrixData::new(shape);
        }

        if !trial_space.is_serial() || !test_space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if shape[0] != test_space.global_size() || shape[1] != trial_space.global_size() {
            panic!("Matrix has wrong shape");
        }

        let grid = test_space.grid();

        let mut qweights = vec![];
        let mut trial_points = vec![];
        let mut test_points = vec![];
        let mut trial_tables = vec![];
        let mut test_tables = vec![];
        let mut test_cell_types = vec![];
        let mut trial_cell_types = vec![];

        let mut cell_blocks = vec![];

        let mut pair_indices = HashMap::new();

        for test_cell_type in grid.entity_types(2) {
            for trial_cell_type in grid.entity_types(2) {
                let qdegree = self.options().singular_quadrature_degrees
                    [&(*test_cell_type, *trial_cell_type)];
                let offset = qweights.len();

                let mut possible_pairs = vec![];
                // Vertex-adjacent
                for i in 0..reference_cell::entity_counts(*test_cell_type)[0] {
                    for j in 0..reference_cell::entity_counts(*trial_cell_type)[0] {
                        possible_pairs.push(vec![(i, j)]);
                    }
                }
                // edge-adjacent
                for test_e in reference_cell::edges(*test_cell_type) {
                    for trial_e in reference_cell::edges(*trial_cell_type) {
                        possible_pairs.push(vec![(test_e[0], trial_e[0]), (test_e[1], trial_e[1])]);
                        possible_pairs.push(vec![(test_e[1], trial_e[0]), (test_e[0], trial_e[1])]);
                    }
                }
                // Same cell
                if test_cell_type == trial_cell_type {
                    possible_pairs.push(
                        (0..reference_cell::entity_counts(*test_cell_type)[0])
                            .map(&|i| (i, i))
                            .collect::<Vec<_>>(),
                    );
                }

                for (i, pairs) in possible_pairs.iter().enumerate() {
                    pair_indices.insert(
                        (*test_cell_type, *trial_cell_type, pairs.clone()),
                        offset + i,
                    );
                    test_cell_types.push(*test_cell_type);
                    trial_cell_types.push(*trial_cell_type);
                }

                for pairs in &possible_pairs {
                    let qrule = get_singular_quadrature_rule(
                        *test_cell_type,
                        *trial_cell_type,
                        pairs,
                        qdegree,
                    );
                    let npts = qrule.weights.len();

                    let mut points = rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts, 2]);
                    for i in 0..npts {
                        for j in 0..2 {
                            *points.get_mut([i, j]).unwrap() =
                                num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                    qrule.trial_points[2 * i + j],
                                )
                                .unwrap();
                        }
                    }
                    let trial_element = trial_space.element(*trial_cell_type);
                    let mut table = rlst_dynamic_array4!(
                        Self::T,
                        trial_element.tabulate_array_shape(Self::TABLE_DERIVS, points.shape()[0])
                    );
                    trial_element.tabulate(&points, Self::TABLE_DERIVS, &mut table);
                    trial_points.push(points);
                    trial_tables.push(table);

                    let mut points = rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts, 2]);
                    for i in 0..npts {
                        for j in 0..2 {
                            *points.get_mut([i, j]).unwrap() =
                                num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                    qrule.test_points[2 * i + j],
                                )
                                .unwrap();
                        }
                    }
                    let test_element = test_space.element(*test_cell_type);
                    let mut table = rlst_dynamic_array4!(
                        Self::T,
                        test_element.tabulate_array_shape(Self::TABLE_DERIVS, points.shape()[0])
                    );
                    test_element.tabulate(&points, Self::TABLE_DERIVS, &mut table);
                    test_points.push(points);
                    test_tables.push(table);
                    qweights.push(
                        qrule
                            .weights
                            .iter()
                            .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                            .collect::<Vec<_>>(),
                    );
                }
            }
        }
        let mut cell_pairs: Vec<Vec<(usize, usize)>> = vec![vec![]; pair_indices.len()];
        for vertex in grid.entity_iter(0) {
            for test_cell_index in vertex.topology().connected_entity_iter(2) {
                let test_cell = grid.entity(2, test_cell_index).unwrap();
                if test_cell.ownership() == Ownership::Owned {
                    let test_cell_type = test_cell.entity_type();
                    for trial_cell_index in vertex.topology().connected_entity_iter(2) {
                        let trial_cell = grid.entity(2, trial_cell_index).unwrap();
                        let trial_cell_type = trial_cell.entity_type();
                        if let Some(pairs) = get_pairs_if_smallest(&test_cell, &trial_cell, vertex.local_index()) {
                            cell_pairs[pair_indices[&(test_cell_type, trial_cell_type, pairs)]]
                                .push((test_cell_index, trial_cell_index));
                        }
                    }
                }
            }
        }


        let batch_size = self.options().batch_size;
        for (i, cells) in cell_pairs.iter().enumerate() {
            let mut start = 0;
            while start < cells.len() {
                let end = std::cmp::min(start + batch_size, cells.len());
                cell_blocks.push((i, &cells[start..end]));
                start = end;
            }
        }
        cell_blocks
            .into_par_iter()
            .map(|(i, cell_block)| {
                assemble_batch_singular::<Self::T, TestGrid, TrialGrid, Element>(
                    self,
                    Self::DERIV_SIZE,
                    shape,
                    trial_cell_types[i],
                    test_cell_types[i],
                    trial_space,
                    test_space,
                    cell_block,
                    &trial_points[i],
                    &test_points[i],
                    &qweights[i],
                    &trial_tables[i],
                    &test_tables[i],
                )
            })
            .reduce(
                || SparseMatrixData::<Self::T>::new(shape),
                |mut a, b| {
                    a.add(b);
                    a
                },
            )
    }

    /// Assemble the singular correction
    ///
    /// The singular correction is the contribution is the terms for adjacent cells are assembled using an (incorrect) non-singular quadrature rule
    fn assemble_singular_correction<
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        shape: [usize; 2],
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) -> SparseMatrixData<Self::T> {
        if !equal_grids(test_space.grid(), trial_space.grid()) {
            // If the test and trial grids are different, there are no neighbouring triangles
            return SparseMatrixData::new(shape);
        }

        if !trial_space.is_serial() || !test_space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if shape[0] != test_space.global_size() || shape[1] != trial_space.global_size() {
            panic!("Matrix has wrong shape");
        }

        let grid = test_space.grid();

        let mut qweights_test = vec![];
        let mut qweights_trial = vec![];
        let mut qpoints_test = vec![];
        let mut qpoints_trial = vec![];
        let mut test_tables = vec![];
        let mut trial_tables = vec![];
        let mut test_cell_types = vec![];
        let mut trial_cell_types = vec![];

        let mut cell_blocks = vec![];

        let mut cell_type_indices = HashMap::new();

        for test_cell_type in grid.entity_types(2) {
            let npts_test = self.options().quadrature_degrees[test_cell_type];
            for trial_cell_type in grid.entity_types(2) {
                let npts_trial = self.options().quadrature_degrees[trial_cell_type];
                test_cell_types.push(*test_cell_type);
                trial_cell_types.push(*trial_cell_type);
                cell_type_indices.insert((*test_cell_type, *trial_cell_type), qweights_test.len());

                let qrule_test = simplex_rule(*test_cell_type, npts_test).unwrap();
                let mut test_pts =
                    rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts_test, 2]);
                for i in 0..npts_test {
                    for j in 0..2 {
                        *test_pts.get_mut([i, j]).unwrap() =
                            num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                qrule_test.points[2 * i + j],
                            )
                            .unwrap();
                    }
                }
                qweights_test.push(
                    qrule_test
                        .weights
                        .iter()
                        .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                        .collect::<Vec<_>>(),
                );
                let test_element = test_space.element(*test_cell_type);
                let mut test_table = rlst_dynamic_array4!(
                    Self::T,
                    test_element.tabulate_array_shape(Self::TABLE_DERIVS, npts_test)
                );
                test_element.tabulate(&test_pts, Self::TABLE_DERIVS, &mut test_table);
                test_tables.push(test_table);
                qpoints_test.push(test_pts);

                let qrule_trial = simplex_rule(*trial_cell_type, npts_trial).unwrap();
                let mut trial_pts =
                    rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts_trial, 2]);
                for i in 0..npts_trial {
                    for j in 0..2 {
                        *trial_pts.get_mut([i, j]).unwrap() =
                            num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                qrule_trial.points[2 * i + j],
                            )
                            .unwrap();
                    }
                }
                qweights_trial.push(
                    qrule_trial
                        .weights
                        .iter()
                        .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                        .collect::<Vec<_>>(),
                );
                let trial_element = trial_space.element(*trial_cell_type);
                let mut trial_table = rlst_dynamic_array4!(
                    Self::T,
                    trial_element.tabulate_array_shape(Self::TABLE_DERIVS, npts_trial)
                );
                trial_element.tabulate(&trial_pts, Self::TABLE_DERIVS, &mut trial_table);
                trial_tables.push(trial_table);
                qpoints_trial.push(trial_pts);
            }
        }
        let mut cell_pairs: Vec<Vec<(usize, usize)>> = vec![vec![]; qweights_test.len()];

        for vertex in grid.entity_iter(0) {
            for test_cell_index in vertex.topology().connected_entity_iter(2) {
                let test_cell = grid.entity(2, test_cell_index).unwrap();
                let test_cell_type = test_cell.entity_type();
                if test_cell.ownership() == Ownership::Owned {
                    for trial_cell_index in vertex.topology().connected_entity_iter(2) {
                        let trial_cell = grid.entity(2, trial_cell_index).unwrap();
                        let trial_cell_type = trial_cell.entity_type();

                        if get_pairs_if_smallest(&test_cell, &trial_cell, vertex.local_index()).is_some() {
                            cell_pairs[cell_type_indices[&(test_cell_type, trial_cell_type)]]
                                .push((test_cell_index, trial_cell_index));
                        }
                    }
                }
            }
        }
        let batch_size = self.options().batch_size;
        for (i, cells) in cell_pairs.iter().enumerate() {
            let mut start = 0;
            while start < cells.len() {
                let end = std::cmp::min(start + batch_size, cells.len());
                cell_blocks.push((i, &cells[start..end]));
                start = end;
            }
        }

        cell_blocks
            .into_par_iter()
            .map(|(i, cell_block)| {
                assemble_batch_singular_correction::<Self::T, TestGrid, TrialGrid, Element>(
                    self,
                    Self::DERIV_SIZE,
                    shape,
                    trial_cell_types[i],
                    test_cell_types[i],
                    trial_space,
                    test_space,
                    cell_block,
                    &qpoints_trial[i],
                    &qweights_trial[i],
                    &qpoints_test[i],
                    &qweights_test[i],
                    &trial_tables[i],
                    &test_tables[i],
                )
            })
            .reduce(
                || SparseMatrixData::<Self::T>::new(shape),
                |mut a, b| {
                    a.add(b);
                    a
                },
            )
    }

    /// Assemble the singular contributions into a dense matrix
    fn assemble_singular_into_dense<
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) {
        let sparse_matrix = self.assemble_singular::<TestGrid, TrialGrid, Element>(
            output.shape(),
            trial_space,
            test_space,
        );
        let data = sparse_matrix.data;
        let rows = sparse_matrix.rows;
        let cols = sparse_matrix.cols;
        for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
            *output.get_mut([*i, *j]).unwrap() += *value;
        }
    }

    /// Assemble the singular contributions into a CSR sparse matrix
    fn assemble_singular_into_csr<
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) -> CsrMatrix<Self::T> {
        let shape = [test_space.global_size(), trial_space.global_size()];
        let sparse_matrix =
            self.assemble_singular::<TestGrid, TrialGrid, Element>(shape, trial_space, test_space);

        CsrMatrix::<Self::T>::from_aij(
            sparse_matrix.shape,
            &sparse_matrix.rows,
            &sparse_matrix.cols,
            &sparse_matrix.data,
        )
        .unwrap()
    }

    #[cfg(feature = "mpi")]
    /// Assemble the singular contributions into a CSR sparse matrix, indexed by global DOF numbers
    fn parallel_assemble_singular_into_csr<
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
        SerialTestSpace: FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync,
        SerialTrialSpace: FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync,
    >(
        &self,
        trial_space: &(impl FunctionSpaceInParallel<SerialSpace = SerialTrialSpace> + FunctionSpace),
        test_space: &(impl FunctionSpaceInParallel<SerialSpace = SerialTestSpace> + FunctionSpace),
    ) -> CsrMatrix<Self::T> {
        self.assemble_singular_into_csr(trial_space.local_space(), test_space.local_space())
    }

    /// Assemble the singular correction into a dense matrix
    ///
    /// The singular correction is the contribution is the terms for adjacent cells are assembled using an (incorrect) non-singular quadrature rule
    fn assemble_singular_correction_into_dense<
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) {
        let sparse_matrix = self.assemble_singular_correction::<TestGrid, TrialGrid, Element>(
            output.shape(),
            trial_space,
            test_space,
        );
        let data = sparse_matrix.data;
        let rows = sparse_matrix.rows;
        let cols = sparse_matrix.cols;
        for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
            *output.get_mut([*i, *j]).unwrap() += *value;
        }
    }

    /// Assemble the singular correction into a CSR matrix
    ///
    /// The singular correction is the contribution is the terms for adjacent cells are assembled using an (incorrect) non-singular quadrature rule
    fn assemble_singular_correction_into_csr<
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) -> CsrMatrix<Self::T> {
        let shape = [test_space.global_size(), trial_space.global_size()];
        let sparse_matrix = self.assemble_singular_correction::<TestGrid, TrialGrid, Element>(
            shape,
            trial_space,
            test_space,
        );

        CsrMatrix::<Self::T>::from_aij(
            sparse_matrix.shape,
            &sparse_matrix.rows,
            &sparse_matrix.cols,
            &sparse_matrix.data,
        )
        .unwrap()
    }

    #[cfg(feature = "mpi")]
    /// Assemble the singular contributions into a CSR sparse matrix, indexed by global DOF numbers
    fn parallel_assemble_singular_correction_into_csr<
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
        SerialTestSpace: FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync,
        SerialTrialSpace: FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync,
    >(
        &self,
        trial_space: &(impl FunctionSpaceInParallel<SerialSpace = SerialTrialSpace> + FunctionSpace),
        test_space: &(impl FunctionSpaceInParallel<SerialSpace = SerialTestSpace> + FunctionSpace),
    ) -> CsrMatrix<Self::T> {
        self.assemble_singular_correction_into_csr(
            trial_space.local_space(),
            test_space.local_space(),
        )
    }

    /// Assemble into a dense matrix
    fn assemble_into_dense<
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) {
        let test_colouring = test_space.cell_colouring();
        let trial_colouring = trial_space.cell_colouring();

        self.assemble_nonsingular_into_dense::<TestGrid, TrialGrid, Element>(
            output,
            trial_space,
            test_space,
            &trial_colouring,
            &test_colouring,
        );
        self.assemble_singular_into_dense::<TestGrid, TrialGrid, Element>(
            output,
            trial_space,
            test_space,
        );
    }

    /// Assemble the non-singular contributions into a dense matrix
    fn assemble_nonsingular_into_dense<
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
        trial_colouring: &HashMap<ReferenceCellType, Vec<Vec<usize>>>,
        test_colouring: &HashMap<ReferenceCellType, Vec<Vec<usize>>>,
    ) {
        if !trial_space.is_serial() || !test_space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if output.shape()[0] != test_space.global_size()
            || output.shape()[1] != trial_space.global_size()
        {
            panic!("Matrix has wrong shape");
        }

        let batch_size = self.options().batch_size;

        for test_cell_type in test_space.grid().entity_types(2) {
            let npts_test = self.options().quadrature_degrees[test_cell_type];
            for trial_cell_type in trial_space.grid().entity_types(2) {
                let npts_trial = self.options().quadrature_degrees[trial_cell_type];
                let qrule_test = simplex_rule(*test_cell_type, npts_test).unwrap();
                let mut qpoints_test =
                    rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts_test, 2]);
                for i in 0..npts_test {
                    for j in 0..2 {
                        *qpoints_test.get_mut([i, j]).unwrap() =
                            num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                qrule_test.points[2 * i + j],
                            )
                            .unwrap();
                    }
                }
                let qweights_test = qrule_test
                    .weights
                    .iter()
                    .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                    .collect::<Vec<_>>();
                let qrule_trial = simplex_rule(*trial_cell_type, npts_trial).unwrap();
                let mut qpoints_trial =
                    rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts_trial, 2]);
                for i in 0..npts_trial {
                    for j in 0..2 {
                        *qpoints_trial.get_mut([i, j]).unwrap() =
                            num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                qrule_trial.points[2 * i + j],
                            )
                            .unwrap();
                    }
                }
                let qweights_trial = qrule_trial
                    .weights
                    .iter()
                    .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                    .collect::<Vec<_>>();

                let test_element = test_space.element(*test_cell_type);
                let mut test_table = rlst_dynamic_array4!(
                    Self::T,
                    test_element.tabulate_array_shape(Self::TABLE_DERIVS, npts_test)
                );
                test_element.tabulate(&qpoints_test, Self::TABLE_DERIVS, &mut test_table);

                let trial_element = trial_space.element(*trial_cell_type);
                let mut trial_table = rlst_dynamic_array4!(
                    Self::T,
                    trial_element.tabulate_array_shape(Self::TABLE_DERIVS, npts_trial)
                );
                trial_element.tabulate(&qpoints_test, Self::TABLE_DERIVS, &mut trial_table);

                let output_raw = RawData2D {
                    data: output.data_mut().as_mut_ptr(),
                    shape: output.shape(),
                };

                for test_c in &test_colouring[test_cell_type] {
                    for trial_c in &trial_colouring[trial_cell_type] {
                        let mut test_cells: Vec<&[usize]> = vec![];
                        let mut trial_cells: Vec<&[usize]> = vec![];

                        let mut test_start = 0;
                        while test_start < test_c.len() {
                            let test_end = if test_start + batch_size < test_c.len() {
                                test_start + batch_size
                            } else {
                                test_c.len()
                            };

                            let mut trial_start = 0;
                            while trial_start < trial_c.len() {
                                let trial_end = if trial_start + batch_size < trial_c.len() {
                                    trial_start + batch_size
                                } else {
                                    trial_c.len()
                                };
                                test_cells.push(&test_c[test_start..test_end]);
                                trial_cells.push(&trial_c[trial_start..trial_end]);
                                trial_start = trial_end;
                            }
                            test_start = test_end
                        }

                        let numtasks = test_cells.len();
                        let r: usize = (0..numtasks)
                            .into_par_iter()
                            .map(&|t| {
                                assemble_batch_nonadjacent::<Self::T, TestGrid, TrialGrid, Element>(
                                    self,
                                    Self::DERIV_SIZE,
                                    &output_raw,
                                    *test_cell_type, *trial_cell_type,
                                    trial_space,
                                    trial_cells[t],
                                    test_space,
                                    test_cells[t],
                                    &qpoints_trial,
                                    &qweights_trial,
                                    &qpoints_test,
                                    &qweights_test,
                                    &trial_table,
                                    &test_table,
                                )
                            })
                            .sum();
                        assert_eq!(r, numtasks);
                    }
                }
            }
        }
    }
}
