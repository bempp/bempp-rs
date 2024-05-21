//! Batched dense assembly of boundary operators
use crate::assembly::common::{equal_grids, RawData2D, SparseMatrixData};
use crate::element::reference_cell;
use crate::grid::common::{compute_dets23, compute_normals_from_jacobians23};
use crate::quadrature::duffy::{
    quadrilateral_duffy, quadrilateral_triangle_duffy, triangle_duffy, triangle_quadrilateral_duffy,
};
use crate::quadrature::simplex_rules::simplex_rule;
use crate::quadrature::types::{CellToCellConnectivity, TestTrialNumericalQuadratureDefinition};
use crate::traits::element::FiniteElement;
use crate::traits::function::FunctionSpace;
#[cfg(feature = "mpi")]
use crate::traits::function::FunctionSpaceInParallel;
use crate::traits::grid::{CellType, GridType, ReferenceMapType, TopologyType};
use crate::traits::types::Ownership;
use crate::traits::types::ReferenceCellType;
use rayon::prelude::*;
use rlst::{
    rlst_dynamic_array2, rlst_dynamic_array3, rlst_dynamic_array4, CsrMatrix, RandomAccessMut,
    RawAccess, RawAccessMut, RlstScalar, Shape, UnsafeRandomAccessByRef,
};
use std::collections::HashMap;

use super::RlstArray;

/// Assemble the contribution to the terms of a matrix for a batch of non-adjacent cells
#[allow(clippy::too_many_arguments)]
fn assemble_batch_nonadjacent<
    T: RlstScalar,
    TestGrid: GridType<T = T::Real>,
    TrialGrid: GridType<T = T::Real>,
    Element: FiniteElement<T = T> + Sync,
>(
    assembler: &impl BatchedAssembler<T = T>,
    deriv_size: usize,
    output: &RawData2D<T>,
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

    assert_eq!(test_grid.physical_dimension(), 3);
    assert_eq!(test_grid.domain_dimension(), 2);
    assert_eq!(trial_grid.physical_dimension(), 3);
    assert_eq!(trial_grid.domain_dimension(), 2);

    let mut k = rlst_dynamic_array3!(T, [npts_test, deriv_size, npts_trial]);
    let zero = num::cast::<f64, T::Real>(0.0).unwrap();
    let mut test_jdet = vec![zero; npts_test];
    let mut test_mapped_pts = rlst_dynamic_array2!(T::Real, [npts_test, 3]);
    let mut test_normals = rlst_dynamic_array2!(T::Real, [npts_test, 3]);
    let mut test_jacobians = rlst_dynamic_array2!(T::Real, [npts_test, 6]);

    let test_evaluator = test_grid.reference_to_physical_map(test_points.data());
    let trial_evaluator = trial_grid.reference_to_physical_map(trial_points.data());

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
        trial_evaluator.jacobian(*trial_cell, trial_jacobians[trial_cell_i].data_mut());
        compute_dets23(
            trial_jacobians[trial_cell_i].data(),
            &mut trial_jdet[trial_cell_i],
        );
        compute_normals_from_jacobians23(
            trial_jacobians[trial_cell_i].data(),
            trial_normals[trial_cell_i].data_mut(),
        );
        trial_evaluator
            .reference_to_physical(*trial_cell, trial_mapped_pts[trial_cell_i].data_mut());
    }

    let mut sum: T;
    let mut trial_integrands = vec![num::cast::<f64, T>(0.0).unwrap(); npts_trial];

    for test_cell in test_cells {
        test_evaluator.jacobian(*test_cell, test_jacobians.data_mut());
        compute_dets23(test_jacobians.data(), &mut test_jdet);
        compute_normals_from_jacobians23(test_jacobians.data(), test_normals.data_mut());
        test_evaluator.reference_to_physical(*test_cell, test_mapped_pts.data_mut());

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

/// Options for a batched assembler
pub struct BatchedPotentialAssemblerOptions {
    /// Number of points used in quadrature for non-singular integrals
    quadrature_degrees: HashMap<ReferenceCellType, usize>,
    /// Maximum size of each batch of cells to send to an assembly function
    batch_size: usize,
}

impl Default for BatchedPotentialAssemblerOptions {
    fn default() -> Self {
        use ReferenceCellType::{Quadrilateral, Triangle};
        Self {
            quadrature_degrees: HashMap::from([(Triangle, 37), (Quadrilateral, 37)]),
            batch_size: 128,
        }
    }
}

pub trait BatchedPotentialAssembler: Sync + Sized {
    //! Batched potential assembler
    //!
    //! Assemble potential operators by processing batches of cells in parallel

    /// Scalar type
    type T: RlstScalar;
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

    /// Set the maximum size of a batch of cells to send to an assembly function
    fn batch_size(&mut self, size: usize) {
        self.options_mut().batch_size = size;
    }

    /// Return the kernel value to use in the integrand
    ///
    /// # Safety
    /// This method is unsafe to allow `get_unchecked` to be used
    unsafe fn kernel_value(
        &self,
        k: &RlstArray<Self::T, 2>,
        normals: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        index: usize,
    ) -> Self::T;

    /// Evaluate the kernel values for all sources and all targets
    ///
    /// For every source, the kernel is evaluated for every target.
    fn kernel_assemble_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    );

    /// Assemble into a dense matrix
    fn assemble_into_dense<
        TestGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        TrialGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        points: TODO,
    ) {
        if !space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if output.shape()[0] != points.shape()[0]
            || output.shape()[1] != space.global_size()
        {
            panic!("Matrix has wrong shape");
        }

        let colouring = space.cell_colouring();

        let batch_size = self.options().batch_size;

        for cell_type in space.grid().cell_types() {
            let npts = self.options().quadrature_degrees[cell_type];
            let qrule = simplex_rule(*trial_cell_type, npts_trial).unwrap();
            let mut qpoints =
                rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts, 2]);
            for i in 0..npts {
                for j in 0..2 {
                    *qpoints.get_mut([i, j]).unwrap() =
                        num::cast::<f64, <Self::T as RlstScalar>::Real>(
                            qrule.points[2 * i + j],
                        )
                        .unwrap();
                }
            }
            let qweights = qrule
                .weights
                .iter()
                .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                .collect::<Vec<_>>();

            let element = trial_space.element(*cell_type);
            let mut table = rlst_dynamic_array4!(
                Self::T,
                element.tabulate_array_shape(Self::TABLE_DERIVS, npts)
            );
            element.tabulate(&qpoints_test, Self::TABLE_DERIVS, &mut table);

            let output_raw = RawData2D {
                data: output.data_mut().as_mut_ptr(),
                shape: output.shape(),
            };

            for c in &colouring[cell_type] {
                let mut cells: Vec<&[usize]> = vec![];

                let mut start = 0;
                while start < c.len() {
                    let end = if start + batch_size < c.len() {
                        start + batch_size
                    } else {
                        c.len()
                    };

                    cells.push(&c[start..end]);
                    start = end
                }

                let numtasks = cells.len();
                let r: usize = (0..numtasks)
                    .into_par_iter()
                    .map(&|t| {
                        assemble_batch_nonadjacent::<Self::T, TestGrid, TrialGrid, Element>(
                            self,
                            Self::DERIV_SIZE,
                            &output_raw,
                            space,
                            points,
                            cells[t],
                            &qpoints,
                            &qweights,
                            &table,
                        )
                    })
                    .sum();
                assert_eq!(r, numtasks);
            }
        }
    }
}
