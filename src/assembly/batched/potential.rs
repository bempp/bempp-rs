//! Batched dense assembly of potential operators
pub(crate) mod double_layer;
pub(crate) mod single_layer;

use crate::assembly::common::RawData2D;
use crate::quadrature::simplex_rules::simplex_rule;
use crate::traits::function::FunctionSpace;
use ndelement::traits::FiniteElement;
use ndelement::types::ReferenceCellType;
use ndgrid::traits::{GeometryMap, Grid};
use rayon::prelude::*;
use rlst::{
    rlst_dynamic_array2, rlst_dynamic_array3, rlst_dynamic_array4, MatrixInverse, RandomAccessMut,
    RawAccess, RawAccessMut, RlstScalar, Shape, UnsafeRandomAccessByRef,
};
use std::collections::HashMap;

use super::RlstArray;

/// Assemble the contribution to the terms of a matrix for a batch of non-adjacent cells
#[allow(clippy::too_many_arguments)]
fn assemble_batch<
    T: RlstScalar + MatrixInverse,
    G: Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    Element: FiniteElement<T = T> + Sync,
>(
    assembler: &impl BatchedPotentialAssembler<T = T>,
    deriv_size: usize,
    output: &RawData2D<T>,
    cell_type: ReferenceCellType,
    space: &impl FunctionSpace<Grid = G, FiniteElement = Element>,
    evaluation_points: &RlstArray<T::Real, 2>,
    cells: &[usize],
    points: &RlstArray<T::Real, 2>,
    weights: &[T::Real],
    table: &RlstArray<T, 4>,
) -> usize {
    let npts = weights.len();
    let nevalpts = evaluation_points.shape()[1];
    debug_assert!(points.shape()[1] == npts);

    let grid = space.grid();

    assert_eq!(grid.geometry_dim(), 3);
    assert_eq!(grid.topology_dim(), 2);

    let mut k = rlst_dynamic_array3!(T, [deriv_size, npts, nevalpts]);
    let zero = num::cast::<f64, T::Real>(0.0).unwrap();
    let mut jdet = vec![zero; npts];
    let mut mapped_pts = rlst_dynamic_array2!(T::Real, [3, npts]);
    let mut normals = rlst_dynamic_array2!(T::Real, [3, npts]);
    let mut jacobians = rlst_dynamic_array2!(T::Real, [6, npts]);

    let evaluator = grid.geometry_map(cell_type, points.data());

    let mut sum: T;

    for cell in cells {
        evaluator.points(*cell, mapped_pts.data_mut());
        evaluator.jacobians_dets_normals(
            *cell,
            jacobians.data_mut(),
            &mut jdet,
            normals.data_mut(),
        );

        assembler.kernel_assemble_st(mapped_pts.data(), evaluation_points.data(), k.data_mut());

        let dofs = space.cell_dofs(*cell).unwrap();

        for (i, dof) in dofs.iter().enumerate() {
            for eval_index in 0..nevalpts {
                sum = num::cast::<f64, T>(0.0).unwrap();
                for (index, wt) in weights.iter().enumerate() {
                    sum += unsafe {
                        assembler.kernel_value(&k, &normals, index, eval_index)
                            * num::cast::<T::Real, T>(*wt * jdet[index]).unwrap()
                            * *table.get_unchecked([0, index, i, 0])
                    };
                }
                unsafe {
                    *output.data.add(eval_index + output.shape[0] * *dof) += sum;
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
    type T: RlstScalar + MatrixInverse;
    /// Number of derivatives
    const DERIV_SIZE: usize;

    /// Get assembler options
    fn options(&self) -> &BatchedPotentialAssemblerOptions;

    /// Get mutable assembler options
    fn options_mut(&mut self) -> &mut BatchedPotentialAssemblerOptions;

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
        k: &RlstArray<Self::T, 3>,
        normals: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        index: usize,
        point_index: usize,
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
        G: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        space: &(impl FunctionSpace<Grid = G, FiniteElement = Element> + Sync),
        points: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
    ) {
        if !space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if output.shape()[0] != points.shape()[1] || output.shape()[1] != space.global_size() {
            panic!("Matrix has wrong shape");
        }

        let colouring = space.cell_colouring();

        let batch_size = self.options().batch_size;

        for cell_type in space.grid().entity_types(2) {
            let npts = self.options().quadrature_degrees[cell_type];
            let qrule = simplex_rule(*cell_type, npts).unwrap();
            let mut qpoints = rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [2, npts]);
            for i in 0..npts {
                for j in 0..2 {
                    *qpoints.get_mut([j, i]).unwrap() =
                        num::cast::<f64, <Self::T as RlstScalar>::Real>(qrule.points[2 * i + j])
                            .unwrap();
                }
            }
            let qweights = qrule
                .weights
                .iter()
                .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                .collect::<Vec<_>>();

            let element = space.element(*cell_type);
            let mut table = rlst_dynamic_array4!(Self::T, element.tabulate_array_shape(0, npts));
            element.tabulate(&qpoints, 0, &mut table);

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
                        assemble_batch::<Self::T, G, Element>(
                            self,
                            Self::DERIV_SIZE,
                            &output_raw,
                            *cell_type,
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
