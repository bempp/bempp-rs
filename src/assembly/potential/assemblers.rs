//! Potential assemblers
mod double_layer;
mod single_layer;

use crate::assembly::common::{RawData2D, RlstArray};
use crate::assembly::potential::cell_assemblers::PotentialCellAssembler;
use crate::quadrature::simplex_rules::simplex_rule;
use crate::traits::{
    CellAssembler, FunctionSpace, KernelEvaluator, PotentialAssembly, PotentialIntegrand,
};
use itertools::izip;
use ndelement::traits::FiniteElement;
use ndelement::types::ReferenceCellType;
use ndgrid::traits::Grid;
use rayon::prelude::*;
use rlst::{
    rlst_dynamic_array2, rlst_dynamic_array4, DefaultIterator, MatrixInverse, RandomAccessMut,
    RawAccess, RawAccessMut, RlstScalar, Shape,
};
use std::collections::HashMap;

/// Assemble the contribution to the terms of a matrix for a batch of non-adjacent cells
#[allow(clippy::too_many_arguments)]
fn assemble_batch<
    T: RlstScalar + MatrixInverse,
    Space: FunctionSpace<T = T> + Sync,
    Integrand: PotentialIntegrand<T = T>,
    Kernel: KernelEvaluator<T = T>,
>(
    assembler: &PotentialAssembler<T, Integrand, Kernel>,
    deriv_size: usize,
    output: &RawData2D<T>,
    cell_type: ReferenceCellType,
    space: &Space,
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

    let evaluator = grid.geometry_map(cell_type, points.data());
    let mut a = PotentialCellAssembler::new(
        npts,
        nevalpts,
        deriv_size,
        &assembler.integrand,
        &assembler.kernel,
        evaluator,
        table,
        evaluation_points,
        weights,
    );

    let mut local_mat = rlst_dynamic_array2!(T, [nevalpts, space.element(cell_type).dim()]);

    for cell in cells {
        a.set_cell(*cell);
        a.assemble(&mut local_mat);

        let dofs = space.cell_dofs(*cell).unwrap();
        for (dof, col) in izip!(dofs, local_mat.col_iter()) {
            for (eval_index, entry) in col.iter().enumerate() {
                unsafe {
                    *output.data.add(eval_index + output.shape[0] * *dof) += entry;
                }
            }
        }
    }
    1
}

/// Options for a potential assembler
pub struct PotentialAssemblerOptions {
    /// Number of points used in quadrature for non-singular integrals
    quadrature_degrees: HashMap<ReferenceCellType, usize>,
    /// Maximum size of each batch of cells to send to an assembly function
    batch_size: usize,
}

impl Default for PotentialAssemblerOptions {
    fn default() -> Self {
        use ReferenceCellType::{Quadrilateral, Triangle};
        Self {
            quadrature_degrees: HashMap::from([(Triangle, 37), (Quadrilateral, 37)]),
            batch_size: 128,
        }
    }
}

/// Potential assembler
///
/// Assemble potential operators by processing batches of cells in parallel
pub struct PotentialAssembler<
    T: RlstScalar + MatrixInverse,
    Integrand: PotentialIntegrand<T = T>,
    Kernel: KernelEvaluator<T = T>,
> {
    pub(crate) integrand: Integrand,
    pub(crate) kernel: Kernel,
    pub(crate) options: PotentialAssemblerOptions,
    pub(crate) deriv_size: usize,
}

unsafe impl<
        T: RlstScalar + MatrixInverse,
        Integrand: PotentialIntegrand<T = T>,
        Kernel: KernelEvaluator<T = T>,
    > Sync for PotentialAssembler<T, Integrand, Kernel>
{
}

impl<
        T: RlstScalar + MatrixInverse,
        Integrand: PotentialIntegrand<T = T>,
        Kernel: KernelEvaluator<T = T>,
    > PotentialAssembler<T, Integrand, Kernel>
{
    /// Create new
    fn new(integrand: Integrand, kernel: Kernel, deriv_size: usize) -> Self {
        Self {
            integrand,
            kernel,
            options: PotentialAssemblerOptions::default(),
            deriv_size,
        }
    }

    /// Set (non-singular) quadrature degree for a cell type
    pub fn quadrature_degree(&mut self, cell: ReferenceCellType, degree: usize) {
        *self.options.quadrature_degrees.get_mut(&cell).unwrap() = degree;
    }

    /// Set the maximum size of a batch of cells to send to an assembly function
    pub fn batch_size(&mut self, size: usize) {
        self.options.batch_size = size;
    }

    fn assemble<Space: FunctionSpace<T = T> + Sync>(
        &self,
        output: &RawData2D<T>,
        space: &Space,
        points: &RlstArray<T::Real, 2>,
        colouring: &HashMap<ReferenceCellType, Vec<Vec<usize>>>,
    ) {
        if !space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if output.shape[0] != points.shape()[1] || output.shape[1] != space.global_size() {
            panic!("Matrix has wrong shape");
        }

        let batch_size = self.options.batch_size;

        for cell_type in space.grid().entity_types(2) {
            let npts = self.options.quadrature_degrees[cell_type];
            let qrule = simplex_rule(*cell_type, npts).unwrap();
            let mut qpoints = rlst_dynamic_array2!(T::Real, [2, npts]);
            for i in 0..npts {
                for j in 0..2 {
                    *qpoints.get_mut([j, i]).unwrap() =
                        num::cast::<f64, T::Real>(qrule.points[2 * i + j]).unwrap();
                }
            }
            let qweights = qrule
                .weights
                .iter()
                .map(|w| num::cast::<f64, T::Real>(*w).unwrap())
                .collect::<Vec<_>>();

            let element = space.element(*cell_type);
            let mut table = rlst_dynamic_array4!(T, element.tabulate_array_shape(0, npts));
            element.tabulate(&qpoints, 0, &mut table);

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
                        assemble_batch(
                            self,
                            self.deriv_size,
                            output,
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

impl<
        T: RlstScalar + MatrixInverse,
        Integrand: PotentialIntegrand<T = T>,
        Kernel: KernelEvaluator<T = T>,
    > PotentialAssembly for PotentialAssembler<T, Integrand, Kernel>
{
    type T = T;

    fn assemble_into_dense<Space: FunctionSpace<T = T> + Sync>(
        &self,
        output: &mut RlstArray<T, 2>,
        space: &Space,
        points: &RlstArray<T::Real, 2>,
    ) {
        if !space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if output.shape()[0] != points.shape()[1] || output.shape()[1] != space.global_size() {
            panic!("Matrix has wrong shape");
        }

        let output_raw = RawData2D {
            data: output.data_mut().as_mut_ptr(),
            shape: output.shape(),
        };

        let colouring = space.cell_colouring();

        self.assemble(&output_raw, space, points, &colouring);
    }
}
