//! Various helper functions to support evaluators.

use std::marker::PhantomData;

use green_kernels::traits::Kernel;
use itertools::{izip, Itertools};
use mpi::traits::{Communicator, Equivalence};
use ndelement::traits::FiniteElement;
use ndgrid::{
    traits::{Entity, GeometryMap, Grid, ParallelGrid},
    types::Ownership,
};

use rayon::prelude::*;
use rlst::{
    operator::interface::{
        distributed_sparse_operator::DistributedCsrMatrixOperator, DistributedArrayVectorSpace,
    },
    rlst_array_from_slice2, rlst_dynamic_array1, rlst_dynamic_array2, rlst_dynamic_array3,
    rlst_dynamic_array4, AsApply, DefaultIterator, DistributedCsrMatrix, Element, IndexLayout,
    OperatorBase, RawAccess, RawAccessMut, RlstScalar,
};

use crate::function::FunctionSpaceTrait;

/// Create a linear operator from the map of a basis to points.
pub fn basis_to_point_map<
    'a,
    C: Communicator,
    DomainLayout: IndexLayout<Comm = C>,
    RangeLayout: IndexLayout<Comm = C>,
    T: RlstScalar + Equivalence,
    Space: FunctionSpaceTrait<T = T>,
>(
    function_space: &Space,
    domain_space: &'a DistributedArrayVectorSpace<'a, DomainLayout, T>,
    range_space: &'a DistributedArrayVectorSpace<'a, RangeLayout, T>,
    quadrature_points: &[T::Real],
    quadrature_weights: &[T::Real],
    return_transpose: bool,
) -> DistributedCsrMatrixOperator<'a, DomainLayout, RangeLayout, T, C>
where
    T::Real: Equivalence,
{
    // Get the grid.
    let grid = function_space.grid();

    // Topological dimension of the grid.
    let tdim = grid.topology_dim();

    // The method is currently restricted to single element type grids.
    // So let's make sure that this is the case.

    assert_eq!(grid.entity_types(tdim).len(), 1);

    let reference_cell = grid.entity_types(tdim)[0];

    // Number of cells. We are only interested in owned cells.
    let n_cells = grid
        .entity_iter(tdim)
        .filter(|e| matches!(e.ownership(), Ownership::Owned))
        .count();

    // Get the number of quadrature points and check that weights
    // and points have compatible dimensions.

    let n_quadrature_points = quadrature_weights.len();
    assert_eq!(quadrature_points.len() % tdim, 0);
    assert_eq!(quadrature_points.len() / tdim, n_quadrature_points);

    // Check that domain space and function space are compatible.

    let n_domain_dofs = domain_space.index_layout().number_of_global_indices();
    let n_range_dofs = range_space.index_layout().number_of_global_indices();

    assert_eq!(function_space.local_size(), n_domain_dofs,);

    assert_eq!(n_cells * n_quadrature_points, n_range_dofs);

    // All the dimensions are OK. Let's get to work. We need to iterate through the elements,
    // get the attached global dofs and the corresponding jacobian map.

    // Let's first tabulate the basis function values at the quadrature points on the reference element.

    // Quadrature point is needed here as a RLST matrix.

    let quadrature_points = rlst_array_from_slice2!(quadrature_points, [tdim, n_quadrature_points]);

    let mut table = rlst_dynamic_array4!(
        T,
        function_space
            .element(reference_cell)
            .tabulate_array_shape(0, n_quadrature_points)
    );
    function_space
        .element(reference_cell)
        .tabulate(&quadrature_points, 0, &mut table);

    // We have tabulated the basis functions on the reference element. Now need
    // the map to physical elements.

    let geometry_evaluator = grid.geometry_map(reference_cell, quadrature_points.data());

    // The following arrays hold the jacobians, their determinants and the normals.

    let mut jacobians =
        rlst_dynamic_array3![T::Real, [grid.geometry_dim(), tdim, n_quadrature_points]];
    let mut jdets = rlst_dynamic_array1![T::Real, [n_quadrature_points]];
    let mut normals = rlst_dynamic_array2![T::Real, [grid.geometry_dim(), n_quadrature_points]];

    // Now iterate through the cells of the grid, get the attached dofs and evaluate the geometry map.

    // These arrays store the data of the transformation matrix.
    let mut rows = Vec::<usize>::default();
    let mut cols = Vec::<usize>::default();
    let mut data = Vec::<T>::default();

    for cell in grid
        .entity_iter(tdim)
        .filter(|e| matches!(e.ownership(), Ownership::Owned))
    {
        // Get the Jacobians
        geometry_evaluator.jacobians_dets_normals(
            cell.local_index(),
            jacobians.data_mut(),
            jdets.data_mut(),
            normals.data_mut(),
        );
        // Get the global dofs of the cell.
        let global_dofs = function_space
            .cell_dofs(cell.local_index())
            .unwrap()
            .iter()
            .map(|local_dof_index| function_space.global_dof_index(*local_dof_index))
            .collect::<Vec<_>>();

        for (qindex, (jdet, qweight)) in izip!(jdets.iter(), quadrature_weights.iter()).enumerate()
        {
            for (i, global_dof) in global_dofs.iter().enumerate() {
                rows.push(n_quadrature_points * cell.global_index() + qindex);
                cols.push(*global_dof);
                data.push(T::from_real(jdet * *qweight) * table[[0, qindex, i, 0]]);
            }
        }
    }

    if return_transpose {
        DistributedCsrMatrixOperator::new(
            DistributedCsrMatrix::from_aij(
                domain_space.index_layout(),
                range_space.index_layout(),
                &cols,
                &rows,
                &data,
            ),
            domain_space,
            range_space,
        )
    } else {
        DistributedCsrMatrixOperator::new(
            DistributedCsrMatrix::from_aij(
                domain_space.index_layout(),
                range_space.index_layout(),
                &rows,
                &cols,
                &data,
            ),
            domain_space,
            range_space,
        )
    }
}

/// A linear operator that evaluates kernel interactions for a nonsingular quadrature rule on neighbouring elements.
///
/// This creates a linear operator which for a given kernel evaluates all the interactions between neighbouring elements on a set
/// of local points per triangle.
pub struct NeighbourEvaluator<
    'a,
    T: RlstScalar + Equivalence,
    K: Kernel<T = T>,
    DomainLayout: IndexLayout<Comm = C>,
    RangeLayout: IndexLayout<Comm = C>,
    GridImpl: ParallelGrid<C, T = T::Real>,
    C: Communicator,
> {
    eval_points: Vec<T::Real>,
    kernel: K,
    domain_space: &'a DistributedArrayVectorSpace<'a, DomainLayout, T>,
    range_space: &'a DistributedArrayVectorSpace<'a, RangeLayout, T>,
    grid: &'a GridImpl,
    active_cells: Vec<usize>,
    _marker: PhantomData<C>,
}

impl<
        'a,
        T: RlstScalar + Equivalence,
        K: Kernel<T = T>,
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        GridImpl: ParallelGrid<C, T = T::Real>,
        C: Communicator,
    > NeighbourEvaluator<'a, T, K, DomainLayout, RangeLayout, GridImpl, C>
{
    /// Create a new neighbour evaluator.
    pub fn new(
        eval_points: &[T::Real],
        kernel: K,
        domain_space: &'a DistributedArrayVectorSpace<'a, DomainLayout, T>,
        range_space: &'a DistributedArrayVectorSpace<'a, RangeLayout, T>,
        grid: &'a GridImpl,
    ) -> Self {
        // Check that the domain space and range space are compatible with the grid.
        // Topological dimension of the grid.
        let tdim = grid.topology_dim();

        // The method is currently restricted to single element type grids.
        // So let's make sure that this is the case.

        assert_eq!(grid.entity_types(tdim).len(), 1);

        // Get the number of points
        assert_eq!(eval_points.len() % tdim, 0);
        let n_points = eval_points.len() / tdim;

        // The active cells are those that we need to iterate over.
        // At the moment these are simply all owned cells in the grid.
        // We sort the active cells by global index. This is important so that in the evaluation
        // we can just iterate the output vector through in chunks and know from the chunk which
        // active cell it is associated with.

        let active_cells: Vec<usize> = grid
            .entity_iter(tdim)
            .filter(|e| matches!(e.ownership(), Ownership::Owned))
            .sorted_by_key(|e| e.global_index())
            .map(|e| e.local_index())
            .collect_vec();

        let n_cells = active_cells.len();

        // Check that domain space and function space are compatible with the grid.

        assert_eq!(
            domain_space.index_layout().number_of_local_indices(),
            n_cells * n_points
        );

        assert_eq!(
            range_space.index_layout().number_of_local_indices(),
            n_cells * n_points
        );

        Self {
            eval_points: eval_points.to_vec(),
            kernel,
            domain_space,
            range_space,
            grid,
            active_cells,
            _marker: PhantomData,
        }
    }
}

impl<
        T: RlstScalar + Equivalence,
        K: Kernel<T = T>,
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        GridImpl: ParallelGrid<C, T = T::Real>,
        C: Communicator,
    > std::fmt::Debug for NeighbourEvaluator<'_, T, K, DomainLayout, RangeLayout, GridImpl, C>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Neighbourhood Evaluator with dimenion [{}, {}].",
            self.range_space.index_layout().number_of_global_indices(),
            self.domain_space.index_layout().number_of_global_indices()
        )
    }
}

impl<
        'a,
        T: RlstScalar + Equivalence,
        K: Kernel<T = T>,
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        GridImpl: ParallelGrid<C, T = T::Real>,
        C: Communicator,
    > OperatorBase for NeighbourEvaluator<'a, T, K, DomainLayout, RangeLayout, GridImpl, C>
{
    type Domain = DistributedArrayVectorSpace<'a, DomainLayout, T>;

    type Range = DistributedArrayVectorSpace<'a, RangeLayout, T>;

    fn domain(&self) -> &Self::Domain {
        self.domain_space
    }

    fn range(&self) -> &Self::Range {
        self.range_space
    }
}

impl<
        T: RlstScalar + Equivalence,
        K: Kernel<T = T>,
        DomainLayout: IndexLayout<Comm = C> + Sync,
        RangeLayout: IndexLayout<Comm = C> + Sync,
        GridImpl: ParallelGrid<C, T = T::Real> + Sync,
        C: Communicator,
    > AsApply for NeighbourEvaluator<'_, T, K, DomainLayout, RangeLayout, GridImpl, C>
{
    fn apply_extended(
        &self,
        alpha: <Self::Range as rlst::LinearSpace>::F,
        x: &<Self::Domain as rlst::LinearSpace>::E,
        beta: <Self::Range as rlst::LinearSpace>::F,
        y: &mut <Self::Range as rlst::LinearSpace>::E,
    ) -> rlst::RlstResult<()> {
        // We need to iterate through the elements.

        let tdim = self.grid.topology_dim();

        // In the new function we already made sure that eval_points is a multiple of tdim.
        let n_points = self.eval_points.len() / tdim;

        // We go through groups of target dofs in chunks of lenth n_points.
        // This corresponds to iteration in active cells since we ordered those
        // already by global index.

        let raw_ptr = y.view_mut().local_mut().data_mut().as_mut_ptr();

        // y.view_mut()
        //     .local_mut()
        //     .data_mut()
        //     .par_chunks_mut(n_points)
        //     .enumerate()
        //     .for_each(|(chunk_index, chunk)| {
        //         let cell = self.active_cells[chunk_index];
        //         let cell_entity = self.grid.entity(tdim, cell).unwrap();

        //         // Get the geometry map for the cell.
        //         let geometry_map = self
        //             .grid
        //             .geometry_map(cell_entity.entity_type(), &self.eval_points);
        //     });

        Ok(())
    }
}
