//! Various helper functions to support evaluators.

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use green_kernels::{traits::Kernel, types::GreenKernelEvalType};
use itertools::{izip, Itertools};
use mpi::traits::{Communicator, Equivalence};
use ndelement::traits::FiniteElement;
use ndgrid::{
    traits::{Entity, GeometryMap, Grid, ParallelGrid, Topology},
    types::Ownership,
};

use rayon::{prelude::*, range};
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
pub fn neighbour_operator<
    'a,
    C: Communicator,
    T: RlstScalar + Equivalence,
    K: Kernel<T = T>,
    DomainLayout: IndexLayout<Comm = C>,
    RangeLayout: IndexLayout<Comm = C>,
    GridImpl: ParallelGrid<C, T = T::Real>,
>(
    grid: &'a GridImpl,
    kernel: K,
    eval_type: GreenKernelEvalType,
    eval_points: &[T::Real],
    domain_space: &'a DistributedArrayVectorSpace<'a, DomainLayout, T>,
    range_space: &'a DistributedArrayVectorSpace<'a, RangeLayout, T>,
) where
    GridImpl::LocalGrid: Sync,
    for<'b> <GridImpl::LocalGrid as Grid>::GeometryMap<'b>: Sync,
{
    // Check that the domain space and range space are compatible with the grid.
    // Topological dimension of the grid.
    let tdim = grid.topology_dim();

    // Also need the geometric dimension.
    let gdim = grid.geometry_dim();

    // The method is currently restricted to single element type grids.
    // So let's make sure that this is the case.

    assert_eq!(grid.entity_types(tdim).len(), 1);

    let reference_cell = grid.entity_types(tdim)[0];

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

    // We need to figure out how many data entries the sparse matrix will have.
    // For this we need to iterate through all cells and figure out how many neighbours
    // each cell has.

    let mut entries = 0;
    for &cell in &active_cells {
        let n_neighbours = grid
            .entity(tdim, cell)
            .unwrap()
            .topology()
            .connected_entity_iter(tdim)
            .count();
        // For each cell have (1 + nneighbours) * n_points * n_points interactions.
        // The + comes from the fact that we also need to consider self interactions for each cell.
        entries += (1 + n_neighbours) * n_points * n_points;
    }

    // Now initiate the aij arrays for the sparse matrix.

    let rows = Arc::new(Mutex::new(Vec::<usize>::with_capacity(entries)));
    let cols = Arc::new(Mutex::new(Vec::<usize>::with_capacity(entries)));
    let data = Arc::new(Mutex::new(Vec::<T>::with_capacity(entries)));

    let local_grid = grid.local_grid();
    let geometry_map = local_grid.geometry_map(reference_cell, eval_points);

    // We now use a parallel iterator to go through each cell and evaluate the interactions.
    active_cells.par_iter().for_each(|&cell| {
        let cell_entity = local_grid.entity(tdim, cell).unwrap();
        let cell_global_index = cell_entity.global_index();
        let mut target_points = rlst_dynamic_array2!(T::Real, [gdim, n_points]);
        geometry_map.points(cell, target_points.data_mut());
        // We also need the corresponding indices.
        let target_indices =
            (n_points * cell_global_index..n_points * (1 + cell_global_index)).collect_vec();
        // Let's get the neighbouring cells. These are the sources.
        let mut source_cells = cell_entity
            .topology()
            .connected_entity_iter(tdim)
            .collect_vec();
        // We push the cell itself for the self interactions.
        source_cells.push(cell);
        // The following matrix is going to hold the result of the kernel calculation.
        let mut kernel_matrix = rlst_dynamic_array2!(T, [n_points, source_cells.len() * n_points]);
        // We also want the global indices of the source points.
    })
}

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

// struct SyncedGridRef<
//     'a,
//     C: Communicator,
//     T: RlstScalar + Equivalence,
//     GridImpl: ParallelGrid<C, T = T::Real>,
// > {
//     grid: &'a GridImpl::LocalGrid,
//     _marker: PhantomData<C>,
//     _marker2: PhantomData<T>,
// }

// unsafe impl<'a, C: Communicator, T: RlstScalar + Equivalence, GridImpl: ParallelGrid<C, T = T::Real>>
//     Sync for SyncedGridRef<'a, C, T, GridImpl>
// {
// }

// impl<'a, C: Communicator, T: RlstScalar + Equivalence, GridImpl: ParallelGrid<C, T = T::Real>>
//     SyncedGridRef<'a, C, T, GridImpl>
// {
//     fn new(grid: &'a GridImpl::LocalGrid) -> Self {
//         Self {
//             grid,
//             _marker: PhantomData,
//             _marker2: PhantomData,
//         }
//     }
// }

// impl<'a, C: Communicator, T: RlstScalar + Equivalence, GridImpl: ParallelGrid<C, T = T::Real>>
//     std::ops::Deref for SyncedGridRef<'a, C, T, GridImpl>
// {
//     type Target = GridImpl::LocalGrid;

//     fn deref(&self) -> &Self::Target {
//         self.grid
//     }
// }

impl<
        T: RlstScalar + Equivalence,
        K: Kernel<T = T>,
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        GridImpl: ParallelGrid<C, T = T::Real>,
        C: Communicator,
    > AsApply for NeighbourEvaluator<'_, T, K, DomainLayout, RangeLayout, GridImpl, C>
where
    GridImpl::LocalGrid: Sync,
    for<'b> <GridImpl::LocalGrid as Grid>::GeometryMap<'b>: Sync,
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
        let gdim = self.grid.geometry_dim();

        // In the new function we already made sure that eval_points is a multiple of tdim.
        let n_points = self.eval_points.len() / tdim;

        // We need the reference cell
        let reference_cell = self.grid.entity_types(tdim)[0];

        // We go through groups of target dofs in chunks of lenth n_points.
        // This corresponds to iteration in active cells since we ordered those
        // already by global index.

        let local_grid = self.grid.local_grid();
        let active_cells = self.active_cells.as_slice();
        let eval_points = self.eval_points.as_slice();
        let geometry_map = local_grid.geometry_map(reference_cell, eval_points);

        y.view_mut()
            .local_mut()
            .data_mut()
            .par_chunks_mut(n_points)
            .zip(active_cells)
            .enumerate()
            .for_each(|(chunk_index, (chunk, &active_cell_index))| {
                let cell_entity = local_grid.entity(tdim, active_cell_index).unwrap();
                let mut physical_points = rlst_dynamic_array2![T::Real, [gdim, n_points]];

                // Get the geometry map for the cell.
                geometry_map.points(active_cell_index, physical_points.data_mut());

                // We now have to iterate through all neighbouring entities of the cell.

                for other_cell in cell_entity.topology().connected_entity_iter(tdim) {
                    // Get the points of the other cell.
                }
            });

        Ok(())
    }
}
