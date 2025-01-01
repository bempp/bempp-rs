//! Various helper functions to support evaluators.

use itertools::izip;
use mpi::traits::{Communicator, Equivalence};
use ndelement::traits::FiniteElement;
use ndgrid::{
    traits::{Entity, GeometryMap, Grid},
    types::Ownership,
};
use rlst::{
    rlst_array_from_slice2, rlst_dynamic_array1, rlst_dynamic_array2, rlst_dynamic_array3,
    rlst_dynamic_array4, DefaultIterator, DistributedCsrMatrix, IndexLayout, RawAccess,
    RawAccessMut, RlstScalar,
};

use crate::function::FunctionSpaceTrait;

/// Create a linear operator from the map of a basis to points.
pub fn basis_to_point_map<
    'a,
    C: Communicator + 'a,
    DomainLayout: IndexLayout<Comm = C>,
    RangeLayout: IndexLayout<Comm = C>,
    T: RlstScalar + Equivalence,
    Space: FunctionSpaceTrait<T = T>,
>(
    function_space: &Space,
    domain_layout: &'a DomainLayout,
    range_layout: &'a RangeLayout,
    quadrature_points: &[T::Real],
    quadrature_weights: &[T::Real],
    return_transpose: bool,
) -> DistributedCsrMatrix<'a, DomainLayout, RangeLayout, T, C>
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

    let n_domain_dofs = domain_layout.number_of_global_indices();
    let n_range_dofs = range_layout.number_of_global_indices();

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
            for global_dof in global_dofs.iter() {
                rows.push(n_quadrature_points * cell.global_index() + qindex);
                cols.push(*global_dof);
                data.push(T::from_real(jdet * *qweight));
            }
        }
    }

    if return_transpose {
        DistributedCsrMatrix::from_aij(
            &domain_layout,
            &range_layout,
            &cols,
            &rows,
            &data,
            domain_layout.comm(),
        )
    } else {
        DistributedCsrMatrix::from_aij(
            &domain_layout,
            &range_layout,
            &rows,
            &cols,
            &data,
            domain_layout.comm(),
        )
    }
}
