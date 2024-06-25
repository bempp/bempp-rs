//! FMM tools
use crate::assembly::common::SparseMatrixData;
use crate::function::SerialFunctionSpace;
use crate::grid::common::compute_dets;
use crate::quadrature::simplex_rules::simplex_rule;
use crate::traits::function::FunctionSpace;
use crate::traits::grid::{GridType, ReferenceMapType};
use ndelement::traits::FiniteElement;
use ndelement::types::ReferenceCellType;
use rlst::CsrMatrix;
use rlst::{
    rlst_dynamic_array2, rlst_dynamic_array4, Array, BaseArray, RandomAccessByRef, RandomAccessMut,
    RawAccess, RlstScalar, Shape, VectorContainer,
};

/// Generate an array of all the quadrature points
pub fn get_all_quadrature_points<T: RlstScalar<Real = T>, Grid: GridType<T = T>>(
    npts: usize,
    grid: &Grid,
) -> Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> {
    let qrule = simplex_rule(ReferenceCellType::Triangle, npts).unwrap();
    let mut qpoints = rlst_dynamic_array2!(T, [npts, 2]);
    for i in 0..npts {
        for j in 0..2 {
            *qpoints.get_mut([i, j]).unwrap() =
                num::cast::<f64, T>(qrule.points[2 * i + j]).unwrap();
        }
    }

    let evaluator = grid.reference_to_physical_map(qpoints.data());

    let mut all_points = rlst_dynamic_array2!(
        T,
        [npts * grid.number_of_cells(), grid.physical_dimension()]
    );
    let mut points = vec![num::cast::<f64, T>(0.0).unwrap(); npts * grid.physical_dimension()];

    for cell in 0..grid.number_of_cells() {
        evaluator.reference_to_physical(cell, &mut points);
        for j in 0..grid.physical_dimension() {
            for i in 0..npts {
                *all_points.get_mut([cell * npts + i, j]).unwrap() = points[j * npts + i];
            }
        }
    }
    all_points
}

/// Generate a dense matrix mapping between basis functions and quadrature points
pub fn basis_to_quadrature_into_dense<
    const BLOCKSIZE: usize,
    T: RlstScalar,
    Grid: GridType<T = T::Real>,
>(
    output: &mut Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
    npts: usize,
    space: &SerialFunctionSpace<'_, T, Grid>,
) {
    let sparse_matrix = basis_to_quadrature::<BLOCKSIZE, T, Grid>(output.shape(), npts, space);
    let data = sparse_matrix.data;
    let rows = sparse_matrix.rows;
    let cols = sparse_matrix.cols;
    for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
        *output.get_mut([*i, *j]).unwrap() += *value;
    }
}

/// Generate a CSR matrix mapping between basis functions and quadrature points
pub fn basis_to_quadrature_into_csr<
    const BLOCKSIZE: usize,
    T: RlstScalar,
    Grid: GridType<T = T::Real>,
>(
    npts: usize,
    space: &SerialFunctionSpace<'_, T, Grid>,
) -> CsrMatrix<T> {
    let grid = space.grid();
    let ncells = grid.number_of_cells();
    let shape = [ncells * npts, space.global_size()];
    let sparse_matrix = basis_to_quadrature::<BLOCKSIZE, T, Grid>(shape, npts, space);

    CsrMatrix::<T>::from_aij(
        sparse_matrix.shape,
        &sparse_matrix.rows,
        &sparse_matrix.cols,
        &sparse_matrix.data,
    )
    .unwrap()
}

/// Generate a dense transpose matrix mapping between basis functions and quadrature points
pub fn transpose_basis_to_quadrature_into_dense<
    const BLOCKSIZE: usize,
    T: RlstScalar,
    Grid: GridType<T = T::Real>,
>(
    output: &mut Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
    npts: usize,
    space: &SerialFunctionSpace<'_, T, Grid>,
) {
    let shape = [output.shape()[1], output.shape()[0]];
    let sparse_matrix = basis_to_quadrature::<BLOCKSIZE, T, Grid>(shape, npts, space);
    let data = sparse_matrix.data;
    let rows = sparse_matrix.rows;
    let cols = sparse_matrix.cols;
    for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
        *output.get_mut([*j, *i]).unwrap() += *value;
    }
}

/// Generate a CSR transpose matrix mapping between basis functions and quadrature points
pub fn transpose_basis_to_quadrature_into_csr<
    const BLOCKSIZE: usize,
    T: RlstScalar,
    Grid: GridType<T = T::Real>,
>(
    npts: usize,
    space: &SerialFunctionSpace<'_, T, Grid>,
) -> CsrMatrix<T> {
    let grid = space.grid();
    let ncells = grid.number_of_cells();
    let shape = [ncells * npts, space.global_size()];
    let sparse_matrix = basis_to_quadrature::<BLOCKSIZE, T, Grid>(shape, npts, space);

    CsrMatrix::<T>::from_aij(
        [space.global_size(), ncells * npts],
        &sparse_matrix.cols,
        &sparse_matrix.rows,
        &sparse_matrix.data,
    )
    .unwrap()
}

fn basis_to_quadrature<const BLOCKSIZE: usize, T: RlstScalar, Grid: GridType<T = T::Real>>(
    shape: [usize; 2],
    npts: usize,
    space: &SerialFunctionSpace<'_, T, Grid>,
) -> SparseMatrixData<T> {
    if !space.is_serial() {
        panic!("Dense assembly can only be used for function spaces stored in serial");
    }
    let grid = space.grid();
    let ncells = grid.number_of_cells();
    if shape[0] != ncells * npts || shape[1] != space.global_size() {
        panic!("Matrix has wrong shape");
    }

    // TODO: pass cell types into this function
    let qrule = simplex_rule(ReferenceCellType::Triangle, npts).unwrap();
    let mut qpoints = rlst_dynamic_array2!(T::Real, [npts, 2]);
    for i in 0..npts {
        for j in 0..2 {
            *qpoints.get_mut([i, j]).unwrap() =
                num::cast::<f64, T::Real>(qrule.points[2 * i + j]).unwrap();
        }
    }
    let qweights = qrule
        .weights
        .iter()
        .map(|w| num::cast::<f64, T>(*w).unwrap())
        .collect::<Vec<_>>();

    let mut table = rlst_dynamic_array4!(
        T,
        space
            .element(ReferenceCellType::Triangle)
            .tabulate_array_shape(0, npts)
    );
    space
        .element(ReferenceCellType::Triangle)
        .tabulate(&qpoints, 0, &mut table);

    let mut output = SparseMatrixData::<T>::new_known_size(
        shape,
        ncells * space.element(ReferenceCellType::Triangle).dim() * npts,
    );
    debug_assert!(qpoints.shape()[0] == npts);

    let evaluator = grid.reference_to_physical_map(qpoints.data());
    let npts = qweights.len();

    let mut jacobians = vec![
        num::cast::<f64, T::Real>(0.0).unwrap();
        grid.physical_dimension() * grid.domain_dimension() * npts
    ];
    let mut jdets = vec![num::cast::<f64, T::Real>(0.0).unwrap(); npts];

    // TODO: batch this?
    for cell in 0..ncells {
        let cell_dofs = space.cell_dofs(cell).unwrap();
        evaluator.jacobian(cell, &mut jacobians);
        compute_dets(
            &jacobians,
            grid.domain_dimension(),
            grid.physical_dimension(),
            &mut jdets,
        );
        for (qindex, w) in qweights.iter().enumerate() {
            for (i, dof) in cell_dofs.iter().enumerate() {
                output.rows.push(cell * npts + qindex);
                output.cols.push(*dof);
                output.data.push(
                    num::cast::<T::Real, T>(jdets[qindex]).unwrap()
                        * *w
                        * *table.get([0, qindex, i, 0]).unwrap(),
                );
            }
        }
    }
    output
}
