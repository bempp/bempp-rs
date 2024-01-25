use crate::function_space::SerialFunctionSpace;
use bempp_quadrature::simplex_rules::simplex_rule;
use bempp_traits::bem::{FunctionSpace, DofMap};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};
use crate::assembly::common::SparseMatrixData;
use rlst_dense::{
    array::Array,
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_dynamic_array2,
    rlst_dynamic_array4,
    traits::{
        RandomAccessByRef, RandomAccessMut, Shape
    },
};
use rlst_sparse::sparse::csr_mat::CsrMatrix;

// TODO: use T not f64
pub fn get_all_quadrature_points<const NPTS: usize>(
    space: &SerialFunctionSpace,
) -> Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> {
    let grid = space.grid();
    let qrule = simplex_rule(ReferenceCellType::Triangle, NPTS).unwrap();
    let mut qpoints = rlst_dynamic_array2!(f64, [NPTS, 2]);
    for i in 0..NPTS {
        for j in 0..2 {
            *qpoints.get_mut([i, j]).unwrap() = qrule.points[2 * i + j];
        }
    }

    let evaluator = grid
        .geometry()
        .get_evaluator(grid.geometry().element(0), &qpoints);

    let mut all_points = rlst_dynamic_array2!(
        f64,
        [
            NPTS * grid.topology().entity_count(grid.topology().dim()),
            grid.geometry().dim()
        ]
    );
    let mut mapped_pts = rlst_dynamic_array2!(f64, [NPTS, grid.geometry().dim()]);

    for cell in 0..grid.topology().entity_count(grid.topology().dim()) {
        let cell_gindex = grid.geometry().index_map()[cell];
        evaluator.compute_points(cell_gindex, &mut mapped_pts);
        for i in 0..NPTS {
            for j in 0..grid.geometry().dim() {
                *all_points.get_mut([cell * NPTS + i, j]).unwrap() =
                    *mapped_pts.get([i, j]).unwrap();
            }
        }
    }
    all_points
}

pub fn basis_to_quadrature_into_dense<const NPTS: usize, const BLOCKSIZE: usize>(
    output: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    space: &SerialFunctionSpace,
) {
    let sparse_matrix = basis_to_quadrature::<NPTS, BLOCKSIZE>(output.shape(), space);
    let data = sparse_matrix.data;
    let rows = sparse_matrix.rows;
    let cols = sparse_matrix.cols;
    for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
        *output.get_mut([*i, *j]).unwrap() += *value;
    }
}

pub fn basis_to_quadrature_into_csr<const NPTS: usize, const BLOCKSIZE: usize>(
    space: &SerialFunctionSpace,
) -> CsrMatrix<f64> {
    let grid = space.grid();
    let ncells = grid.topology().entity_count(grid.topology().dim());
    let shape = [ncells * NPTS, space.dofmap().global_size()];
    let sparse_matrix = basis_to_quadrature::<NPTS, BLOCKSIZE>(shape, space);

    CsrMatrix::<f64>::from_aij(
        sparse_matrix.shape,
        &sparse_matrix.rows,
        &sparse_matrix.cols,
        &sparse_matrix.data,
    )
    .unwrap()
}


pub fn transpose_basis_to_quadrature_into_dense<
    const NPTS: usize,
    const BLOCKSIZE: usize,
>(
    output: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    space: &SerialFunctionSpace,
) {
    let sparse_matrix = basis_to_quadrature::<NPTS, BLOCKSIZE>(output.shape(), space);
    let data = sparse_matrix.data;
    let rows = sparse_matrix.rows;
    let cols = sparse_matrix.cols;
    for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
        *output.get_mut([*j, *i]).unwrap() += *value;
    }
}

pub fn transpose_basis_to_quadrature_into_csr<
    const NPTS: usize,
    const BLOCKSIZE: usize,
>(
    space: &SerialFunctionSpace,
) -> CsrMatrix<f64> {
    let grid = space.grid();
    let ncells = grid.topology().entity_count(grid.topology().dim());
    let shape = [ncells * NPTS, space.dofmap().global_size()];
    let sparse_matrix = basis_to_quadrature::<NPTS, BLOCKSIZE>(shape, space);

    CsrMatrix::<f64>::from_aij(
        [space.dofmap().global_size(), ncells * NPTS],
        &sparse_matrix.cols,
        &sparse_matrix.rows,
        &sparse_matrix.data,
    )
    .unwrap()
}

fn basis_to_quadrature<const NPTS: usize, const BLOCKSIZE: usize>(
    shape: [usize; 2],
    space: &SerialFunctionSpace,
) -> SparseMatrixData<f64> {
    if !space.is_serial() {
        panic!("Dense assembly can only be used for function spaces stored in serial");
    }
    let grid = space.grid();
    let ncells = grid.topology().entity_count(grid.topology().dim());
    if shape[0] != ncells * NPTS || shape[1] != space.dofmap().global_size() {
        panic!("Matrix has wrong shape");
    }

    // TODO: pass cell types into this function
    let qrule = simplex_rule(ReferenceCellType::Triangle, NPTS).unwrap();
    let mut qpoints = rlst_dynamic_array2!(f64, [NPTS, 2]);
    for i in 0..NPTS {
        for j in 0..2 {
            *qpoints.get_mut([i, j]).unwrap() = qrule.points[2 * i + j];
        }
    }
    let qweights = qrule.weights;

    let mut table = rlst_dynamic_array4!(f64, space.element().tabulate_array_shape(0, NPTS));
    space.element().tabulate(&qpoints, 0, &mut table);

    let mut output =
        SparseMatrixData::<f64>::new_known_size(shape, ncells * space.element().dim() * NPTS);
    debug_assert!(qpoints.shape()[0] == NPTS);

    let element = grid.geometry().element(0);

    let evaluator = grid.geometry().get_evaluator(element, &qpoints);

    let mut jdets = vec![0.0; NPTS];
    let mut normals = rlst_dynamic_array2!(f64, [NPTS, 3]);

    // TODO: batch this?
    for cell in 0..ncells {
        let cell_tindex = grid.topology().index_map()[cell];
        let cell_gindex = grid.geometry().index_map()[cell];
        evaluator.compute_normals_and_jacobian_determinants(cell_gindex, &mut normals, &mut jdets);
        for (qindex, (w, jdet)) in qweights.iter().zip(&jdets).enumerate() {
            for (i, dof) in space
                .dofmap()
                .cell_dofs(cell_tindex)
                .unwrap()
                .iter()
                .enumerate()
            {
                output.rows.push(cell * NPTS + qindex);
                output.cols.push(*dof);
                output
                    .data
                    .push(jdet * w * table.get([0, qindex, i, 0]).unwrap());
            }
        }
    }
    output
}

