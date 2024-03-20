use approx::*;
use bempp_bem::assembly::{batched, batched::BatchedAssembler};
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::LagrangeElementFamily;
use bempp_grid::{
    mixed_grid::SerialMixedGridBuilder, shapes::regular_sphere,
    single_element_grid::SerialSingleElementGridBuilder,
};
use bempp_traits::{
    bem::FunctionSpace, element::Continuity, grid::Builder, types::ReferenceCellType,
};
use cauchy::c64;
use rlst::{rlst_dynamic_array2, RandomAccessByRef};

extern crate blas_src;
extern crate lapack_src;

// TODO: paste this for multiple operators
#[test]
fn compare_mixed_to_single_element_dp0() {
    let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);

    let mut b = SerialMixedGridBuilder::<3, f64>::new(());
    b.add_point(0, [0.0, 0.0, 0.0]);
    b.add_point(1, [0.5, 0.0, 0.0]);
    b.add_point(2, [1.0, 0.0, 0.0]);
    b.add_point(3, [0.0, 0.5, 0.0]);
    b.add_point(4, [0.5, 0.5, 0.0]);
    b.add_point(5, [1.0, 0.5, 0.0]);
    b.add_point(6, [0.0, 1.0, 0.0]);
    b.add_point(7, [0.5, 1.0, 0.0]);
    b.add_point(8, [1.0, 1.0, 0.0]);
    b.add_cell(0, (vec![0, 1, 3, 4], ReferenceCellType::Quadrilateral, 1));
    b.add_cell(1, (vec![3, 4, 6, 7], ReferenceCellType::Quadrilateral, 1));
    b.add_cell(2, (vec![1, 2, 5], ReferenceCellType::Triangle, 1));
    b.add_cell(3, (vec![2, 5, 4], ReferenceCellType::Triangle, 1));
    b.add_cell(4, (vec![4, 5, 8], ReferenceCellType::Triangle, 1));
    b.add_cell(5, (vec![5, 8, 7], ReferenceCellType::Triangle, 1));
    let grid = b.create_grid();
    let space = SerialFunctionSpace::new(&grid, &element);
    let ndofs = space.global_size();
    let mut mixed_matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    let a = batched::LaplaceSingleLayerAssembler::<128, f64>::default();
    a.assemble_into_dense(&mut mixed_matrix, &space, &space);

    let mut b =
        SerialSingleElementGridBuilder::<3, f64>::new((ReferenceCellType::Quadrilateral, 1));
    b.add_point(0, [0.0, 0.0, 0.0]);
    b.add_point(1, [0.5, 0.0, 0.0]);
    b.add_point(3, [0.0, 0.5, 0.0]);
    b.add_point(4, [0.5, 0.5, 0.0]);
    b.add_point(6, [0.0, 1.0, 0.0]);
    b.add_point(7, [0.5, 1.0, 0.0]);
    b.add_point(8, [1.0, 1.0, 0.0]);
    b.add_cell(0, vec![0, 1, 3, 4]);
    b.add_cell(1, vec![3, 4, 6, 7]);
    let grid = b.create_grid();
    let space = SerialFunctionSpace::new(&grid, &element);
    let ndofs = space.global_size();
    let mut quad_matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    let a = batched::LaplaceSingleLayerAssembler::<128, f64>::default();
    a.assemble_into_dense(&mut quad_matrix, &space, &space);

    let mut b = SerialSingleElementGridBuilder::<3, f64>::new((ReferenceCellType::Triangle, 1));
    b.add_point(1, [0.5, 0.0, 0.0]);
    b.add_point(2, [1.0, 0.0, 0.0]);
    b.add_point(4, [0.5, 0.5, 0.0]);
    b.add_point(5, [1.0, 0.5, 0.0]);
    b.add_point(7, [0.5, 1.0, 0.0]);
    b.add_point(8, [1.0, 1.0, 0.0]);
    b.add_cell(2, vec![1, 2, 5]);
    b.add_cell(3, vec![2, 5, 4]);
    b.add_cell(4, vec![4, 5, 8]);
    b.add_cell(5, vec![5, 8, 7]);
    let grid = b.create_grid();
    let space = SerialFunctionSpace::new(&grid, &element);
    let ndofs = space.global_size();
    let mut tri_matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    let a = batched::LaplaceSingleLayerAssembler::<128, f64>::default();
    a.assemble_into_dense(&mut tri_matrix, &space, &space);

    for (i0, i1) in (0..2).enumerate() {
        for (j0, j1) in (0..2).enumerate() {
            assert_relative_eq!(
                *quad_matrix.get([i0, j0]).unwrap(),
                *mixed_matrix.get([i1, j1]).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    for (i0, i1) in (2..6).enumerate() {
        for (j0, j1) in (2..6).enumerate() {
            assert_relative_eq!(
                *tri_matrix.get([i0, j0]).unwrap(),
                *mixed_matrix.get([i1, j1]).unwrap(),
                epsilon = 1e-10
            );
        }
    }
}

// TODO: paste this for multiple operators
#[test]
fn compare_mixed_to_single_element_dp1() {
    let element = LagrangeElementFamily::<f64>::new(1, Continuity::Discontinuous);

    let mut b = SerialMixedGridBuilder::<3, f64>::new(());
    b.add_point(0, [0.0, 0.0, 0.0]);
    b.add_point(1, [0.5, 0.0, 0.0]);
    b.add_point(2, [1.0, 0.0, 0.0]);
    b.add_point(3, [0.0, 0.5, 0.0]);
    b.add_point(4, [0.5, 0.5, 0.0]);
    b.add_point(5, [1.0, 0.5, 0.0]);
    b.add_point(6, [0.0, 1.0, 0.0]);
    b.add_point(7, [0.5, 1.0, 0.0]);
    b.add_point(8, [1.0, 1.0, 0.0]);
    b.add_cell(0, (vec![0, 1, 3, 4], ReferenceCellType::Quadrilateral, 1));
    b.add_cell(1, (vec![3, 4, 6, 7], ReferenceCellType::Quadrilateral, 1));
    b.add_cell(2, (vec![1, 2, 5], ReferenceCellType::Triangle, 1));
    b.add_cell(3, (vec![2, 5, 4], ReferenceCellType::Triangle, 1));
    b.add_cell(4, (vec![4, 5, 8], ReferenceCellType::Triangle, 1));
    b.add_cell(5, (vec![5, 8, 7], ReferenceCellType::Triangle, 1));
    let grid = b.create_grid();
    let space = SerialFunctionSpace::new(&grid, &element);
    let ndofs = space.global_size();
    let mut mixed_matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    let a = batched::LaplaceSingleLayerAssembler::<128, f64>::default();
    a.assemble_into_dense(&mut mixed_matrix, &space, &space);

    let mut b =
        SerialSingleElementGridBuilder::<3, f64>::new((ReferenceCellType::Quadrilateral, 1));
    b.add_point(0, [0.0, 0.0, 0.0]);
    b.add_point(1, [0.5, 0.0, 0.0]);
    b.add_point(3, [0.0, 0.5, 0.0]);
    b.add_point(4, [0.5, 0.5, 0.0]);
    b.add_point(6, [0.0, 1.0, 0.0]);
    b.add_point(7, [0.5, 1.0, 0.0]);
    b.add_point(8, [1.0, 1.0, 0.0]);
    b.add_cell(0, vec![0, 1, 3, 4]);
    b.add_cell(1, vec![3, 4, 6, 7]);
    let grid = b.create_grid();
    let space = SerialFunctionSpace::new(&grid, &element);
    let ndofs = space.global_size();
    let mut quad_matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    let a = batched::LaplaceSingleLayerAssembler::<128, f64>::default();
    a.assemble_into_dense(&mut quad_matrix, &space, &space);

    let mut b = SerialSingleElementGridBuilder::<3, f64>::new((ReferenceCellType::Triangle, 1));
    b.add_point(1, [0.5, 0.0, 0.0]);
    b.add_point(2, [1.0, 0.0, 0.0]);
    b.add_point(4, [0.5, 0.5, 0.0]);
    b.add_point(5, [1.0, 0.5, 0.0]);
    b.add_point(7, [0.5, 1.0, 0.0]);
    b.add_point(8, [1.0, 1.0, 0.0]);
    b.add_cell(2, vec![1, 2, 5]);
    b.add_cell(3, vec![2, 5, 4]);
    b.add_cell(4, vec![4, 5, 8]);
    b.add_cell(5, vec![5, 8, 7]);
    let grid = b.create_grid();
    let space = SerialFunctionSpace::new(&grid, &element);
    let ndofs = space.global_size();
    let mut tri_matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    let a = batched::LaplaceSingleLayerAssembler::<128, f64>::default();
    a.assemble_into_dense(&mut tri_matrix, &space, &space);

    for (i0, i1) in (0..8).enumerate() {
        for (j0, j1) in (0..8).enumerate() {
            assert_relative_eq!(
                *quad_matrix.get([i0, j0]).unwrap(),
                *mixed_matrix.get([i1, j1]).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    for (i0, i1) in (8..20).enumerate() {
        for (j0, j1) in (8..20).enumerate() {
            assert_relative_eq!(
                *tri_matrix.get([i0, j0]).unwrap(),
                *mixed_matrix.get([i1, j1]).unwrap(),
                epsilon = 1e-10
            );
        }
    }
}
