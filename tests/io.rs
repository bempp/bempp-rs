//! Test input/output
use bempp::grid::mixed_grid::MixedGridBuilder;
use bempp::grid::shapes::regular_sphere;
use bempp::grid::single_element_grid::SingleElementGridBuilder;
use bempp::traits::grid::{Builder, GmshIO};
use bempp::traits::types::ReferenceCellType;

extern crate lapack_src;
extern crate blas_src;

#[test]
fn test_regular_sphere_gmsh_io() {
    let g = regular_sphere::<f64>(2);
    g.export_as_gmsh(String::from("_test_io_sphere.msh"));
}

#[test]
fn test_gmsh_output_quads() {
    let mut b = SingleElementGridBuilder::<3, f64>::new((ReferenceCellType::Quadrilateral, 1));
    b.add_point(0, [0.0, 0.0, 0.0]);
    b.add_point(1, [1.0, 0.0, 0.0]);
    b.add_point(2, [0.0, 1.0, 0.0]);
    b.add_point(3, [1.0, 1.0, 0.0]);
    b.add_point(4, [0.0, 0.0, 1.0]);
    b.add_point(5, [1.0, 0.0, 1.0]);
    b.add_point(6, [0.0, 1.0, 1.0]);
    b.add_point(7, [1.0, 1.0, 1.0]);
    b.add_cell(0, vec![0, 2, 1, 3]);
    b.add_cell(0, vec![0, 1, 4, 5]);
    b.add_cell(0, vec![0, 4, 2, 6]);
    b.add_cell(0, vec![1, 3, 5, 7]);
    b.add_cell(0, vec![2, 6, 3, 7]);
    b.add_cell(0, vec![4, 5, 6, 7]);
    let g = b.create_grid();
    g.export_as_gmsh(String::from("_test_io_cube.msh"));
}

#[test]
fn test_gmsh_output_mixed() {
    let mut b = MixedGridBuilder::<3, f64>::new(());
    b.add_point(0, [-1.0, 0.0, 0.0]);
    b.add_point(1, [-0.5, 0.0, 0.2]);
    b.add_point(2, [0.0, 0.0, 0.0]);
    b.add_point(3, [1.0, 0.0, 0.0]);
    b.add_point(4, [2.0, 0.0, 0.0]);
    b.add_point(
        5,
        [
            -std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
        ],
    );
    b.add_point(6, [0.0, 0.5, 0.0]);
    b.add_point(7, [0.0, 1.0, 0.0]);
    b.add_point(8, [1.0, 1.0, 0.0]);
    b.add_cell(0, (vec![0, 2, 7, 6, 5, 1], ReferenceCellType::Triangle, 2));
    b.add_cell(1, (vec![2, 3, 7, 8], ReferenceCellType::Quadrilateral, 1));
    b.add_cell(2, (vec![3, 4, 8], ReferenceCellType::Triangle, 1));

    let g = b.create_grid();
    g.export_as_gmsh(String::from("_test_io_mixed.msh"));
}
