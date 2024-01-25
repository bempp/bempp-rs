use approx::*;
use bempp_bem::assembly::batched;
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::create_element;
use bempp_grid::shapes::regular_sphere;
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_quadrature::simplex_rules::simplex_rule;
use bempp_traits::bem::{DofMap, FunctionSpace};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, ElementFamily};
use bempp_traits::grid::{Geometry, Grid, Topology};
use bempp_traits::kernel::Kernel;
use bempp_traits::types::EvalType;
use rlst_dense::{
    rlst_dynamic_array2,
    traits::{RandomAccessByRef, RandomAccessMut, RawAccess, RawAccessMut, Shape},
};

#[test]
fn test_fmm_prototype_dp0_dp0() {
    #[cfg(debug_assertions)]
    let grid = regular_sphere(0);
    #[cfg(not(debug_assertions))]
    let grid = regular_sphere(2);

    let element = create_element(
        ElementFamily::Lagrange,
        ReferenceCellType::Triangle,
        0,
        Continuity::Discontinuous,
    );
    let space = SerialFunctionSpace::new(&grid, &element);

    let ndofs = space.dofmap().global_size();
    let kernel = Laplace3dKernel::new();

    let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    batched::assemble::<128>(&mut matrix, &kernel, &space, &space);

    let mut matrix2 = rlst_dynamic_array2!(f64, [ndofs, ndofs]);

    //let colouring = space.compute_cell_colouring();

    const NPTS: usize = 16;

    let qrule = simplex_rule(ReferenceCellType::Triangle, NPTS).unwrap();
    let mut qpoints = rlst_dynamic_array2!(f64, [NPTS, 2]);
    for i in 0..NPTS {
        for j in 0..2 {
            *qpoints.get_mut([i, j]).unwrap() = qrule.points[2 * i + j];
        }
    }

    batched::assemble_singular_into_dense::<4, 128>(&mut matrix2, &kernel, &space, &space);
    let correction =
        batched::assemble_singular_correction_into_csr::<NPTS, NPTS, 128>(&kernel, &space, &space);

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

    let mut k = rlst_dynamic_array2!(
        f64,
        [
            NPTS * grid.topology().entity_count(grid.topology().dim()),
            NPTS * grid.topology().entity_count(grid.topology().dim())
        ]
    );

    kernel.assemble_st(
        EvalType::Value,
        all_points.data(),
        all_points.data(),
        k.data_mut(),
    );

    let p_t = batched::assemble_transpose_basis_to_quadrature_into_csr::<NPTS, 128>(&space);
    let p = batched::assemble_basis_to_quadrature_into_csr::<NPTS, 128>(&space);

    let mut temp = rlst_dynamic_array2!(f64, [p_t.shape()[0], k.shape()[1]]);
    // temp = p_t @ k
    let mut row = 0;
    for (i, j) in p_t.indices().iter().enumerate() {
        while p_t.indptr()[row + 1] <= i {
            row += 1;
        }
        for col in 0..k.shape()[1] {
            *temp.get_mut([row, col]).unwrap() += p_t.data()[i] * *k.get([*j, col]).unwrap();
        }
    }

    // matrix2 += temp @ p
    let mut j = 0;
    for (i, col) in p.indices().iter().enumerate() {
        while p.indptr()[j + 1] <= i {
            j += 1;
        }
        for row in 0..temp.shape()[0] {
            *matrix2.get_mut([row, *col]).unwrap() += *temp.get([row, j]).unwrap() * p.data()[i];
        }
    }

    // TODO: add same cell to correction

    // matrix2 -= correction
    let mut row = 0;
    for (i, j) in correction.indices().iter().enumerate() {
        while correction.indptr()[row + 1] <= i {
            row += 1;
        }
        *matrix2.get_mut([row, *j]).unwrap() -= correction.data()[i];
    }

    for i in 0..ndofs {
        for j in 0..ndofs {
            assert_relative_eq!(
                *matrix.get([i, j]).unwrap(),
                *matrix2.get([i, j]).unwrap(),
                epsilon = 1e-8
            );
        }
    }
}
