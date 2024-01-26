use approx::*;
use bempp_bem::assembly::{batched, fmm_tools};
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::create_element;
use bempp_grid::shapes::regular_sphere;
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::bem::{DofMap, FunctionSpace};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, ElementFamily};
use bempp_traits::grid::{Grid, Topology};
use bempp_traits::kernel::Kernel;
use bempp_traits::types::EvalType;
use rlst_dense::{
    array::empty_array,
    rlst_dynamic_array2,
    traits::{MultIntoResize, RandomAccessByRef, RandomAccessMut, RawAccess, RawAccessMut},
};

fn fmm_prototype(trial_space: &SerialFunctionSpace, test_space: &SerialFunctionSpace) {
    const NPTS: usize = 16;

    if test_space.grid() != trial_space.grid() {
        panic!("Assembly on different grid not yet supported");
    }

    let grid = trial_space.grid();

    let test_ndofs = test_space.dofmap().global_size();
    let trial_ndofs = trial_space.dofmap().global_size();
    let nqpts = NPTS * grid.topology().entity_count(grid.topology().dim());
    let kernel = Laplace3dKernel::new();

    // Compute dense
    let mut matrix = rlst_dynamic_array2!(f64, [test_ndofs, trial_ndofs]);
    batched::assemble::<128>(&mut matrix, &kernel, trial_space, test_space);
    // Compute using FMM method
    let all_points = fmm_tools::get_all_quadrature_points::<NPTS>(grid);

    // k is the matrix that FMM will give us
    let mut k = rlst_dynamic_array2!(f64, [nqpts, nqpts]);
    kernel.assemble_st(
        EvalType::Value,
        all_points.data(),
        all_points.data(),
        k.data_mut(),
    );

    let mut p_t = rlst_dynamic_array2!(f64, [test_ndofs, nqpts]);
    fmm_tools::transpose_basis_to_quadrature_into_dense::<NPTS, 128>(&mut p_t, test_space);

    let mut p = rlst_dynamic_array2!(f64, [nqpts, trial_ndofs]);
    fmm_tools::basis_to_quadrature_into_dense::<NPTS, 128>(&mut p, trial_space);

    // matrix 2 = p_t @ k @ p - c + singular
    let mut matrix2 = rlst_dynamic_array2!(f64, [test_ndofs, trial_ndofs]);

    // matrix 2 = singular
    batched::assemble_singular_into_dense::<4, 128>(
        &mut matrix2,
        &kernel,
        trial_space,
        test_space,
    );

    let mut correction = rlst_dynamic_array2!(f64, [test_ndofs, trial_ndofs]);
    batched::assemble_singular_correction_into_dense::<NPTS, NPTS, 128>(
        &mut correction,
        &kernel,
        trial_space,
        test_space,
    );

    let temp = empty_array::<f64, 2>()
        .simple_mult_into_resize(empty_array::<f64, 2>().simple_mult_into_resize(p_t, k), p);
    for j in 0..trial_ndofs {
        for i in 0..test_ndofs {
            *matrix2.get_mut([i, j]).unwrap() +=
                *temp.get([i, j]).unwrap() - *correction.get([i, j]).unwrap();
        }
    }

    // Check two matrices are equal
    for i in 0..test_ndofs {
        for j in 0..trial_ndofs {
            assert_relative_eq!(
                *matrix.get([i, j]).unwrap(),
                *matrix2.get([i, j]).unwrap(),
                epsilon = 1e-8
            );
        }
    }
}

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

    fmm_prototype(&space, &space);
}

#[test]
fn test_fmm_prototype_p1_p1() {
    #[cfg(debug_assertions)]
    let grid = regular_sphere(0);
    #[cfg(not(debug_assertions))]
    let grid = regular_sphere(2);

    let element = create_element(
        ElementFamily::Lagrange,
        ReferenceCellType::Triangle,
        1,
        Continuity::Continuous,
    );
    let space = SerialFunctionSpace::new(&grid, &element);

    fmm_prototype(&space, &space);
}

#[test]
fn test_fmm_prototype_dp0_p1() {
    #[cfg(debug_assertions)]
    let grid = regular_sphere(0);
    #[cfg(not(debug_assertions))]
    let grid = regular_sphere(2);

    let element0 = create_element(
        ElementFamily::Lagrange,
        ReferenceCellType::Triangle,
        0,
        Continuity::Discontinuous,
    );
    let element1 = create_element(
        ElementFamily::Lagrange,
        ReferenceCellType::Triangle,
        1,
        Continuity::Continuous,
    );
    let space0 = SerialFunctionSpace::new(&grid, &element0);
    let space1 = SerialFunctionSpace::new(&grid, &element1);

    fmm_prototype(&space0, &space1);
}
