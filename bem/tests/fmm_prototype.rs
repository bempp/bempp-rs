use approx::*;
use bempp_bem::assembly::batched;
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::create_element;
use bempp_grid::shapes::regular_sphere;
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::bem::{DofMap, FunctionSpace};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, ElementFamily};
use rlst_dense::{rlst_dynamic_array2, traits::RandomAccessByRef};

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

    let colouring = space.compute_cell_colouring();

    batched::assemble_nonsingular::<16, 16, 128>(
        &mut matrix2,
        &kernel,
        &space,
        &space,
        &colouring,
        &colouring,
    );

    batched::assemble_singular_into_dense::<4, 128>(&mut matrix2, &kernel, &space, &space);

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
