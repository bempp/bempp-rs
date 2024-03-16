use approx::*;
use bempp_bem::assembly::batched::BatchedAssembler;
use bempp_bem::assembly::{batched, fmm_tools};
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::lagrange;
use bempp_field::types::FftFieldTranslationKiFmm;
use bempp_fmm::types::KiFmmBuilderSingleNode;
use bempp_grid::shapes::regular_sphere;
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::element::Continuity;
use bempp_traits::fmm::Fmm;
use bempp_traits::grid::GridType;
use bempp_traits::kernel::Kernel;
use bempp_traits::tree::FmmTree;
#[cfg(not(debug_assertions))]
use bempp_traits::tree::Tree;
use bempp_traits::types::EvalType;
use bempp_traits::types::ReferenceCellType;
use rand::prelude::*;
use rlst::{
    empty_array, rlst_dynamic_array2, MultIntoResize, RandomAccessByRef, RandomAccessMut,
    RawAccess, RawAccessMut,
};

extern crate blas_src;
extern crate lapack_src;

fn fmm_prototype<TestGrid: GridType<T = f64> + Sync, TrialGrid: GridType<T = f64> + Sync>(
    trial_space: &SerialFunctionSpace<f64, TrialGrid>,
    test_space: &SerialFunctionSpace<f64, TestGrid>,
) {
    const NPTS: usize = 16;

    let test_grid = test_space.grid();
    let trial_grid = test_space.grid();
    if std::ptr::addr_of!(*test_grid) as usize != std::ptr::addr_of!(*trial_grid) as usize {
        panic!("Assembly on different grids not yet supported");
    }

    let grid = trial_space.grid();

    let test_ndofs = test_space.global_size();
    let trial_ndofs = trial_space.global_size();
    let nqpts = NPTS * grid.number_of_cells();
    let kernel = Laplace3dKernel::new();

    // Compute dense
    let mut matrix = rlst_dynamic_array2!(f64, [test_ndofs, trial_ndofs]);
    let a = batched::LaplaceSingleLayerAssembler::default();
    a.assemble_into_dense::<128, TestGrid, TrialGrid>(&mut matrix, trial_space, test_space);

    // Compute using FMM method
    let all_points = fmm_tools::get_all_quadrature_points::<NPTS, f64, TrialGrid>(grid);

    // k is the matrix that FMM will give us
    let mut k = rlst_dynamic_array2!(f64, [nqpts, nqpts]);
    kernel.assemble_st(
        EvalType::Value,
        all_points.data(),
        all_points.data(),
        k.data_mut(),
    );

    let mut p_t = rlst_dynamic_array2!(f64, [test_ndofs, nqpts]);
    fmm_tools::transpose_basis_to_quadrature_into_dense::<NPTS, 128, f64, f64, TestGrid>(
        &mut p_t, test_space,
    );

    let mut p = rlst_dynamic_array2!(f64, [nqpts, trial_ndofs]);
    fmm_tools::basis_to_quadrature_into_dense::<NPTS, 128, f64, f64, TrialGrid>(
        &mut p,
        trial_space,
    );

    // matrix 2 = p_t @ k @ p - c + singular
    let mut matrix2 = rlst_dynamic_array2!(f64, [test_ndofs, trial_ndofs]);

    // matrix 2 = singular
    a.assemble_singular_into_dense::<4, 128, TestGrid, TrialGrid>(
        &mut matrix2,
        trial_space,
        test_space,
    );

    let mut correction = rlst_dynamic_array2!(f64, [test_ndofs, trial_ndofs]);
    a.assemble_singular_correction_into_dense::<NPTS, NPTS, 128, TestGrid, TrialGrid>(
        &mut correction,
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

#[cfg(not(debug_assertions))]
fn fmm_matvec<TrialGrid: GridType<T = f64> + Sync, TestGrid: GridType<T = f64> + Sync>(
    trial_space: &SerialFunctionSpace<f64, TrialGrid>,
    test_space: &SerialFunctionSpace<f64, TestGrid>,
) {
    const NPTS: usize = 16;

    let test_grid = test_space.grid();
    let trial_grid = test_space.grid();
    if std::ptr::addr_of!(*test_grid) as usize != std::ptr::addr_of!(*trial_grid) as usize {
        panic!("Assembly on different grids not yet supported");
    }

    let grid = trial_space.grid();

    let test_ndofs = test_space.global_size();
    let trial_ndofs = trial_space.global_size();
    let nqpts = NPTS * grid.number_of_cells();

    // Compute dense
    let mut matrix = rlst_dynamic_array2!(f64, [test_ndofs, trial_ndofs]);
    let a = batched::LaplaceSingleLayerAssembler::default();
    a.assemble_into_dense::<128, TestGrid, TrialGrid>(&mut matrix, trial_space, test_space);

    // Compute using FMM method
    let all_points = fmm_tools::get_all_quadrature_points::<NPTS, f64, TrialGrid>(grid);

    // FMM parameters
    let expansion_order = 6;
    let n_crit = Some(150);
    let sparse = true;

    let p_t = fmm_tools::transpose_basis_to_quadrature_into_csr::<NPTS, 128, f64, f64, TestGrid>(
        test_space,
    );
    let p = fmm_tools::basis_to_quadrature_into_csr::<NPTS, 128, f64, f64, TrialGrid>(trial_space);
    let singular =
        a.assemble_singular_into_csr::<4, 128, TestGrid, TrialGrid>(trial_space, test_space);

    let correction = a
        .assemble_singular_correction_into_csr::<NPTS, NPTS, 128, TestGrid, TrialGrid>(
            trial_space,
            test_space,
        );

    // matrix2 = p_t @ k @ p - c + singular
    let mut rng = rand::thread_rng();
    for _ in 0..10 {
        let mut vec = rlst_dynamic_array2!(f64, [trial_ndofs, 1]);
        for i in 0..trial_ndofs {
            *vec.get_mut([i, 0]).unwrap() = rng.gen();
        }
        let dense_result =
            empty_array::<f64, 2>().simple_mult_into_resize(matrix.view(), vec.view());

        let mut fmm_result = rlst_dynamic_array2!(f64, [test_ndofs, 1]);
        // (p_t @ k @ p - c + singular) @ vec
        let mut row = 0;
        for (i, (index, data)) in singular.indices().iter().zip(singular.data()).enumerate() {
            while i >= singular.indptr()[row + 1] {
                row += 1;
            }
            *fmm_result.get_mut([row, 0]).unwrap() += data * vec.get([*index, 0]).unwrap();
        }
        let mut row = 0;
        for (i, (index, data)) in correction
            .indices()
            .iter()
            .zip(correction.data())
            .enumerate()
        {
            while i >= correction.indptr()[row + 1] {
                row += 1;
            }
            *fmm_result.get_mut([row, 0]).unwrap() -= data * vec.get([*index, 0]).unwrap();
        }

        let mut temp0 = rlst_dynamic_array2!(f64, [nqpts, 1]);
        let mut row = 0;
        for (i, (index, data)) in p.indices().iter().zip(p.data()).enumerate() {
            while i >= p.indptr()[row + 1] {
                row += 1;
            }
            *temp0.get_mut([row, 0]).unwrap() += data * vec.get([*index, 0]).unwrap();
        }

        let fmm = KiFmmBuilderSingleNode::new()
            .tree(&all_points, &all_points, n_crit, sparse)
            .unwrap()
            .parameters(
                &temp0,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslationKiFmm::new(),
            )
            .unwrap()
            .build()
            .unwrap();

        fmm.evaluate();

        let mut temp1 = rlst_dynamic_array2!(f64, [nqpts, 1]);
        let indices = &fmm.tree().target_tree().all_global_indices().unwrap();

        for (i, j) in indices.iter().enumerate() {
            *temp1.get_mut([*j, 0]).unwrap() = fmm.potentials[i];
        }

        let mut row = 0;
        for (i, (index, data)) in p_t.indices().iter().zip(p_t.data()).enumerate() {
            while i >= p_t.indptr()[row + 1] {
                row += 1;
            }
            *fmm_result.get_mut([row, 0]).unwrap() += data * temp1.get([*index, 0]).unwrap();
        }

        for i in 0..test_ndofs {
            assert_relative_eq!(
                *dense_result.get([i, 0]).unwrap(),
                *fmm_result.get([i, 0]).unwrap(),
                epsilon = 1e-5
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

    let element = lagrange::create(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    fmm_prototype(&space, &space);
}

#[test]
fn test_fmm_prototype_p1_p1() {
    #[cfg(debug_assertions)]
    let grid = regular_sphere(0);
    #[cfg(not(debug_assertions))]
    let grid = regular_sphere(2);

    let element = lagrange::create(ReferenceCellType::Triangle, 1, Continuity::Continuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    fmm_prototype(&space, &space);
}

#[cfg(not(debug_assertions))]
#[test]
fn test_fmm_prototype_dp0_p1() {
    let grid = regular_sphere(2);

    let element0 = lagrange::create(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
    let element1 = lagrange::create(ReferenceCellType::Triangle, 1, Continuity::Continuous);
    let space0 = SerialFunctionSpace::new(&grid, &element0);
    let space1 = SerialFunctionSpace::new(&grid, &element1);

    fmm_prototype(&space0, &space1);
}

#[cfg(not(debug_assertions))]
#[test]
fn test_fmm_dp0_dp0() {
    let grid = regular_sphere(2);

    let element = lagrange::create(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    fmm_matvec(&space, &space);
}

#[cfg(not(debug_assertions))]
#[test]
fn test_fmm_p1_p1() {
    let grid = regular_sphere(2);

    let element = lagrange::create(ReferenceCellType::Triangle, 1, Continuity::Continuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    fmm_matvec(&space, &space);
}

#[cfg(not(debug_assertions))]
#[test]
fn test_fmm_dp0_p1() {
    let grid = regular_sphere(2);

    let element0 = lagrange::create(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
    let element1 = lagrange::create(ReferenceCellType::Triangle, 1, Continuity::Continuous);
    let space0 = SerialFunctionSpace::new(&grid, &element0);
    let space1 = SerialFunctionSpace::new(&grid, &element1);

    fmm_matvec(&space0, &space1);
}

#[test]
fn test_fmm_result() {
    let grid = regular_sphere(2);

    const NPTS: usize = 1;

    let nqpts = NPTS * grid.number_of_cells();
    let kernel = Laplace3dKernel::new();

    let all_points = fmm_tools::get_all_quadrature_points::<NPTS, f64, _>(&grid);

    let expansion_order = 6;
    let n_crit = Some(1);
    let sparse = true;

    let mut k = rlst_dynamic_array2!(f64, [nqpts, nqpts]);
    kernel.assemble_st(
        EvalType::Value,
        all_points.data(),
        all_points.data(),
        k.data_mut(),
    );

    // let mut rng: ThreadRng = rand::thread_rng();
    let mut rng = StdRng::seed_from_u64(0);

    let mut vec = rlst_dynamic_array2!(f64, [nqpts, 1]);
    for i in 0..nqpts {
        *vec.get_mut([i, 0]).unwrap() = rng.gen();
    }
    let dense_result = empty_array::<f64, 2>().simple_mult_into_resize(k.view(), vec.view());

    let fmm = KiFmmBuilderSingleNode::new()
        .tree(&all_points, &all_points, n_crit, sparse)
        .unwrap()
        .parameters(
            &vec,
            expansion_order,
            Laplace3dKernel::new(),
            EvalType::Value,
            FftFieldTranslationKiFmm::new(),
        )
        .unwrap()
        .build()
        .unwrap();

    fmm.evaluate();

    let indices = &fmm.tree().target_tree().global_indices;
    let mut fmm_result = rlst_dynamic_array2!(f64, [nqpts, 1]);
    for (i, j) in indices.iter().enumerate() {
        *fmm_result.get_mut([*j, 0]).unwrap() = fmm.potentials[i];
    }

    for i in 0..nqpts {
        assert_relative_eq!(
            *dense_result.get([i, 0]).unwrap(),
            *fmm_result.get([i, 0]).unwrap(),
            epsilon = 1e-5
        );
    }
}
