use approx::*;
use bempp_field::types::FftFieldTranslationKiFmm;
use bempp_fmm::charge::build_charge_dict;
use bempp_fmm::types::{FmmDataUniform, KiFmmLinear};
use bempp_traits::fmm::FmmLoop;
use bempp_tree::types::single_node::SingleNodeTree;
use rand::prelude::*;
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
use bempp_traits::tree::Tree;
use rlst_dense::rlst_array_from_slice2;
use rlst_dense::{
    array::empty_array,
    rlst_dynamic_array1,
    rlst_dynamic_array2,
    traits::{MultIntoResize, RandomAccessByRef, RandomAccessMut, RawAccess, RawAccessMut, Shape},
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
    batched::assemble_singular_into_dense::<4, 128>(&mut matrix2, &kernel, trial_space, test_space);

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

fn fmm_prototype_matvec(trial_space: &SerialFunctionSpace, test_space: &SerialFunctionSpace) {
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
    let all_points_t = fmm_tools::get_all_quadrature_points::<NPTS>(grid);
    // println!("shape {:?}", all_points.shape());

    let mut all_points = rlst_dynamic_array2!(f64, [all_points_t.shape()[1], all_points_t.shape()[0]]);
    all_points.view_mut().fill_from(all_points_t.transpose());

    // k is the matrix that FMM will give us
    let mut k = rlst_dynamic_array2!(f64, [nqpts, nqpts]);
    kernel.assemble_st(
        EvalType::Value,
        all_points.data(),
        all_points.data(),
        k.data_mut(),
    );

    /// FMM
    let order = 6;
    let alpha_inner = 1.05;
    let alpha_outer = 2.95;
    let depth = 3;
    let global_idxs: Vec<_> = (0..nqpts).collect();


    ///////

    let p_t = fmm_tools::transpose_basis_to_quadrature_into_csr::<NPTS, 128>(test_space);
    let p = fmm_tools::basis_to_quadrature_into_csr::<NPTS, 128>(trial_space);
    let singular = batched::assemble_singular_into_csr::<4, 128>(&kernel, trial_space, test_space);

    let mut correction = batched::assemble_singular_correction_into_csr::<NPTS, NPTS, 128>(
        &kernel,
        trial_space,
        test_space,
    );

    // matrix2 = p_t @ k @ p - c + singular
    let mut rng = rand::thread_rng();
    for _ in 0..10 {

        let mut vec = rlst_dynamic_array2!(f64, [trial_ndofs, 1]);
        for i in 0..trial_ndofs {
            *vec.get_mut([i, 0]).unwrap() = 1.0; //rng.gen();
        }
        let dense_result = empty_array::<f64, 2>().simple_mult_into_resize(
            matrix.view(), vec.view());


        let mut fmm_result = rlst_dynamic_array2!(f64, [test_ndofs, 1]);
        // (p_t @ k @ p - c + singular) @ vec
        let mut row = 0;
        for (i, (index, data)) in singular.indices().iter().zip(singular.data()).enumerate() {
            while i >= singular.indptr()[row + 1] { row += 1; }
            *fmm_result.get_mut([row, 0]).unwrap() += data * vec.get([*index, 0]).unwrap();
        }
        let mut row = 0;
        for (i, (index, data)) in correction.indices().iter().zip(correction.data()).enumerate() {
            while i >= correction.indptr()[row + 1] { row += 1; }
            *fmm_result.get_mut([row, 0]).unwrap() -= data * vec.get([*index, 0]).unwrap();
        }

        let mut temp0 = rlst_dynamic_array2!(f64, [nqpts, 1]);
        let mut row = 0;
        for (i, (index, data)) in p.indices().iter().zip(p.data()).enumerate() {
            while i >= p.indptr()[row + 1] { row += 1; }
            *temp0.get_mut([row, 0]).unwrap() += data * vec.get([*index, 0]).unwrap();
        }

        ////
        let tree = SingleNodeTree::new(all_points.data(), false, None, Some(depth), &global_idxs, true);

        let m2l_data =
            FftFieldTranslationKiFmm::new(kernel.clone(), order, *tree.get_domain(), alpha_inner);
        let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel.clone(), tree, m2l_data);
        // let charges = vec![1f64; nqpts];
        let charge_dict = build_charge_dict(&global_idxs, &temp0.data());
        let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();
        datatree.run(false);
        ///

        // let mut temp1 = empty_array::<f64, 2>().simple_mult_into_resize(k.view(), temp0);
        let temp1: rlst::Array<f64, rlst_dense::base_array::BaseArray<f64, rlst_dense::data_container::SliceContainer<'_, f64>, 2>, 2> = rlst_array_from_slice2!(f64, datatree.potentials.as_slice(), [nqpts, 1]);
        let mut row = 0;
        for (i, (index, data)) in p_t.indices().iter().zip(p_t.data()).enumerate() {
            while i >= p_t.indptr()[row + 1] { row += 1; }
            *fmm_result.get_mut([row, 0]).unwrap() += data * temp1.get([*index, 0]).unwrap();
        }

        for i in 0..test_ndofs {
            assert_relative_eq!(
                *dense_result.get([i, 0]).unwrap(),
                *fmm_result.get([i, 0]).unwrap(),
                epsilon = 1e-6
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
    fmm_prototype_matvec(&space, &space);
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
