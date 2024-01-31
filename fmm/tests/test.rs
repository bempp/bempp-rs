use bempp_bem::assembly::{batched, fmm_tools};
use bempp_field::types::FftFieldTranslationKiFmm;
use bempp_fmm::charge::build_charge_dict;
use bempp_fmm::types::FmmDataUniform;
use bempp_fmm::types::KiFmmLinear;
use bempp_grid::shapes::regular_sphere;
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::fmm::Fmm;
use bempp_traits::fmm::FmmLoop;
use bempp_traits::grid::Grid;
use bempp_traits::grid::Topology;
use bempp_traits::kernel::Kernel;
use bempp_traits::tree::Tree;
use bempp_traits::types::EvalType;
use bempp_tree::types::single_node::SingleNodeTree;
use rand::Rng;
use rlst_dense::array::empty_array;
use rlst_dense::rlst_array_from_slice2;
use rlst_dense::rlst_dynamic_array2;
use rlst_dense::traits::MultIntoResize;
use rlst_dense::traits::RandomAccessByRef;
use rlst_dense::traits::RandomAccessMut;
use rlst_dense::traits::RawAccess;
use rlst_dense::traits::RawAccessMut;

#[test]
fn test_fmm_result() {
    let grid = regular_sphere(0);

    const NPTS: usize = 1;

    let nqpts = NPTS * grid.topology().entity_count(grid.topology().dim());
    let kernel = Laplace3dKernel::new();

    let all_points = fmm_tools::get_all_quadrature_points::<NPTS>(&grid);

    let order = 6;
    let alpha_inner = 1.05;
    let alpha_outer = 2.95;
    let depth = 3;
    let global_idxs: Vec<_> = (0..nqpts).collect();

    println!("all points {:?}", &all_points.data()[0..9]);

    // k is the matrix that FMM will give us
    let mut k = rlst_dynamic_array2!(f64, [nqpts, nqpts]);
    kernel.assemble_st(
        EvalType::Value,
        all_points.data(),
        all_points.data(),
        k.data_mut(),
    );

    let mut rng = rand::thread_rng();

    let mut vec = rlst_dynamic_array2!(f64, [nqpts, 1]);
    for i in 0..nqpts {
        *vec.get_mut([i, 0]).unwrap() = rng.gen();
    }
    let dense_result = empty_array::<f64, 2>().simple_mult_into_resize(k.view(), vec.view());

    let tree = SingleNodeTree::new(
        all_points.data(),
        false,
        None,
        Some(depth),
        &global_idxs,
        true,
    );

    let m2l_data =
        FftFieldTranslationKiFmm::new(kernel.clone(), order, *tree.get_domain(), alpha_inner);
    let fmm = KiFmmLinear::new(
        order,
        alpha_inner,
        alpha_outer,
        kernel.clone(),
        tree,
        m2l_data,
    );
    // let charges = vec![1f64; nqpts];
    let charge_dict = build_charge_dict(&global_idxs, &vec.data());
    let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();
    datatree.run(false);

    println!(
        "fmm points {:?}",
        &datatree.fmm.tree().get_all_coordinates().unwrap()[0..9]
    );
    println!("global indices {:?}", &datatree.fmm.tree().global_indices);

    let indices = &datatree.fmm.tree().global_indices;

    let mut fmm_result = rlst_dynamic_array2!(f64, [nqpts, 1]);
    for (i, j) in indices.iter().enumerate() {
        *fmm_result.get_mut([*j, 0]).unwrap() = datatree.potentials[i];
    }

    //let fmm_result = rlst_array_from_slice2!(f64, datatree.potentials.as_slice(), [nqpts, 1]);

    for i in 0..nqpts {
        println!(
            "{} {}",
            dense_result.get([i, 0]).unwrap(),
            fmm_result.get([i, 0]).unwrap()
        );
    }
    assert_eq!(1, 0);
}
