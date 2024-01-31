use rand::Rng;
use bempp_grid::shapes::regular_sphere;
use rlst_dense::array::empty_array;
use rlst_dense::traits::MultIntoResize;
use bempp_traits::types::EvalType;
use bempp_kernel::laplace_3d::Laplace3dKernel;
use rlst_dense::traits::RandomAccessByRef;
use bempp_traits::kernel::Kernel;
use rlst_dense::rlst_dynamic_array2;
use rlst_dense::traits::RandomAccessMut;
use rlst_dense::traits::RawAccessMut;
use rlst_dense::traits::RawAccess;
use bempp_bem::assembly::{batched, fmm_tools};
use bempp_traits::grid::Grid;
use bempp_traits::grid::Topology;

#[test]
fn test_fmm_result() {
    let grid = regular_sphere(0);

    const NPTS: usize = 1;

    let nqpts = NPTS * grid.topology().entity_count(grid.topology().dim());
    let kernel = Laplace3dKernel::new();

    let all_points = fmm_tools::get_all_quadrature_points::<NPTS>(&grid);


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
    let dense_result = empty_array::<f64, 2>().simple_mult_into_resize(
        k.view(), vec.view());


    let fmm_result = rlst_dynamic_array2!(f64, [nqpts, 1]);

    for i in 0..nqpts {
        
        println!("{} {}", dense_result.get([0,0]).unwrap(), fmm_result.get([0,0]).unwrap());
    }
    assert_eq!(1, 0);
}
