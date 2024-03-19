use bempp_bem::assembly::{batched, batched::BatchedAssembler};
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::LagrangeElementFamily;
use bempp_grid::shapes::regular_sphere;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::element::Continuity;
use rlst::{rlst_dynamic_array2, RandomAccessByRef};

extern crate blas_src;
extern crate lapack_src;

fn main() {
    // Create a grid, family of elements, and function space
    let grid = regular_sphere(0);
    let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    // Create an array to store the assembled discrete operator
    let ndofs = space.global_size();
    let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);

    // Create an assembler for the Laplace single layer operator and use it to assemble the discrete operator
    let a = batched::LaplaceSingleLayerAssembler::<128, f64>::default();
    a.assemble_into_dense(&mut matrix, &space, &space);

    // Print the entries of the matrix
    for i in 0..ndofs {
        for j in 0..ndofs {
            println!("{}", matrix.get([i, j]).unwrap());
        }
        println!();
    }
}
