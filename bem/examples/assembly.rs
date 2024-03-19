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
    let element = LagrangeElementFamily::<f64>::new(1, Continuity::Continuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    // Create an array to store the assembled discrete operator
    let ndofs = space.global_size();
    let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);

    // Create an assembler for the Laplace single layer operator and use it to assemble the discrete operator
    let a = batched::LaplaceSingleLayerAssembler::<128, f64>::default();
    a.assemble_into_dense(&mut matrix, &space, &space);

    // Print the entries of the matrix
    println!("Lagrange single layer matrix");
    for i in 0..ndofs {
        for j in 0..ndofs {
            print!("{:.4} ", matrix.get([i, j]).unwrap());
        }
        println!();
    }
    println!();

    // Assemble just the singular part
    let mut singular_matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    a.assemble_singular_into_dense(&mut singular_matrix, 4, &space, &space);
    println!("Lagrange single layer matrix (singular part)");
    for i in 0..ndofs {
        for j in 0..ndofs {
            print!("{:.4} ", singular_matrix.get([i, j]).unwrap());
        }
        println!();
    }
    println!();

    // For grids with a larger number of cells, the singular part will be sparse. It can be assembled as a CSR matrix as follows
    println!("Lagrange single layer matrix (singular part) as CSR matrix");
    let singular_sparse_matrix = a.assemble_singular_into_csr(4, &space, &space);
    println!("indices: {:?}", singular_sparse_matrix.indices());
    println!("indptr: {:?}", singular_sparse_matrix.indptr());
    println!("data: {:?}", singular_sparse_matrix.data());
    println!();

    // Assemble just the non-singular part
    let colouring = space.cell_colouring();
    let mut nonsingular_matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    a.assemble_nonsingular_into_dense(
        &mut nonsingular_matrix,
        37,
        37,
        &space,
        &space,
        &colouring,
        &colouring,
    );
    println!("Lagrange single layer matrix (nonsingular part)");
    for i in 0..ndofs {
        for j in 0..ndofs {
            print!("{:.4} ", nonsingular_matrix.get([i, j]).unwrap());
        }
        println!();
    }
    println!();
}
