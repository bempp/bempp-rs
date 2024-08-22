use bempp::assembly::boundary::BoundaryAssembler;
use bempp::function::SerialFunctionSpace;
use bempp::traits::{BoundaryAssembly, FunctionSpace};
use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::types::{Continuity, ReferenceCellType};
use ndgrid::shapes::regular_sphere;
use rlst::{rlst_dynamic_array2, RandomAccessByRef};

fn main() {
    // Create a grid, family of elements, and function space
    let grid = regular_sphere(0);
    let element = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
    let space = SerialFunctionSpace::new(&grid, &element);

    // Create an array to store the assembled discrete operator
    let ndofs = space.global_size();
    let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);

    // Create an assembler for the Laplace single layer operator
    let mut a = BoundaryAssembler::<f64, _, _>::new_laplace_single_layer();

    // Adjust the quadrature degree for non-singular integrals on a triangle.
    // This makes the integrals use a quadrature rule with 16 points
    a.quadrature_degree(ReferenceCellType::Triangle, 16);

    // Adjust the quadrature degree for singular integrals on a pair ortriangles.
    // This makes the integrals use a quadrature rule based on a rule on an interval with 4 points
    a.singular_quadrature_degree(
        (ReferenceCellType::Triangle, ReferenceCellType::Triangle),
        4,
    );

    // Assemble the discrete operator
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
    a.assemble_singular_into_dense(&mut singular_matrix, &space, &space);
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
    let singular_sparse_matrix = a.assemble_singular_into_csr(&space, &space);
    println!("indices: {:?}", singular_sparse_matrix.indices());
    println!("indptr: {:?}", singular_sparse_matrix.indptr());
    println!("data: {:?}", singular_sparse_matrix.data());
    println!();

    // Assemble just the non-singular part
    let colouring = space.cell_colouring();
    let mut nonsingular_matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    a.assemble_nonsingular_into_dense(
        &mut nonsingular_matrix,
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
