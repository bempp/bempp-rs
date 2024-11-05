use bempp::boundary_assemblers::BoundaryAssemblerOptions;
use bempp::function::SerialFunctionSpace;
use bempp::laplace::assembler::laplace_single_layer;
use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::types::{Continuity, ReferenceCellType};
use ndgrid::shapes::regular_sphere;
use rlst::{RandomAccessByRef, Shape};

fn main() {
    // Create a grid, family of elements, and function space
    let grid = regular_sphere(0);
    let element = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
    let space = SerialFunctionSpace::new(&grid, &element);

    // Adjust the quadrature degree for non-singular integrals on a triangle.
    // This makes the integrals use a quadrature rule with 16 points
    let mut options = BoundaryAssemblerOptions::default();
    options.set_regular_quadrature_degree(ReferenceCellType::Triangle, 16);

    // Adjust the quadrature degree for singular integrals on a pair ortriangles.
    // This makes the integrals use a quadrature rule based on a rule on an interval with 4 points
    options.set_singular_quadrature_degree(
        (ReferenceCellType::Triangle, ReferenceCellType::Triangle),
        4,
    );

    // Assemble the single layer Laplace boundary operator.
    let matrix = laplace_single_layer(&options).assemble(&space, &space);

    // Print the entries of the matrix
    println!("Lagrange single layer matrix");
    for i in 0..matrix.shape()[0] {
        for j in 0..matrix.shape()[1] {
            print!("{:.4} ", matrix.get([i, j]).unwrap());
        }
        println!();
    }
    println!();

    // // Assemble just the singular part
    // let mut singular_matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    // a.assemble_singular_into_dense(&mut singular_matrix, &space, &space);
    // println!("Lagrange single layer matrix (singular part)");
    // for i in 0..ndofs {
    //     for j in 0..ndofs {
    //         print!("{:.4} ", singular_matrix.get([i, j]).unwrap());
    //     }
    //     println!();
    // }
    // println!();

    // // For grids with a larger number of cells, the singular part will be sparse. It can be assembled as a CSR matrix as follows
    // println!("Lagrange single layer matrix (singular part) as CSR matrix");
    // let singular_sparse_matrix = a.assemble_singular_into_csr(&space, &space);
    // println!("indices: {:?}", singular_sparse_matrix.indices());
    // println!("indptr: {:?}", singular_sparse_matrix.indptr());
    // println!("data: {:?}", singular_sparse_matrix.data());
    // println!();

    // // Assemble just the non-singular part
    // let colouring = space.cell_colouring();
    // let mut nonsingular_matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    // a.assemble_nonsingular_into_dense(
    //     &mut nonsingular_matrix,
    //     &space,
    //     &space,
    //     &colouring,
    //     &colouring,
    // );
    // println!("Lagrange single layer matrix (nonsingular part)");
    // for i in 0..ndofs {
    //     for j in 0..ndofs {
    //         print!("{:.4} ", nonsingular_matrix.get([i, j]).unwrap());
    //     }
    //     println!();
    // }
    // println!();
}
