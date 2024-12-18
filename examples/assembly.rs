use bempp::boundary_assemblers::BoundaryAssemblerOptions;
use bempp::function::DefaultFunctionSpace;
use bempp::laplace::assembler::single_layer;
use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::types::{Continuity, ReferenceCellType};
use rlst::{RandomAccessByRef, Shape};

fn main() {
    // Create a grid, family of elements, and function space
    let _ = mpi::initialize().unwrap();
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = bempp::shapes::regular_sphere(0, 1, &comm);
    let element = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
    let space = DefaultFunctionSpace::new(&grid, &element);

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
    let matrix = single_layer(&options).assemble(&space, &space);

    // Print the entries of the matrix
    println!("Lagrange single layer matrix");
    for i in 0..matrix.shape()[0] {
        for j in 0..matrix.shape()[1] {
            print!("{:.4} ", matrix.get([i, j]).unwrap());
        }
        println!();
    }
    println!();
}
