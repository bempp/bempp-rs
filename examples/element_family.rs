use bempp::element::ciarlet::LagrangeElementFamily;
use bempp::traits::{
    element::{Continuity, ElementFamily, FiniteElement},
    types::ReferenceCell,
};

extern crate blas_src;
extern crate lapack_src;

fn main() {
    // Create the degree 2 Lagrange element family. A family is a set of finite elements with the
    // same family type, degree, and continuity across a set of cells
    let family = LagrangeElementFamily::<f64>::new(2, Continuity::Continuous);

    // Get the element in the family on a triangle
    let element = family.element(ReferenceCell::Triangle);
    println!("Cell: {:?}", element.cell_type());

    // Get the element in the family on a triangle
    let element = family.element(ReferenceCell::Quadrilateral);
    println!("Cell: {:?}", element.cell_type());
}
