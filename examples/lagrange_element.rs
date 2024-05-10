use bempp::element::ciarlet::lagrange;
use bempp::traits::{
    element::{Continuity, FiniteElement},
    types::ReferenceCellType,
};
use rlst::{rlst_dynamic_array2, rlst_dynamic_array4, RawAccess};

extern crate lapack_src;
extern crate blas_src;

fn main() {
    // Create a P2 element on a triangle
    let element = lagrange::create::<f64>(ReferenceCellType::Triangle, 2, Continuity::Continuous);

    println!("This element has {} basis functions.", element.dim());

    // Create an array to store the basis function values
    let mut basis_values = rlst_dynamic_array4!(f64, element.tabulate_array_shape(0, 1));
    // Create array containing the point [1/3, 1/3]
    let mut points = rlst_dynamic_array2!(f64, [1, 2]);
    points[[0, 0]] = 1.0 / 3.0;
    points[[0, 1]] = 1.0 / 3.0;
    // Tabulate the element's basis functions at the point
    element.tabulate(&points, 0, &mut basis_values);
    println!(
        "The values of the basis functions at the point (1/3, 1/3) are: {:?}",
        basis_values.data()
    );

    // Set point to [1, 0]
    points[[0, 0]] = 1.0;
    points[[0, 1]] = 0.0;
    // Tabulate the element's basis functions at the point
    element.tabulate(&points, 0, &mut basis_values);
    println!(
        "The values of the basis functions at the point (1, 0) are: {:?}",
        basis_values.data()
    );
}
