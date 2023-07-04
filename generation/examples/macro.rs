use bempp_generation::generate_kernels;
use bempp_traits::element::FiniteElement;

fn main() {
    generate_kernels!(
        bempp_dp0_dp0_triangle_kernel,
        "Lagrange",
        "Triangle",
        1,
        false,
        "Lagrange",
        "Triangle",
        1,
        false,
        "Lagrange",
        "Triangle",
        1,
        false,
        "Lagrange",
        "Triangle",
        1,
        false
    );

    generate_kernels!(
        bempp_dp0_dp0_triangle_kernel_higher_order,
        "Lagrange",
        "Triangle",
        0,
        true,
        "Lagrange",
        "Triangle",
        1,
        false,
        "Lagrange",
        "Triangle",
        1,
        false,
        "Lagrange",
        "Triangle",
        2,
        false
    );
}
