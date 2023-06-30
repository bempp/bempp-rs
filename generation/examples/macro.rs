use rs_macros::{generate_kernels, make_answer};

/*
pub fn generate_kernels2(trial_element: &impl FiniteElement, test_element: &impl FiniteElement) -> TokenStream {
    let mut a = 10;
    for i in 0..32 {
        a += 1;
    }

    let kernel_id = "untitled2";
    let mut code = String::new();
    code += &format!("fn __bempp_kernel_{kernel_id}() -> u32 {{ {a} }}");
    code += "\n\n";
    code += &format!("fn __bempp_kernel2_{kernel_id}() -> u32 {{ {} }}", trial_element.highest_degree());
    println!("{}", code);
    code.parse().unwrap()
}
*/

fn main() {
    make_answer!();
    println!("{}", __bempp_kernel_untitled());
    assert_eq!(__bempp_kernel_untitled(), 42);

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
}
