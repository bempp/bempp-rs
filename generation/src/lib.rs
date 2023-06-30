extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn make_answer(_item: TokenStream) -> TokenStream {
    let a = 42;
    let kernel_id = "untitled";
    let mut code = String::new();
    code += &format!("fn __bempp_kernel_{kernel_id}() -> u32 {{ {a} }}");
    code.parse().unwrap()
}
