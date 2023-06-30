use rs_macros::make_answer;

fn main() {
    make_answer!();
    println!("{}", __bempp_kernel_untitled());
    assert_eq!(__bempp_kernel_untitled(), 42);
}
