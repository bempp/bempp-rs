//! Interface to kernels

pub use cauchy::Scalar;

pub enum EvalType {
    Value,
    Deriv,
    ValueDeriv,
}

pub trait Kernel {
    type Item: Scalar;
    // Space dimensions for the input of the kernel
    fn dim(&self) -> usize;

    // Dimensionality of the output values
    fn value_size(&self) -> usize;

    fn evaluate(
        &self,
        sources: &[f64],
        charges: &[f64],
        targets: &[f64],
        output: &mut [Self::Item],
        eval_type: &EvalType,
    );
}
