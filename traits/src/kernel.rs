// Definition of kernel functions.
use crate::types::Result;

pub trait Kernel {
    // Evaluation data;
    type Data: KernelEvaluationData;

    // Space dimensions for the input of the kernel
    fn dim(&self) -> usize;

    // Dimensionality of the output values
    fn value_dimension(&self) -> usize;

    // Return of the kernel is singular.
    //
    // A singular kernel is not defined
    // when sources and charges are identical.
    fn is_singular(&self) -> bool;

    // Evaluate the kernel.
    fn evaluate(
        &self,
        sources: &[f64],
        charges: &[f64],
        targets: &[f64],
        eval_type: &EvalType,
    ) -> Result<Data>;
}

// A trait that describes evaluation data for a kernel.
pub trait KernelEvaluationData {
    type Item: crate::types::Scalar;

    // The number of targets.
    fn number_of_targets(&self) -> usize;

    // Dimensionality of the kernel output (e.g. scalar=1, vectorial = 3)
    fn value_dimension(&self) -> usize;

    // Return the data at a given target index.
    fn data_at_target(&self, index: usize) -> &[Item];
}
