// Definition of kernel functions.
use crate::types::Result;

pub trait Kernel {
    // Evaluation data.
    type PotentialData;
    type GradientData;

    // Space dimensions for the input of the kernel
    fn dim(&self) -> usize;

    // Dimensionality of the output values
    fn value_dimension(&self) -> usize;

    // Return of the kernel is singular.
    //
    // A singular kernel is not defined
    // when sources and charges are identical.
    fn is_singular(&self) -> bool;

    // Evaluate the potential kernel.
    fn potential(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
        potentials: &mut [f64],
    );

    // Evaluate the kernel gradient.
    fn gradient(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
    ) -> Result<Self::GradientData>;

    fn gram(&self, sources: &[[f64; 3]], targets: &[[f64; 3]]) -> Result<Self::PotentialData>;

    fn scale(&self, level: u64) -> f64;
}

// A trait that describes evaluation data for a kernel.
pub trait KernelEvaluationData {
    type Item: crate::types::Scalar;

    // The number of targets.
    fn number_of_targets(&self) -> usize;

    // Dimensionality of the kernel output (e.g. scalar=1, vectorial = 3)
    fn value_dimension(&self) -> usize;

    // Return the data at a given target index.
    fn data_at_target(&self, index: usize) -> &[&Self::Item];
}
