//! Traits for creating integral equation kernels.
use crate::types::Result;

/// Interface for FMM kernels.
pub trait Kernel {
    /// Potential data container.
    type PotentialData;

    /// Gradient data container.
    type GradientData;

    /// Space dimensions for the input of the kernel.
    fn dim(&self) -> usize;

    /// Dimensionality of the output values.
    fn value_dimension(&self) -> usize;

    /// Return of the kernel is singular.
    ///
    /// A singular kernel is not defined
    /// when sources and charges are identical.
    fn is_singular(&self) -> bool;

    /// Evaluate the potential kernel.
    fn potential(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
        potentials: &mut [f64],
    );

    /// Evaluate the gradient kernel.
    fn gradient(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
        gradients: &mut [[f64; 3]],
    );

    /// Evaluate the Gram matrix.
    fn gram(&self, sources: &[[f64; 3]], targets: &[[f64; 3]]) -> Result<Self::PotentialData>;

    /// Scale the kernel to a given level of an associated tree.
    fn scale(&self, level: u64) -> f64;
}
