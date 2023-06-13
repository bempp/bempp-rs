//! Traits for creating integral equation kernels.

/// Interface for FMM kernels.
pub trait Kernel {

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
    fn potential(&self, sources: &[f64], charges: &[f64], targets: &[f64], potentials: &mut [f64]);

    /// Evaluate the Gram matrix.
    fn gram(&self, sources: &[f64], targets: &[f64], result: &mut Vec<f64>);

    /// Scale the kernel to a given level of an associated tree.
    fn scale(&self, level: u64) -> f64;
}
