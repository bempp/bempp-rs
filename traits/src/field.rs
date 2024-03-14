//! Field traits
use crate::kernel::Kernel;

/// Container for metadata associated with a field translation implementation.
pub trait SourceToTargetData<T>
where
    T: Kernel,
{
    /// Metadata for applying each to source to target translation, depends on both the kernel
    /// and translation method
    type OperatorData;

    /// The computational domain for these operators, defined by the input points distribution.
    type Domain;

    /// Compute the field translation operators corresponding to each unique transfer vector.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    /// * `domain` - Domain associated with the global point set.
    fn set_operator_data(&mut self, expansion_order: usize, domain: Self::Domain);

    /// Set expansion order
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    fn set_expansion_order(&mut self, expansion_order: usize);

    /// Set the associated kernel
    ///
    /// # Arguments
    /// * `kernel` - The kernel being used
    fn set_kernel(&mut self, kernel: T);
}
