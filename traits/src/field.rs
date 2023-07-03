//! Traits for multipole-to-local (M2L) field translations

use crate::kernel::Kernel;

pub trait FieldTranslationData<T>
where
    T: Kernel,
{
    type TransferVector;

    type M2LOperators;

    type Domain;

    // Compute unique transfer vectors
    fn compute_transfer_vectors(&self) -> Self::TransferVector;

    fn compute_m2l_operators(
        &self,
        expansion_order: usize,
        domain: Self::Domain,
    // );
    ) -> Self::M2LOperators;

    fn ncoeffs(&self, expansion_order: usize) -> usize;
}

// Designed to be implemented by DataTrees which contain the required data
// for computing field translations
pub trait FieldTranslation {
    // How to scale operator with tree level.
    fn m2l_scale(&self, level: u64) -> f64;

    // Convolution operation over each level.
    fn m2l(&self, level: u64);
}
