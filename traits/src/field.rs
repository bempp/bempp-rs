use crate::kernel::Kernel;

/// Container for metadata associated with a field translation implementation.
pub trait FieldTranslationData<T>
where
    T: Kernel,
{
    /// A vector specifying the displacement between a source and target node.
    /// Defines the field translation operator being applied
    type TransferVector;

    /// The specific data structure holding the field translation operators for this method.
    /// Each translation operator corresponds to a transfer vector.
    type M2LOperators;

    /// The computational domain for these operators, defined by the input points distribution.
    type Domain;

    /// Compute the field translation operators corresponding to each unique transfer vector.
    ///
    /// # Arguments
    /// * `order` - the order of expansions used in constructing the surface grid
    /// * `domain` - Domain associated with the global point set.
    fn compute_m2l_operators(&self, order: usize, domain: Self::Domain) -> Self::M2LOperators;

    /// Number of coefficients for a given expansion order in a given FMM scheme.
    ///
    /// # Arguments
    /// * `order` - the order of expansions used in constructing the surface grid
    fn ncoeffs(&self, order: usize) -> usize;
}

/// Interface for field translations.
pub trait FieldTranslation {
    /// # Warning
    /// This method is only applicable to homogeneous kernels, which are currently
    /// implemented by our software. This method is staged to be deprecated.
    ///
    /// # Arguments
    /// * `level` - The level of the tree at which a field translation is being applied.
    fn m2l_scale(&self, level: u64) -> f64;

    /// Interface for a field translation operation, takes place over each level of an octree.
    ///
    /// # Arguments
    /// * `level` - The level of the tree at which a field translation is being applied.
    fn m2l(&self, level: u64);
}
