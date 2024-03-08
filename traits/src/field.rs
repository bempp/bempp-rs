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
    /// * `order` - the order of expansions used in constructing the surface grid
    /// * `domain` - Domain associated with the global point set.
    fn set_operator_data(&mut self, order: usize, domain: Self::Domain);

    fn set_expansion_order(&mut self, expansion_order: usize);

    fn set_kernel(&mut self, kernel: T);
}

/// Interface for field translations.
pub trait SourceToTarget {
    /// Interface for a field translation operation, takes place over each level of an octree.
    ///
    /// # Arguments
    /// * `level` - The level of the tree at which a field translation is being applied.
    fn m2l(&self, level: u64);

    /// Particle to local translations, applies to leaf boxes when a source box is within
    /// the far field of a target box, but is too large for the multipole expansion to converge
    /// at the target, so instead its contribution is computed directly.
    ///
    /// # Arguments
    /// * `level` - The level of the tree at which a field translation is being applied.
    fn p2l(&self, level: u64);
}
