//! Helper functions

/// Number of coefficients for multipole and local expansions for the kernel independent FMM
/// for a given expansion order. Coefficients correspond to points on the equivalent surface.
///
/// # Arguments
/// * `expansion_order` - Expansion order of the FMM
pub fn ncoeffs_kifmm(expansion_order: usize) -> usize {
    6 * (expansion_order - 1).pow(2) + 2
}
