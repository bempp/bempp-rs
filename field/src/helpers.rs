pub fn ncoeffs_kifmm(expansion_order: usize) -> usize {
    6 * (expansion_order - 1).pow(2) + 2
}
