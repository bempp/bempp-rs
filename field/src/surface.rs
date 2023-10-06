//! Tools for handling the reflection of surface and convolution grids.
use itertools::Itertools;

use crate::array::argsort;

/// For a multi-index corresponding to a position on the surface grids created during the KiFMM, reflect its associated multi-index such that it
/// matches the new position positioning of its associated transfer vector now placed in the reference octant.  
///
/// # Arguments
/// * `multi_index` - A three vector describing a surface grid point.
/// * `transfer_vector` - A three vector describing an associated transfer vector with this surface.
/// * `order` - The expansion order relating this surface grid created for the KiFMM.
pub fn axial_reflection_surface(
    multi_index: &[usize],
    transfer_vector: &[i64],
    order: usize,
) -> Vec<usize> {
    // Only valid transfer vectors and surface multi indices are three vectors
    assert!(transfer_vector.len() == 3);
    assert!(multi_index.len() == 3);
    // Order must be greater than 1
    assert!(order > 1);

    fn helper(m: usize, t: i64, order: usize) -> usize {
        if t >= 0 {
            m
        } else {
            order - (m - 1)
        }
    }

    let res = multi_index
        .iter()
        .enumerate()
        .map(|(i, &m)| helper(m, transfer_vector[i], order))
        .collect_vec();

    res
}

/// For a multi-index corresponding to a position on the convolution grid created for the sparsificaiton of the M2L operator in the KIFMM via an FFT,
/// reflect its multi-index such that it matches the new position positioning of its associated transfer vector in the reference octant.  
///
/// # Arguments
/// * `multi_index` - A three vector describing a convolution grid point.
/// * `transfer_vector` - A three vector describing an associated transfer vector with this surface.
/// * `order` - The expansion order relating this convolution grid created for the sparsification of the KiFMM via an FFT.
pub fn axial_reflection_convolution(
    multi_index: &[usize],
    transfer_vector: &[i64],
    order: usize,
) -> Vec<usize> {
    // Only valid transfer vectors are three vectors
    assert!(transfer_vector.len() == 3);
    // Order must be greater than 1
    assert!(order > 1);
    fn helper(m: usize, t: i64, order: usize) -> usize {
        if t >= 0 {
            m
        } else {
            (2 * order - 1) - (m - 1)
        }
    }

    let res = multi_index
        .iter()
        .enumerate()
        .map(|(i, &m)| helper(m, transfer_vector[i], order))
        .collect_vec();

    res
}

/// Perform a diagonal reflection to get a surface multi-index into reference cone. At this point the transfer vectors are assumed to correspond
/// to translations already in the reference cone.
///
/// # Arguments
/// * `multi_index` - A three vector describing a convolution grid point.
/// * `transfer_vector` - A three vector describing an associated transfer vector with this surface.
pub fn diagonal_reflection(multi_index: &[usize], transfer_vector: &[i64]) -> Vec<usize> {
    // Only valid transfer vectors are three vectors
    assert!(transfer_vector.len() == 3);

    let idxs = argsort(transfer_vector);

    let res = idxs.iter().map(|&i| multi_index[i]).collect_vec();

    res
}

#[cfg(test)]
pub mod test {

    use super::*;

    #[test]
    fn test_axial_reflection_surface() {
        // Test when transfer vector has only positive components
        let multi_index = &[1, 2, 3];
        let transfer_vector = &[1, 2, 3];
        let order = 4;

        let expected = vec![1, 2, 3];
        let result = axial_reflection_surface(multi_index, transfer_vector, order);
        assert_eq!(result, expected);

        // Test when transfer vector has all negative components
        let multi_index = &[1, 2, 3];
        let transfer_vector = &[-1, -2, -3];
        let order = 4;

        let expected = vec![4, 3, 2];
        let result = axial_reflection_surface(multi_index, transfer_vector, order);
        assert_eq!(result, expected);

        // Test a mixed component transfer vector
        let multi_index = &[1, 2, 3];
        let transfer_vector = &[1, -2, 3];
        let order = 4;

        let expected = vec![1, 3, 3];
        let result = axial_reflection_surface(multi_index, transfer_vector, order);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_axial_reflection_convolution() {
        let multi_index = &[2, 3, 4];
        let transfer_vector_positive = &[1, 1, 1];
        let transfer_vector_negative = &[-1, -1, -1];
        let order = 5;

        // When transfer_vector is positive, the result should be the same as multi_index
        let result_positive = axial_reflection_convolution(multi_index, transfer_vector_positive, order);
        assert_eq!(result_positive, vec![2, 3, 4]);

        // // When transfer_vector is negative, the result should be reflected based on the order
        // let result_negative = axial_reflection_convolution(multi_index, transfer_vector_negative, order);
        // assert_eq!(result_negative, vec![7, 6, 5]);

        // // Test edge cases, such as if transfer_vector has mixed positive and negative values
        // let transfer_vector_mixed = &[1, -1, 1];
        // let result_mixed = axial_reflection_convolution(multi_index, transfer_vector_mixed, order);
        // assert_eq!(result_mixed, vec![2, 6, 4]);

    }

    #[test]
    fn test_diagonal_reflection() {
        let multi_index = [2, 1, 3];
        let transfer_vector = [3i64, 1i64, 2i64];
        let result = diagonal_reflection(&multi_index, &transfer_vector);

        // Given the sorting of the transfer_vector, we expect the multi_index to be rearranged as [1, 3, 2]
        assert_eq!(result, vec![1, 3, 2]);
    }
}
