//! Fast Hadamard product kernels for coefficient data on octrees.

use num::complex::Complex64;

/// Compute Hadamard product between the Fourier transform of a set of coefficients corresponding to the coefficients of 8
/// octree siblings, and the 16 unique sets of kernel evaluations corresponding to unique field translation expected when using
/// translationally invariant, homogenous kernels that have been reflected into the reference cone.
///
/// # Arguments
/// * `order` - The expansion order for the multipole and local expansions.
/// * `sibling_coefficients` - A set of Fourier transforms of the multipole coefficients arranged on a convolution grid for
///  a set of siblings, arranged in Morton order.
/// * `kernel_evaluations` - The set of 16 unique kernel evaluations corresponding to unique transfer vectors in the reference cone for
/// translationally invariant and homogenous kernels.
/// * `result` - Mutable vector to store results
pub fn hadamard_product_sibling(
    order: usize,
    sibling_coefficients: &[Complex64],
    kernel_evaluations: &[Complex64],
    result: &mut [Complex64],
) {
    let n = 2 * order - 1;
    let &(m, n, o) = &(n, n, n);

    let p = m + 1;
    let q = n + 1;
    let r = o + 1;
    let size_real = p * q * (r / 2 + 1);

    let nsiblings = 8;
    let nconvolutions = 16;

    for i in 0..nconvolutions {
        let offset_i = i * size_real;

        // Load each kernel matrix into cache, the most expensive operation
        let kernel = &kernel_evaluations[offset_i..offset_i + size_real];

        // Iterate over siblings in inner loop, applying same convolution to each one.
        for j in 0..nsiblings {
            let offset_j = j * size_real;
            let signal = &sibling_coefficients[offset_j..offset_j + size_real];

            for k in 0..size_real {
                result[j * size_real * nconvolutions + i * size_real + k] += kernel[k] * signal[k];
            }
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use cauchy::c64;
    use rlst::dense::{rlst_mat, rlst_rand_mat, RawAccess, RawAccessMut};

    #[test]
    fn test_hadamard_product_sibling() {
        let nconv = 16;
        let nsiblings = 8;
        let order = 5;
        let n = 2 * order - 1;
        let &(m, n, o) = &(n, n, n);

        let p = m + 1;
        let q = n + 1;
        let r = o + 1;
        let size_real = p * q * (r / 2 + 1);

        let sibling_coefficients = rlst_rand_mat![c64, (nsiblings * size_real, 1)];
        let kernel_evaluations = rlst_rand_mat![c64, (nconv * size_real, 1)];
        let mut result = rlst_mat![c64, (size_real * nsiblings * nconv, 1)];

        hadamard_product_sibling(
            order,
            sibling_coefficients.data(),
            kernel_evaluations.data(),
            result.data_mut(),
        );

        // Test just the convolutions being applied to the first child. Also provides an example of how to access convolutions.
        for i in 0..16 {
            for j in 0..size_real {
                let expected =
                    sibling_coefficients.data()[j] * kernel_evaluations.data()[i * size_real + j];
                let res = result.data()[j + i * size_real];
                assert_eq!(res, expected)
            }
        }
    }
}
