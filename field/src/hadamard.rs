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

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub mod aarch64 {

    use super::*;
    use std::arch::aarch64::*;

    /// Compute Hadamard product between the Fourier transform of a set of coefficients corresponding to the coefficients of 8
    /// octree siblings, and the 16 unique sets of kernel evaluations corresponding to unique field translation expected when using
    /// translationally invariant, homogenous kernels that have been reflected into the reference cone. In this implementation we use
    /// explicit Neon SIMD intrinsics to implement the Complex multiplication kernel at the centre of the Hadamard product function.
    ///
    /// # Arguments
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `sibling_coefficients` - A set of Fourier transforms of the multipole coefficients arranged on a convolution grid for
    ///  a set of siblings, arranged in Morton order.
    /// * `kernel_evaluations` - The set of 16 unique kernel evaluations corresponding to unique transfer vectors in the reference cone for
    /// translationally invariant and homogenous kernels.
    /// * `result` - Mutable vector to store results
    pub fn hadamard_product_simd_neon(
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

        for i in 0..16 {
            // Load each kernel matrix into cache, the most expensive operation
            let offset_i = i * size_real;
            let kernel = &kernel_evaluations[offset_i..offset_i + size_real];

            for j in 0..8 {
                let offset_j = j * size_real;
                let signal = &sibling_coefficients[offset_j..offset_j + size_real];

                let chunks = size_real;

                for k in 0..chunks {
                    let simd_index = k;
                    unsafe {
                        let complex_ref = &signal[simd_index];
                        let tuple_ptr: *const (f64, f64) =
                            complex_ref as *const _ as *const (f64, f64);
                        let ptr = tuple_ptr as *const f64;
                        let signal_chunk = vld1q_f64(ptr);

                        let complex_ref = &kernel[simd_index];
                        let tuple_ptr: *const (f64, f64) =
                            complex_ref as *const _ as *const (f64, f64);
                        let ptr = tuple_ptr as *const f64;
                        let kernel_chunk = vld1q_f64(ptr);

                        let complex_ref = &result[simd_index];
                        let tuple_ptr: *const (f64, f64) =
                            complex_ref as *const _ as *const (f64, f64);
                        let ptr = tuple_ptr as *const f64;
                        let res_chunk = vld1q_f64(ptr);

                        // Find component wise product, add with what's already there
                        let product = hadamard_product_kernel_neon(signal_chunk, kernel_chunk);
                        let tmp = vaddq_f64(product, res_chunk);

                        // Save
                        let complex_ref = &result[simd_index];
                        let tuple_ptr: *const (f64, f64) =
                            complex_ref as *const _ as *const (f64, f64);
                        let ptr: *mut f64 = tuple_ptr as *mut f64;

                        vst1q_f64(ptr, tmp)
                    }
                }
            }
        }
    }

    /// Kernel for multiplication of complex numbers, each lane contains a single complex number in the form [[a, b]] where the number is a+ib.
    ///
    /// # Arguments
    /// * `a_ra` - [[a, b]], corresponding to a+ib
    /// * `b_ra` - [[c, d]], corresponding to b+id
    ///
    pub fn hadamard_product_kernel_neon(a_ra: float64x2_t, b_ra: float64x2_t) -> float64x2_t {
        unsafe {
            // Extract real parts [a1, a1]
            let a_real = vdupq_lane_f64(vget_low_f64(a_ra), 0);

            // Extract imaginary parts [b1, b1]
            let a_imag = vdupq_lane_f64(vget_high_f64(a_ra), 0);

            // Multiply real parts [a1c1, a1d1]
            let real_mul = vmulq_f64(a_real, b_ra);

            // Multiply imag parts [b1c1, b1d1]
            let imag_mul = vmulq_f64(a_imag, b_ra);

            // Construct results for real and imaginary parts
            let real = vsub_f64(vget_low_f64(real_mul), vget_high_f64(imag_mul));
            let imag = vadd_f64(vget_high_f64(real_mul), vget_low_f64(imag_mul));

            vcombine_f64(real, imag)
        }
    }
}

#[cfg(all(target_arch = "x86", feature = "avx"))]
pub mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    /// Compute Hadamard product between the Fourier transform of a set of coefficients corresponding to the coefficients of 8
    /// octree siblings, and the 16 unique sets of kernel evaluations corresponding to unique field translation expected when using
    /// translationally invariant, homogenous kernels that have been reflected into the reference cone. In this implementation we use
    /// explicit AVX2 SIMD intrinsics to implement the Complex multiplication kernel at the centre of the Hadamard product function.
    ///
    /// # Arguments
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `sibling_coefficients` - A set of Fourier transforms of the multipole coefficients arranged on a convolution grid for
    ///  a set of siblings, arranged in Morton order.
    /// * `kernel_evaluations` - The set of 16 unique kernel evaluations corresponding to unique transfer vectors in the reference cone for
    /// translationally invariant and homogenous kernels.
    /// * `result` - Mutable vector to store results
    pub fn hadamard_product_simd(
        expansion_order: usize,
        sibling_coefficients: &[Complex64],
        kernel_evaluations: &[Complex64],
        result: &mut [Complex64],
    ) {
        let n = 2 * expansion_order - 1;
        let &(m, n, o) = &(n, n, n);

        let p = m + 1;
        let q = n + 1;
        let r = o + 1;
        let size_real = p * q * (r / 2 + 1);

        for i in 0..16 {
            let offset_i = i * size_real;
            let kernel = &kernel_evaluations[offset_i..offset_i + size_real];

            for j in 0..8 {
                let offset_j = j * size_real;
                let signal = &sibling_coefficients[offset_j..offset_j + size_real];

                let chunk_size = 2;
                let chunks = size_real / chunk_size;

                for k in 0..chunks {
                    let simd_index = k * chunk_size;
                    unsafe {
                        let complex_ref = &signal[simd_index];
                        let tuple_ptr: *const (f64, f64) =
                            complex_ref as *const _ as *const (f64, f64);
                        let ptr = tuple_ptr as *const f64;
                        let signal_chunk = _mm256_loadu_pd(ptr);

                        let complex_ref = &m2l_matrix[simd_index];
                        let tuple_ptr: *const (f64, f64) =
                            complex_ref as *const _ as *const (f64, f64);
                        let ptr = tuple_ptr as *const f64;
                        let kernel_chunk = _mm256_loadu_pd(ptr);

                        let complex_ref = &result[simd_index];
                        let tuple_ptr: *const (f64, f64) =
                            complex_ref as *const _ as *const (f64, f64);
                        let ptr = tuple_ptr as *const f64;
                        let res_chunk = _mm256_loadu_pd(ptr);

                        // Find component wise product, add with what's already there
                        let product = hadamard_product_kernel_avx2(signal_chunk, kernel_chunk);
                        let tmp = _mm256_add_pd(product, res_chunk);

                        // Save
                        let complex_ref = &result[simd_index];
                        let tuple_ptr: *const (f64, f64) =
                            complex_ref as *const _ as *const (f64, f64);
                        let ptr = tuple_ptr as *mut f64;

                        _mm256_storeu_pd(ptr, tmp);
                    }
                }

                // Handle remainder
                let start_remainder = chunks * chunk_size;
                for j in start_remainder..size_real {
                    result[res_offset + j] += signal[j] * m2l_matrix[j];
                }
            }
        }
    }

    /// Kernel for multiplication of complex numbers, each lane contains a two complex number in the form [[a, b, c, d]] where the numbers are a+ib, c+id.
    ///
    /// # Arguments
    /// * `a_ra` - [[a, b, c, d]], corresponding to a+ib, c+id
    /// * `b_ra` - [[e, f, g, h]], corresponding to e+if, g+ih
    pub fn hadamard_product_kernel_avx2(a_ra: __m256d, b_ra: __m256d) -> __m256d {
        unsafe {
            // Extract real parts [a1, a1, a2, a2]
            let a_real = _mm256_shuffle_pd(a_ra, a_ra, 0b0);

            // Extract imaginary parts [b1, b1, b2, b2]
            let a_imag = _mm256_shuffle_pd(a_ra, a_ra, 0b1111);

            // Multiply real parts [a1c1, a1d1, a2c2, a2d2]
            let real_mul = _mm256_mul_pd(a_real, b_ra);

            // Multiply imag parts [b1c1, b1d1, b2c2, b2d2]
            let imag_mul = _mm256_mul_pd(a_imag, b_ra);

            // Find components that go in the real and imag part of solution
            // [a1c1, b1d1, a2c2, b2d2]
            let real_part = _mm256_blend_pd(real_mul, imag_mul, 0b1010);
            // [b1c1, a1d1, b2c2, a2d2]
            let imag_part = _mm256_blend_pd(real_mul, imag_mul, 0b0101);

            // [a1c1-b1d1, a1c1-b1d1, a2c2-b2d2, a2c2-b2d2]
            let real = _mm256_hsub_pd(real_part, real_part);
            let imag = _mm256_hadd_pd(imag_part, imag_part);

            let result = _mm256_blend_pd(real, imag, 0b1010);

            result
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

        // Test just the convolutions being applied to the first child. Also provides an example of how to access convolutions.d
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