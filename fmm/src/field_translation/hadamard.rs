//! Implementations of 8x8 operation for Hadamard product in FFT based M2L operations.

use bempp_traits::types::Scalar;
use num::complex::Complex;

#[inline(always)]
pub unsafe fn matmul8x8x2<U>(
    kernel_data_freq: &[Complex<U>],
    signal: &[Complex<U>],
    save_locations: &mut [Complex<U>],
    scale: Complex<U>,
) where
    U: Scalar,
{
    for j in 0..8 {
        let kernel_data_j = &kernel_data_freq[j * 8..(j + 1) * 8];
        let sig = signal[j];

        save_locations
            .iter_mut()
            .zip(kernel_data_j.iter())
            .for_each(|(sav, &ker)| *sav += scale * ker * sig)
    }
}
