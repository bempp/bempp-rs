//! Implementations of 8x8 operation for Hadamard product in FFT based M2L operations.
use bempp_traits::types::Scalar;
use num::{complex::Complex, Zero};

#[inline(always)]
pub fn matmul8x8<U>(
    kernel: &[Complex<U>],
    signal: &[Complex<U>],
    save_locations: &mut [Complex<U>],
    scale: Complex<U>,
) where
    U: Scalar,
{
    for i in 0..8 {
        let mut sum: Complex<U> = Complex::zero();
        sum += kernel[i * 8] * signal[0];
        sum += kernel[i * 8 + 1] * signal[1];
        sum += kernel[i * 8 + 2] * signal[2];
        sum += kernel[i * 8 + 3] * signal[3];
        sum += kernel[i * 8 + 4] * signal[4];
        sum += kernel[i * 8 + 5] * signal[5];
        sum += kernel[i * 8 + 6] * signal[6];
        sum += kernel[i * 8 + 7] * signal[7];
        save_locations[i] += sum * scale;
    }
}
