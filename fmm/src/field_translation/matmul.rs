//! Implementations of 8x8 matrix multiplication operation during Hadamard product in FFT based M2L operations.
use bempp_traits::types::RlstScalar;
use num::{complex::Complex, Zero};

/// The 8x8 matmul operation, always inlined. Implemented via a fully unrolled inner loop, and partially unrolled outer loop.
///
/// # Arguments
/// * - `kernel` - The kernel data for a specific translation for a specific frequency for a set of 8 siblings.
/// * - `signal` - The signal data at a specific frequency for a set of 8 siblings.
/// * `save_locations` - Reference to where the check potential, in frequency space, is being stored for this frequency and set of siblings.
/// * `scale` - The scaling factor of the M2L translation.
#[inline(always)]
pub fn matmul8x8<U>(
    kernel: &[Complex<U>],
    signal: &[Complex<U>],
    save_locations: &mut [Complex<U>],
    scale: Complex<U>,
) where
    U: RlstScalar,
{
    let s1 = signal[0];
    let s2 = signal[1];
    let s3 = signal[2];
    let s4 = signal[3];
    let s5 = signal[4];
    let s6 = signal[5];
    let s7 = signal[6];
    let s8 = signal[7];

    for i in 0..4 {
        let mut sum1: Complex<U> = Complex::zero();
        let mut sum2: Complex<U> = Complex::zero();
        let i1 = 2 * i;
        let i2 = 2 * i + 1;

        sum1 += kernel[i1 * 8] * s1;
        sum1 += kernel[i1 * 8 + 1] * s2;
        sum1 += kernel[i1 * 8 + 2] * s3;
        sum1 += kernel[i1 * 8 + 3] * s4;
        sum1 += kernel[i1 * 8 + 4] * s5;
        sum1 += kernel[i1 * 8 + 5] * s6;
        sum1 += kernel[i1 * 8 + 6] * s7;
        sum1 += kernel[i1 * 8 + 7] * s8;

        sum2 += kernel[i2 * 8] * s1;
        sum2 += kernel[i2 * 8 + 1] * s2;
        sum2 += kernel[i2 * 8 + 2] * s3;
        sum2 += kernel[i2 * 8 + 3] * s4;
        sum2 += kernel[i2 * 8 + 4] * s5;
        sum2 += kernel[i2 * 8 + 5] * s6;
        sum2 += kernel[i2 * 8 + 6] * s7;
        sum2 += kernel[i2 * 8 + 7] * s8;

        save_locations[i1] += sum1 * scale;
        save_locations[i2] += sum2 * scale;
    }
}
