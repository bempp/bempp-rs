//! Implementations of 8x8 operation for Hadamard product in FFT based M2L operations.
use bempp_traits::types::Scalar;
use num::complex::Complex;


/// The 8x8 operation computed naively with Rust iterators, always inlined.
/// 
/// # Arguments
/// * - `kernel` - The kernel data for a specific translation for a specific frequency for a set of 8 siblings.
/// * - `signal` - The signal data at a specific frequencty for a set of 8 siblings.
/// * `save_locations` - Reference to where the check potential, in frequency space, is being stored for this frequency and set of siblings.
/// * `scale` - The scaling factor of the M2L translation.
#[inline(always)]
pub fn matmul8x8<U>(
    kernel: &[Complex<U>],
    signal: &[Complex<U>],
    save_locations: &mut [Complex<U>],
    scale: Complex<U>,
) where
    U: Scalar,
{
    for j in 0..8 {
        let kernel_data_j = &kernel[j * 8..(j + 1) * 8];
        let sig = signal[j];

        save_locations
            .iter_mut()
            .zip(kernel_data_j.iter())
            .for_each(|(sav, &ker)| *sav += scale * ker * sig)
    }
}
