//! Implementations of 8x8 operation for Hadamard product in FFT based M2L operations.
use bempp_traits::types::Scalar;
use num::{complex::Complex, Zero};

// /// The 8x8 operation computed naively with Rust iterators, always inlined.
// ///
// /// # Arguments
// /// * - `kernel` - The kernel data for a specific translation for a specific frequency for a set of 8 siblings.
// /// * - `signal` - The signal data at a specific frequency for a set of 8 siblings.
// /// * `save_locations` - Reference to where the check potential, in frequency space, is being stored for this frequency and set of siblings.
// /// * `scale` - The scaling factor of the M2L translation.
// #[inline(always)]
// pub fn matmul8x8<U>(
//     kernel: &[Complex<U>],
//     signal: &[Complex<U>],
//     save_locations: &mut [Complex<U>],
//     scale: Complex<U>,
// ) where
//     U: Scalar,
// {
//     for j in 0..8 {
//         let kernel_data_j = &kernel[j * 8..(j + 1) * 8];
//         let sig = signal[j];

//         save_locations
//             .iter_mut()
//             .zip(kernel_data_j.iter())
//             .for_each(|(sav, &ker)| *sav += scale * ker * sig)
//     }
// }
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
