//! Wrappers for FFTW functions, including multithreaded implementations.
use fftw::{plan::*, types::*};
use num::Complex;
use rayon::prelude::*;

use crate::types::{FftMatrixc32, FftMatrixc64, FftMatrixf32, FftMatrixf64};
use rlst_dense::traits::RawAccessMut;

pub trait Fft<DtypeReal, DtypeCplx>
where
    Self: Sized,
{
    /// Compute a Real FFT over a rlst matrix which stores data corresponding to multiple 3 dimensional arrays of shape `shape`, stored in column major order.
    /// This function is multithreaded, and uses the FFTW library.
    ///
    /// # Arguments
    /// * `input` - Input slice of real data, corresponding to a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of input data.
    fn rfft3_fftw_par_vec(input: &mut DtypeReal, output: &mut DtypeCplx, shape: &[usize]);

    /// Compute an inverse Real FFT over a rlst matrix which stores data corresponding to multiple 3 dimensional arrays of shape `shape`, stored in column major order.
    /// This function is multithreaded, and uses the FFTW library.
    ///
    /// # Arguments
    /// * `input` - Input slice of complex data, corresponding to an FFT of a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of output data.
    fn irfft_fftw_par_vec(input: &mut DtypeCplx, output: &mut DtypeReal, shape: &[usize]);

    /// Compute a Real FFT of an input slice corresponding to a 3D array stored in column major format, specified by `shape` using the FFTW library.
    ///
    /// # Arguments
    /// * `input` - Input slice of real data, corresponding to a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of input data.
    fn rfft3_fftw(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]);

    /// Compute an inverse Real FFT of an input slice corresponding to the FFT of a 3D array stored in column major format, specified by `shape` using the FFTW library.
    /// This function normalises the output.
    ///
    /// # Arguments
    /// * `input` - Input slice of complex data, corresponding to an FFT of a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of output data.
    fn irfft3_fftw(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]);
}

// impl Fft<FftMatrixf32, FftMatrixc32> for f32 {
//     fn rfft3_fftw_par_vec(input: &mut FftMatrixf32, output: &mut FftMatrixc32, shape: &[usize]) {
//         let size: usize = shape.iter().product();
//         let size_d = shape.last().unwrap();
//         let size_real = (size / size_d) * (size_d / 2 + 1);
//         let plan: R2CPlan32 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();

//         let it_inp = input.data_mut().par_chunks_exact_mut(size).into_par_iter();
//         let it_out = output
//             .data_mut()
//             .par_chunks_exact_mut(size_real)
//             .into_par_iter();

//         it_inp.zip(it_out).for_each(|(inp, out)| {
//             let _ = plan.r2c(inp, out);
//         });
//     }

//     fn irfft_fftw_par_vec(input: &mut FftMatrixc32, output: &mut FftMatrixf32, shape: &[usize]) {
//         let size: usize = shape.iter().product();
//         let size_d = shape.last().unwrap();
//         let size_real = (size / size_d) * (size_d / 2 + 1);
//         let plan: C2RPlan32 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();

//         let it_inp = input
//             .data_mut()
//             .par_chunks_exact_mut(size_real)
//             .into_par_iter();
//         let it_out = output.data_mut().par_chunks_exact_mut(size).into_par_iter();

//         it_inp.zip(it_out).for_each(|(inp, out)| {
//             let _ = plan.c2r(inp, out);
//             // Normalise output
//             out.iter_mut()
//                 .for_each(|value| *value *= 1.0 / (size as f32));
//         })
//     }

//     fn rfft3_fftw(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]) {
//         assert!(shape.len() == 3);
//         let plan: R2CPlan32 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();
//         let _ = plan.r2c(input, output);
//     }

//     fn irfft3_fftw(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]) {
//         assert!(shape.len() == 3);
//         let size: usize = shape.iter().product();
//         let plan: C2RPlan32 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();
//         let _ = plan.c2r(input, output);
//         // Normalise
//         output
//             .iter_mut()
//             .for_each(|value| *value *= 1.0 / (size as f32));
//     }
// }

// impl Fft<FftMatrixf64, FftMatrixc64> for f64 {
//     fn rfft3_fftw_par_vec(input: &mut FftMatrixf64, output: &mut FftMatrixc64, shape: &[usize]) {
//         let size: usize = shape.iter().product();
//         let size_d = shape.last().unwrap();
//         let size_real = (size / size_d) * (size_d / 2 + 1);
//         let plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();

//         let it_inp = input.data_mut().par_chunks_exact_mut(size).into_par_iter();
//         let it_out = output
//             .data_mut()
//             .par_chunks_exact_mut(size_real)
//             .into_par_iter();

//         it_inp.zip(it_out).for_each(|(inp, out)| {
//             let _ = plan.r2c(inp, out);
//         });
//     }

//     fn irfft_fftw_par_vec(input: &mut FftMatrixc64, output: &mut FftMatrixf64, shape: &[usize]) {
//         let size: usize = shape.iter().product();
//         let size_d = shape.last().unwrap();
//         let size_real = (size / size_d) * (size_d / 2 + 1);
//         let plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();

//         let it_inp = input
//             .data_mut()
//             .par_chunks_exact_mut(size_real)
//             .into_par_iter();
//         let it_out = output.data_mut().par_chunks_exact_mut(size).into_par_iter();

//         it_inp.zip(it_out).for_each(|(inp, out)| {
//             let _ = plan.c2r(inp, out);
//             // Normalise output
//             out.iter_mut()
//                 .for_each(|value| *value *= 1.0 / (size as f64));
//         })
//     }

//     fn rfft3_fftw(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]) {
//         assert!(shape.len() == 3);
//         let plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();
//         let _ = plan.r2c(input, output);
//     }

//     fn irfft3_fftw(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]) {
//         assert!(shape.len() == 3);
//         let size: usize = shape.iter().product();
//         let plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();
//         let _ = plan.c2r(input, output);
//         // Normalise
//         output
//             .iter_mut()
//             .for_each(|value| *value *= 1.0 / (size as f64));
//     }
// }

impl Fft<FftMatrixf32, FftMatrixc32> for f32 {
    fn rfft3_fftw_par_vec(input: &mut FftMatrixf32, output: &mut FftMatrixc32, shape: &[usize]) {
        let size: usize = shape.iter().product();
        let size_d = shape.last().unwrap();
        let size_real = (size / size_d) * (size_d / 2 + 1);
        let plan: R2CPlan32 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();

        let it_inp = input.par_chunks_exact_mut(size).into_par_iter();
        let it_out = output.par_chunks_exact_mut(size_real).into_par_iter();

        it_inp.zip(it_out).for_each(|(inp, out)| {
            let _ = plan.r2c(inp, out);
        });
    }

    fn irfft_fftw_par_vec(input: &mut FftMatrixc32, output: &mut FftMatrixf32, shape: &[usize]) {
        let size: usize = shape.iter().product();
        let size_d = shape.last().unwrap();
        let size_real = (size / size_d) * (size_d / 2 + 1);
        let plan: C2RPlan32 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();

        let it_inp = input.par_chunks_exact_mut(size_real).into_par_iter();
        let it_out = output.par_chunks_exact_mut(size).into_par_iter();

        it_inp.zip(it_out).for_each(|(inp, out)| {
            let _ = plan.c2r(inp, out);
            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (size as f32));
        })
    }

    fn rfft3_fftw(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]) {
        assert!(shape.len() == 3);
        let plan: R2CPlan32 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();
        let _ = plan.r2c(input, output);
    }

    fn irfft3_fftw(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]) {
        assert!(shape.len() == 3);
        let size: usize = shape.iter().product();
        let plan: C2RPlan32 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();
        let _ = plan.c2r(input, output);
        // Normalise
        output
            .iter_mut()
            .for_each(|value| *value *= 1.0 / (size as f32));
    }
}

impl Fft<FftMatrixf64, FftMatrixc64> for f64 {
    fn rfft3_fftw_par_vec(input: &mut FftMatrixf64, output: &mut FftMatrixc64, shape: &[usize]) {
        let size: usize = shape.iter().product();
        let size_d = shape.last().unwrap();
        let size_real = (size / size_d) * (size_d / 2 + 1);
        let plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();

        let it_inp = input.par_chunks_exact_mut(size).into_par_iter();
        let it_out = output.par_chunks_exact_mut(size_real).into_par_iter();

        it_inp.zip(it_out).for_each(|(inp, out)| {
            let _ = plan.r2c(inp, out);
        });
    }

    fn irfft_fftw_par_vec(input: &mut FftMatrixc64, output: &mut FftMatrixf64, shape: &[usize]) {
        let size: usize = shape.iter().product();
        let size_d = shape.last().unwrap();
        let size_real = (size / size_d) * (size_d / 2 + 1);
        let plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();

        let it_inp = input.par_chunks_exact_mut(size_real).into_par_iter();
        let it_out = output.par_chunks_exact_mut(size).into_par_iter();

        it_inp.zip(it_out).for_each(|(inp, out)| {
            let _ = plan.c2r(inp, out);
            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (size as f64));
        })
    }

    fn rfft3_fftw(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]) {
        assert!(shape.len() == 3);
        let plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();
        let _ = plan.r2c(input, output);
    }

    fn irfft3_fftw(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]) {
        assert!(shape.len() == 3);
        let size: usize = shape.iter().product();
        let plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();
        let _ = plan.c2r(input, output);
        // Normalise
        output
            .iter_mut()
            .for_each(|value| *value *= 1.0 / (size as f64));
    }
}
