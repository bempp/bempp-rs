//! Wrappers for FFTW functions, including multithreaded implementations.
use fftw::{
    plan::{C2RPlan, C2RPlan32, C2RPlan64, R2CPlan, R2CPlan32, R2CPlan64},
    types::Flag,
};
use num::Complex;
use rayon::prelude::*;

/// Interface for taking FFTs over containers of this type using the FFTW library.
pub trait Fft
where
    Self: Sized,
{
    /// Compute a parallel real to complex FFT over a slice which stores data corresponding to multiple 3 dimensional arrays of shape `shape`, stored in column major order.
    /// This function is multithreaded, and uses the FFTW library.
    ///
    /// # Arguments
    /// * `input` - Input slice of real data, corresponding to a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of input data.
    fn rfft3_fftw_par_slice(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]);

    /// Compute a real to complex FFT over a slice which stores data corresponding to multiple 3 dimensional arrays of shape `shape`, stored in column major order.
    /// This function is multithreaded, and uses the FFTW library.
    ///
    /// # Arguments
    /// * `input` - Input slice of real data, corresponding to a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of input data.
    fn rfft3_fftw_slice(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]);

    /// Compute an parallel complex to real inverse FFT over a slice which stores data corresponding to multiple 3 dimensional arrays of shape `shape`, stored in column major order.
    /// This function is multithreaded, and uses the FFTW library.
    ///
    /// # Arguments
    /// * `input` - Input slice of complex data, corresponding to an FFT of a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of output data.
    fn irfft3_fftw_par_slice(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]);

    /// Compute an complex to real inverse FFT over a rlst matrix which stores data corresponding to multiple 3 dimensional arrays of shape `shape`, stored in column major order.
    /// This function is multithreaded, and uses the FFTW library.
    ///
    /// # Arguments
    /// * `input` - Input slice of complex data, corresponding to an FFT of a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of output data.
    fn irfft3_fftw_slice(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]);

    /// Compute a real to complex FFT of an input slice corresponding to a 3D array stored in column major format, specified by `shape` using the FFTW library.
    ///
    /// # Arguments
    /// * `input` - Input slice of real data, corresponding to a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of input data.
    fn rfft3_fftw(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]);

    /// Compute an complex to real inverse FFT of an input slice corresponding to the FFT of a 3D array stored in column major format, specified by `shape` using the FFTW library.
    /// This function normalises the output.
    ///
    /// # Arguments
    /// * `input` - Input slice of complex data, corresponding to an FFT of a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of output data.
    fn irfft3_fftw(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]);
}

impl Fft for f32 {
    fn rfft3_fftw_par_slice(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]) {
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

    fn irfft3_fftw_par_slice(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]) {
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

    fn rfft3_fftw_slice(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]) {
        let size: usize = shape.iter().product();
        let size_d = shape.last().unwrap();
        let size_real = (size / size_d) * (size_d / 2 + 1);
        let plan: R2CPlan32 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();

        let it_inp = input.chunks_exact_mut(size);
        let it_out = output.chunks_exact_mut(size_real);

        it_inp.zip(it_out).for_each(|(inp, out)| {
            let _ = plan.r2c(inp, out);
        });
    }

    fn irfft3_fftw_slice(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]) {
        let size: usize = shape.iter().product();
        let size_d = shape.last().unwrap();
        let size_real = (size / size_d) * (size_d / 2 + 1);
        let plan: C2RPlan32 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();

        let it_inp = input.chunks_exact_mut(size_real);
        let it_out = output.chunks_exact_mut(size);

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

impl Fft for f64 {
    fn rfft3_fftw_par_slice(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]) {
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

    fn irfft3_fftw_par_slice(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]) {
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

    fn rfft3_fftw_slice(input: &mut [Self], output: &mut [Complex<Self>], shape: &[usize]) {
        let size: usize = shape.iter().product();
        let size_d = shape.last().unwrap();
        let size_real = (size / size_d) * (size_d / 2 + 1);
        let plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();

        let it_inp = input.chunks_exact_mut(size);
        let it_out = output.chunks_exact_mut(size_real);

        it_inp.zip(it_out).for_each(|(inp, out)| {
            let _ = plan.r2c(inp, out);
        });
    }

    fn irfft3_fftw_slice(input: &mut [Complex<Self>], output: &mut [Self], shape: &[usize]) {
        let size: usize = shape.iter().product();
        let size_d = shape.last().unwrap();
        let size_real = (size / size_d) * (size_d / 2 + 1);
        let plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();

        let it_inp = input.chunks_exact_mut(size_real);
        let it_out = output.chunks_exact_mut(size);

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
