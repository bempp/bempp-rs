//! Wrappers for FFTW functions, including multithreaded implementations.
use std::any::{Any, TypeId};

use cauchy::Scalar;
use fftw::{plan::*, types::*};
use num::Float;
use rayon::prelude::*;

use rlst::dense::{
    base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix, traits::*, Dynamic,
};

/// Type alias for real coefficients for into FFTW wrappers
pub type FftMatrixf64 = Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic>, Dynamic>;

/// Type alias for complex coefficients for FFTW wrappers
pub type FftMatrixc64 = Matrix<c64, BaseMatrix<c64, VectorContainer<c64>, Dynamic>, Dynamic>;

/// Compute a Real FFT of an input slice corresponding to a 3D array stored in column major format, specified by `shape` using the FFTW library.
///
/// # Arguments
/// * `input` - Input slice of real data, corresponding to a 3D array stored in column major order.
/// * `output` - Output slice.
/// * `shape` - Shape of input data.
pub fn rfft3_fftw64(input: &mut [f64], output: &mut [c64], shape: &[usize]) {
    assert!(shape.len() == 3);
    let plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();
    let _ = plan.r2c(input, output);
}

/// Compute a Real FFT of an input slice corresponding to a 3D array stored in column major format, specified by `shape` using the FFTW library.
///
/// # Arguments
/// * `input` - Input slice of real data, corresponding to a 3D array stored in column major order.
/// * `output` - Output slice.
/// * `shape` - Shape of input data.
pub fn rfft3_fftw32(input: &mut [f32], output: &mut [c32], shape: &[usize]) {
    assert!(shape.len() == 3);
    let plan: R2CPlan32 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();
    let _ = plan.r2c(input, output);
}

pub fn rfft3_fftw<R, C>(input: &mut [R::Real], output: &mut [C::Complex], shape: &[usize])
where
    R: Default + Scalar<Real = R> + Any,
    C: Default + Scalar<Complex = C> + Any,
{
    if TypeId::of::<R>() == TypeId::of::<f32>() && TypeId::of::<C>() == TypeId::of::<c32>() {
        let input_f32 =
            unsafe { std::slice::from_raw_parts_mut(input.as_mut_ptr() as *mut f32, input.len()) };
        let output_c32 = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut c32, output.len())
        };
        rfft3_fftw32(input_f32, output_c32, shape);
    } else if TypeId::of::<R>() == TypeId::of::<f64>() && TypeId::of::<C>() == TypeId::of::<c64>() {
        let input_f64 =
            unsafe { std::slice::from_raw_parts_mut(input.as_mut_ptr() as *mut f64, input.len()) };
        let output_c64 = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut c64, output.len())
        };
        rfft3_fftw64(input_f64, output_c64, shape);
    }
}

/// Compute an inverse Real FFT of an input slice corresponding to the FFT of a 3D array stored in column major format, specified by `shape` using the FFTW library.
/// This function normalises the output.
///
/// # Arguments
/// * `input` - Input slice of complex data, corresponding to an FFT of a 3D array stored in column major order.
/// * `output` - Output slice.
/// * `shape` - Shape of output data.
pub fn irfft3_fftw(input: &mut [c64], output: &mut [f64], shape: &[usize]) {
    assert!(shape.len() == 3);
    let size: usize = shape.iter().product();
    let plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();
    let _ = plan.c2r(input, output);
    // Normalise
    output
        .iter_mut()
        .for_each(|value| *value *= 1.0 / (size as f64));
}

/// Compute a Real FFT over a rlst matrix which stores data corresponding to multiple 3 dimensional arrays of shape `shape`, stored in column major order.
/// This function is multithreaded, and uses the FFTW library.
///
/// # Arguments
/// * `input` - Input slice of real data, corresponding to a 3D array stored in column major order.
/// * `output` - Output slice.
/// * `shape` - Shape of input data.
pub fn rfft3_fftw_par_vec(input: &mut FftMatrixf64, output: &mut FftMatrixc64, shape: &[usize]) {
    assert!(shape.len() == 3);

    let size: usize = shape.iter().product();
    let size_d = shape.last().unwrap();
    let size_real = (size / size_d) * (size_d / 2 + 1);

    let plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();
    let it_inp = input.data_mut().par_chunks_exact_mut(size).into_par_iter();
    let it_out = output
        .data_mut()
        .par_chunks_exact_mut(size_real)
        .into_par_iter();

    it_inp.zip(it_out).for_each(|(inp, out)| {
        let _ = plan.r2c(inp, out);
    });
}

/// Compute an inverse Real FFT over a rlst matrix which stores data corresponding to multiple 3 dimensional arrays of shape `shape`, stored in column major order.
/// This function is multithreaded, and uses the FFTW library.
///
/// # Arguments
/// * `input` - Input slice of complex data, corresponding to an FFT of a 3D array stored in column major order.
/// * `output` - Output slice.
/// * `shape` - Shape of output data.
pub fn irfft3_fftw_par_vec(input: &mut FftMatrixc64, output: &mut FftMatrixf64, shape: &[usize]) {
    assert!(shape.len() == 3);
    let size: usize = shape.iter().product();
    let size_d = shape.last().unwrap();
    let size_real = (size / size_d) * (size_d / 2 + 1);
    let plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();

    let it_inp = input
        .data_mut()
        .par_chunks_exact_mut(size_real)
        .into_par_iter();
    let it_out = output.data_mut().par_chunks_exact_mut(size).into_par_iter();

    it_inp.zip(it_out).for_each(|(inp, out)| {
        let _ = plan.c2r(inp, out);
        // Normalise output
        out.iter_mut()
            .for_each(|value| *value *= 1.0 / (size as f64));
    })
}
