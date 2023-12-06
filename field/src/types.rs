//! Types for storing field translation data.
use std::collections::HashMap;

use cauchy::{c32, c64};
use num::{Complex, Float};
use rlst_dense::{
    array::Array, base_array::BaseArray, data_container::VectorContainer, rlst_dynamic_array2,
};

use bempp_traits::kernel::Kernel;
use bempp_traits::types::Scalar;
use bempp_tree::types::morton::MortonKey;

/// Simple type alias for a 2D `Matrix<f64>`
pub type SvdM2lEntry<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// Simple type alias for pre-computed FFT of green's function evaluations computed for each transfer vector in a box's halo
/// Each index corresponds to a halo position, and contains 64 convolutions, one for each of a box's siblings with each child
/// of the halo box.
pub type FftKernelData<C> = Vec<Vec<C>>;

/// A type to store the M2L field translation meta-data and data for an FFT based sparsification in the kernel independent FMM.
pub struct FftFieldTranslationKiFmm<T, U>
where
    T: Default + Scalar<Real = T> + Float,
    U: Kernel<T = T> + Default,
{
    /// Amount to dilate inner check surface by
    pub alpha: T,

    /// Map between indices of surface convolution grid points.
    pub surf_to_conv_map: HashMap<usize, usize>,

    /// Map between indices of convolution and surface grid points.
    pub conv_to_surf_map: HashMap<usize, usize>,

    /// Precomputed data required for FFT compressed M2L interaction.
    pub operator_data: FftM2lOperatorData<Complex<T>>,

    /// Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    /// The associated kernel with this translation operator.
    pub kernel: U,
}

/// A type to store the M2L field translation meta-data  and datafor an SVD based sparsification in the kernel independent FMM.
pub struct SvdFieldTranslationKiFmm<T, U>
where
    T: Scalar<Real = T> + Float + Default,
    U: Kernel<T = T> + Default,
{
    /// Amount to dilate inner check surface by when computing operator.
    pub alpha: T,

    /// Maximum rank taken for SVD compression, if unspecified estimated from data.
    pub k: usize,

    /// Precomputed data required for SVD compressed M2L interaction.
    pub operator_data: SvdM2lOperatorData<T>,

    /// Unique transfer vectors to lookup m2l unique kernel interactions.
    pub transfer_vectors: Vec<TransferVector>,

    /// The associated kernel with this translation operator.
    pub kernel: U,
}

/// A type to store a transfer vector between a `source` and `target` Morton key.
#[derive(Debug)]
pub struct TransferVector {
    /// Three vector of components.
    pub components: [i64; 3],

    /// Unique identifier for transfer vector, for easy lookup.
    pub hash: usize,

    /// The `source` Morton key associated with this transfer vector.
    pub source: MortonKey,

    /// The `target` Morton key associated with this transfer vector.
    pub target: MortonKey,
}

#[derive(Default)]
pub struct FftM2lOperatorData<C> {
    // FFT of unique kernel evaluations for each transfer vector in a halo of a sibling set
    pub kernel_data: FftKernelData<C>,
    pub kernel_data_rearranged: FftKernelData<C>,
}

/// Container to store precomputed data required for SVD field translations.
/// See Fong & Darve (2009) for the definitions of 'fat' and 'thin' M2L matrices.
pub struct SvdM2lOperatorData<T>
where
    T: Scalar,
{
    /// Left singular vectors from SVD of fat M2L matrix.
    pub u: SvdM2lEntry<T>,

    /// Right singular vectors from SVD of thin M2L matrix, cutoff to a maximum rank of 'k'.
    pub st_block: SvdM2lEntry<T>,

    /// The quantity $C_{block} = \Sigma \cdot V^T_{block} S_{block} $, where $\Sigma$ is diagonal matrix of singular values
    /// from the SVD of the fat M2L matrix, $V^T_{block}$ is a is a block of the right singular vectors corresponding
    /// to each transfer vector from the same SVD, and $S_{block}$ is a block of the transposed right singular vectors
    /// from the SVD of the thin M2L matrix. $C$ is composed of $C_{block}$, with one for each unique transfer vector.
    pub c: SvdM2lEntry<T>,
}

impl<T> Default for SvdM2lOperatorData<T>
where
    T: Scalar,
{
    fn default() -> Self {
        SvdM2lOperatorData {
            u: rlst_dynamic_array2!(T, [1, 1]),
            st_block: rlst_dynamic_array2!(T, [1, 1]),
            c: rlst_dynamic_array2!(T, [1, 1]),
        }
    }
}

/// Type alias for real coefficients for into FFTW wrappers
// pub type FftMatrixf64 = Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic>, Dynamic>;
pub type FftMatrixf64 = Vec<f64>;

/// Type alias for real coefficients for into FFTW wrappers
// pub type FftMatrixf32 = Matrix<f32, BaseMatrix<f32, VectorContainer<f32>, Dynamic>, Dynamic>;
pub type FftMatrixf32 = Vec<f32>;

/// Type alias for complex coefficients for FFTW wrappers
// pub type FftMatrixc64 = Matrix<c64, BaseMatrix<c64, VectorContainer<c64>, Dynamic>, Dynamic>;
pub type FftMatrixc64 = Vec<c64>;

/// Type alias for complex coefficients for FFTW wrappers
// pub type FftMatrixc32 = Matrix<c32, BaseMatrix<c32, VectorContainer<c32>, Dynamic>, Dynamic>;
pub type FftMatrixc32 = Vec<c32>;

/// Type alias for real coefficients for into FFTW wrappers
// pub type FftMatrix<T> = Matrix<T, BaseMatrix<T, VectorContainer<T>, Dynamic>, Dynamic>;
pub type FftMatrix<T> = Vec<T>;
