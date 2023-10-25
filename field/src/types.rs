//! Types for storing field translation data.
use std::collections::HashMap;

use cauchy::c64;

use rlst::{
    common::traits::{Eval, NewLikeSelf},
    dense::{
        base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix, rlst_mat, Dynamic,
    },
};

use bempp_traits::kernel::Kernel;
use bempp_tree::types::morton::MortonKey;

/// Simple alias for a 2D Matrix<f64>
pub type SvdM2lEntry =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

/// Type alias for pre-computed FFT of green's function evaluations corresponding a given transfer vector.
pub type FftKernelData = Vec<Vec<c64>>;

/// A type to store the M2L field translation meta-data and data for an FFT based sparsification in the kernel independent FMM.
pub struct FftFieldTranslationKiFmm<T>
where
    T: Kernel + Default,
{
    /// Amount to dilate inner check surface by
    pub alpha: f64,

    /// Map between indices of surface convolution grid points.
    pub surf_to_conv_map: HashMap<usize, usize>,

    /// Map between indices of convolution and surface grid points.
    pub conv_to_surf_map: HashMap<usize, usize>,

    /// Precomputed data required for FFT compressed M2L interaction.
    pub operator_data: FftM2lOperatorData,

    /// Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    /// The associated kernel with this translation operator.
    pub kernel: T,
}

/// A type to store the M2L field translation meta-data  and datafor an SVD based sparsification in the kernel independent FMM.
pub struct SvdFieldTranslationKiFmm<T>
where
    T: Kernel + Default,
{
    /// Amount to dilate inner check surface by when computing operator.
    pub alpha: f64,

    /// Maximum rank taken for SVD compression, if unspecified estimated from data.
    pub k: usize,

    /// Precomputed data required for SVD compressed M2L interaction.
    pub operator_data: SvdM2lOperatorData,

    /// Unique transfer vectors to lookup m2l unique kernel interactions.
    pub transfer_vectors: Vec<TransferVector>,

    /// The associated kernel with this translation operator.
    pub kernel: T,
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
pub struct FftM2lOperatorData {
    // FFT of unique kernel evaluations for each transfer vector in a halo of a sibling set
    pub kernel_data: FftKernelData,
    pub kernel_data_rearranged: FftKernelData,
}

/// Container to store precomputed data required for SVD field translations.
/// See Fong & Darve (2009) for the definitions of 'fat' and 'thin' M2L matrices.
pub struct SvdM2lOperatorData {
    /// Left singular vectors from SVD of fat M2L matrix.
    pub u: SvdM2lEntry,

    /// Right singular vectors from SVD of thin M2L matrix, cutoff to a maximum rank of 'k'.
    pub st_block: SvdM2lEntry,

    /// The quantity $C_{block} = \Sigma \cdot V^T_{block} S_{block} $, where $\Sigma$ is diagonal matrix of singular values
    /// from the SVD of the fat M2L matrix, $V^T_{block}$ is a is a block of the right singular vectors corresponding
    /// to each transfer vector from the same SVD, and $S_{block}$ is a block of the transposed right singular vectors
    /// from the SVD of the thin M2L matrix. $C$ is composed of $C_{block}$, with one for each unique transfer vector.
    pub c: SvdM2lEntry,
}

impl Default for SvdM2lOperatorData {
    fn default() -> Self {
        let tmp = rlst_mat![f64, (1, 1)];

        SvdM2lOperatorData {
            u: tmp.new_like_self().eval(),
            st_block: tmp.new_like_self().eval(),
            c: tmp.new_like_self().eval(),
        }
    }
}
