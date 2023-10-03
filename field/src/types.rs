use std::collections::HashMap;

use bempp_tools::Array3D;
use num::Complex;
use rlst::dense::{
    base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix, Dynamic,
};

use bempp_traits::kernel::Kernel;
use bempp_tree::types::morton::MortonKey;

/// Simple alias for a 2D Matrix<f64>
pub type SvdM2lEntry =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

/// Simple alias for an Array3D<Complexf64>
pub type FftM2lEntry = Array3D<Complex<f64>>;

/// A type to store the M2L field translation meta-data for an FFT based sparsification in the kernel independent FMM.
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

    /// Precomputed FFT of unique kernel interactions placed on convolution grid.
    pub m2l: Vec<FftM2lEntry>,

    /// Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    /// Map between redundant and unique transfer vectors when in checksum form
    pub transfer_vector_map: HashMap<usize, usize>,

    /// Maps between grid position indices from surface grids, to the reflected surface grids corresponding to unique transfer vectors.
    pub surf_grid_maps: Vec<HashMap<usize, usize>>,

    /// Inverse maps between grid position indices from reflected surface grids, to the un-reflected surface grids corresponding to non-unique transfer vectors.
    pub inv_surf_grid_maps: Vec<HashMap<usize, usize>>,

    /// Maps between grid position indices from convolution grids, to the reflected convolution grids corresponding to unique transfer vectors.
    pub conv_grid_maps: Vec<HashMap<usize, usize>>,

    /// Inverse maps between grid position indices from reflected convolution grids, to the un-reflected convolution grids corresponding to non-unique transfer vectors.
    pub inv_conv_grid_maps: Vec<HashMap<usize, usize>>,

    /// The associated kernel with this translation operator.
    pub kernel: T,
}

/// A type to store the M2L field translation meta-data for an SVD based sparsification in the kernel independent FMM.
pub struct SvdFieldTranslationKiFmm<T>
where
    T: Kernel + Default,
{
    /// Amount to dilate inner check surface by when computing operator.
    pub alpha: f64,

    /// Maximum rank taken for SVD compression, if unspecified estimated from data.
    pub k: usize,

    /// Precomputed SVD compressed M2L interaction.
    pub m2l: (SvdM2lEntry, SvdM2lEntry, SvdM2lEntry),

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
