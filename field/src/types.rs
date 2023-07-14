use std::collections::HashMap;

use num::Complex;
use bempp_tools::Array3D;
use rlst::dense::{
    base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix, Dynamic,
};

use bempp_traits::kernel::Kernel;
use bempp_tree::types::morton::MortonKey;

// type FftM2LEntry = ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 3]>>;
pub type SvdM2lEntry =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

pub type FftM2lEntry = Array3D<Complex<f64>>; 

pub struct FftFieldTranslationNaiveKiFmm<T>
where
    T: Kernel + Default,
{
    // Amount to dilate inner check surface by
    pub alpha: f64,

    // Maps between convolution and surface grids
    pub surf_to_conv_map: HashMap<usize, usize>,
    pub conv_to_surf_map: HashMap<usize, usize>,

    // Precomputed FFT of unique kernel interactions placed on
    // convolution grid.
    pub m2l: Vec<FftM2lEntry>,

    // Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    pub kernel: T,
}

pub struct FftFieldTranslationKiFmm<T>
where
    T: Kernel + Default,
{
    // Amount to dilate inner check surface by
    pub alpha: f64,

    // Maps between convolution and surface grids
    pub surf_to_conv_map: HashMap<usize, usize>,
    pub conv_to_surf_map: HashMap<usize, usize>,

    // Precomputed FFT of unique kernel interactions placed on
    // convolution grid.
    pub m2l: Vec<FftM2lEntry>,

    // Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    pub kernel: T,
}

pub struct SvdFieldTranslationKiFmm<T>
where
    T: Kernel + Default,
{
    // Amount to dilate inner check surface by
    pub alpha: f64,

    // Compression rank, if unspecified estimated from data.
    pub k: usize,

    // Precomputed SVD compressed m2l interaction
    pub m2l: (SvdM2lEntry, SvdM2lEntry, SvdM2lEntry),

    // Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    pub kernel: T,
}

#[derive(Debug)]
pub struct TransferVector {
    pub vector: usize,
    pub source: MortonKey,
    pub target: MortonKey,
}
