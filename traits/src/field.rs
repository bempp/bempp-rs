//! Traits for multipole-to-local (M2L) field translations

pub trait PrecompTransData {
    fn compute_transfer_vectors();
}

pub trait PrecompSvdTransData: PrecompTransData {
    fn compute_m2l_data();
}

pub trait PrecompFftTransData: PrecompTransData {
    fn compute_m2l_data();
}

// Designed to be implemented by DataTrees which contain the required data
// for computing field translations
pub trait SvdTranslation {
    
    // How to scale operator with tree level.
    fn scale(&self, level: u64);
    
    // Convolution operation over each level.
    fn m2l(&self, level: u64);
}

pub trait FftTranslation {

    // How to scale operator with tree level.
    fn scale(&self, level: u64);
    
    // Convolution operation over each level.
    fn m2l(&self, level: u64);
}