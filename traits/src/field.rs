//! Traits for multipole-to-local (M2L) field translations

pub trait PrecompTransData {
    fn compute_transfer_vectors();
    
    fn compute_m2l_data();
}

// Designed to be implemented by DataTrees which contain the required data
// for computing field translations
pub trait FieldTranslation {
    
    // How to scale operator with tree level.
    fn scale(&self, level: u64);
    
    // Convolution operation over each level.
    fn m2l(&self, level: u64);
}
