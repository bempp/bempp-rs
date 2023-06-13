use bempp_traits::field::{FieldTranslationData, FieldTranslation};

#[derive(Default)]
pub struct FftFieldTranslation {

    // Maps between convolution and surface grids
    surf_to_conv_map: bool,
    conv_to_surf_map: bool,

    // Map from potentials to surface grid
    potentials_to_surf: bool,

    // Precomputed FFT of unique kernel interactions placed on
    // convolution grid.
    m2l: bool,

    // Unique transfer vectors to lookup m2l unique kernel interactions
    transfer_vectors: bool,    
}


#[derive(Default)]
pub struct SvdFieldTranslation {

    // Compression rank, if unspecified estimated from data.
    k: bool,
    
    // Precomputed SVD compressed m2l interaction
    m2l: bool, 

    // Unique transfer vectors to lookup m2l unique kernel interactions
    transfer_vectors: bool,
}


impl FieldTranslationData for FftFieldTranslation {

    fn new() -> Self {
        FftFieldTranslation::default()
    }

    fn compute_transfer_vectors() {}
    fn compute_m2l_data() {}

}


impl FieldTranslationData for SvdFieldTranslation {

    fn new() -> Self {
        SvdFieldTranslation::default()
    }

    fn compute_transfer_vectors() {}
    fn compute_m2l_data() {}

}

