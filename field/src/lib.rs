use std::collections::HashSet;

use itertools::Itertools;
use ndarray::*;
use ndarray_linalg::SVDDC;

use bempp_traits::{
    field::{FieldTranslation, FieldTranslationData},
    kernel::Kernel,
};
use bempp_tree::types::{
    domain::{self, Domain},
    morton::MortonKey,
};

#[derive(Default)]
pub struct FftFieldTranslationNaive {
    // Maps between convolution and surface grids
    surf_to_conv_map: bool,
    conv_to_surf_map: bool,

    // Map from potentials to surface grid
    potentials_to_surf: bool,

    // Precomputed FFT of unique kernel interactions placed on
    // convolution grid.
    m2l: bool,

    // Unique transfer vectors to lookup m2l unique kernel interactions
    transfer_vectors: Vec<TransferVector>,
}

#[derive(Default)]
pub struct FftFieldTranslation {
    // Maps between convolution and surface grids
    pub surf_to_conv_map: bool,
    pub conv_to_surf_map: bool,

    // Map from potentials to surface grid
    pub potentials_to_surf: bool,

    // Precomputed FFT of unique kernel interactions placed on
    // convolution grid.
    pub m2l: bool,

    // Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<usize>,
}

#[derive(Default)]
pub struct SvdFieldTranslationNaiveKiFmm<T>
where
    T: Kernel + Default,
{
    // Compression rank, if unspecified estimated from data.
    pub k: usize,

    // Precomputed SVD compressed m2l interaction
    pub m2l: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),

    // Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    pub kernel: T,
}

#[derive(Default)]
pub struct SvdFieldTranslationKiFmm<T>
where
    T: Kernel + Default,
{
    // Compression rank, if unspecified estimated from data.
    pub k: usize,

    // Precomputed SVD compressed m2l interaction
    pub m2l: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),

    // Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    pub kernel: T,
}

pub struct TransferVector {
    pub vector: usize,
    pub source: MortonKey,
    pub target: MortonKey,
}

/// Algebraically defined list of unique M2L interactions, called 'transfer vectors', for 3D FMM.
pub fn compute_transfer_vectors() -> Vec<TransferVector> {
    let point = [0.5, 0.5, 0.5];
    let domain = Domain {
        origin: [0., 0., 0.],
        diameter: [1., 1., 1.],
    };

    // Encode point in centre of domain
    let key = MortonKey::from_point(&point, &domain, 3);

    // Add neighbours, and their resp. siblings to v list.
    let mut neighbours = key.neighbors();
    let mut keys: Vec<MortonKey> = Vec::new();
    keys.push(key);
    keys.append(&mut neighbours);

    for key in neighbours.iter() {
        let mut siblings = key.siblings();
        keys.append(&mut siblings);
    }

    // Keep only unique keys
    let keys: Vec<&MortonKey> = keys.iter().unique().collect();

    let mut transfer_vectors: Vec<usize> = Vec::new();
    let mut targets: Vec<MortonKey> = Vec::new();
    let mut sources: Vec<MortonKey> = Vec::new();

    for key in keys.iter() {
        // Dense v_list
        let v_list = key
            .parent()
            .neighbors()
            .iter()
            .flat_map(|pn| pn.children())
            .filter(|pnc| !key.is_adjacent(pnc))
            .collect_vec();

        // Find transfer vectors for everything in dense v list of each key
        let tmp: Vec<usize> = v_list
            .iter()
            .map(|v| key.find_transfer_vector(v))
            .collect_vec();

        transfer_vectors.extend(&mut tmp.iter().cloned());
        sources.extend(&mut v_list.iter().cloned());

        let tmp_targets = vec![**key; tmp.len()];
        targets.extend(&mut tmp_targets.iter().cloned());
    }

    let mut unique_transfer_vectors = Vec::new();
    let mut unique_indices = HashSet::new();

    for (i, vec) in transfer_vectors.iter().enumerate() {
        if !unique_transfer_vectors.contains(vec) {
            unique_transfer_vectors.push(*vec);
            unique_indices.insert(i);
        }
    }

    let unique_sources: Vec<MortonKey> = sources
        .iter()
        .enumerate()
        .filter(|(i, _)| unique_indices.contains(i))
        .map(|(_, x)| *x)
        .collect_vec();

    let unique_targets: Vec<MortonKey> = targets
        .iter()
        .enumerate()
        .filter(|(i, _)| unique_indices.contains(i))
        .map(|(_, x)| *x)
        .collect_vec();

    let mut result = Vec::<TransferVector>::new();

    for ((t, s), v) in unique_targets
        .into_iter()
        .zip(unique_sources)
        .zip(unique_transfer_vectors)
    {
        result.push(TransferVector {
            vector: v,
            target: t,
            source: s,
        })
    }

    result
}

/// Scaling function for the M2L operator at a given level.
pub fn m2l_scale(level: u64) -> f64 {
    if level < 2 {
        panic!("M2L only performed on level 2 and below")
    }

    if level == 2 {
        1. / 2.
    } else {
        2_f64.powf((level - 3) as f64)
    }
}

// impl <T>FieldTranslationData<T> for FftFieldTranslation
// where
//     T: Kernel
// {

//     fn compute_transfer_vectors() {
//         let transfer_vectors = compute_transfer_vectors();

//     }

// }

impl<T> FieldTranslationData<T> for FftFieldTranslationNaive
where
    T: Kernel,
{
    type TransferVector = Vec<TransferVector>;
    type M2LOperators = bool;
    type Domain = Domain;

    fn compute_transfer_vectors(&self) -> Self::TransferVector {
        compute_transfer_vectors()
    }

    fn compute_m2l_operators(&self, expansion_order: usize, domain: Domain) -> Self::M2LOperators {
        true
    }

    fn ncoeffs(&self, expansion_order: usize) -> usize {
        1
    }
}

impl<T> FieldTranslationData<T> for SvdFieldTranslationKiFmm<T>
where
    T: Kernel + Default,
{
    type TransferVector = Vec<TransferVector>;
    type M2LOperators = (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    );
    type Domain = Domain;

    fn compute_transfer_vectors(&self) -> Self::TransferVector {
        compute_transfer_vectors()
    }

    fn ncoeffs(&self, expansion_order: usize) -> usize {
        6 * (expansion_order - 1).pow(2) + 2
    }

    fn compute_m2l_operators(
        &self,
        expansion_order: usize,
        domain: Self::Domain,
    ) -> Self::M2LOperators {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)
        // let (targets, sources, transfer_vectors) = find_unique_v_list_interactions();

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = self.ncoeffs(expansion_order);
        let ncols = self.ncoeffs(expansion_order);

        let alpha_inner = 1.05;

        let mut se2tc_fat: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
            Array2::zeros((nrows, ncols * self.transfer_vectors.len()));

        let mut se2tc_thin: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
            Array2::zeros((ncols * self.transfer_vectors.len(), nrows));

        for (i, t) in self.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface = t
                .source
                .compute_surface(&domain, expansion_order, alpha_inner)
                .into_iter()
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            let target_check_surface = t
                .target
                .compute_surface(&domain, expansion_order, alpha_inner)
                .into_iter()
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            let mut tmp_gram = Vec::new();
            self.kernel.gram(
                &source_equivalent_surface[..],
                &target_check_surface[..],
                &mut tmp_gram,
            );

            let tmp_gram = Array::from_shape_vec((nrows, ncols), tmp_gram).unwrap();
            let lidx_sources = i * ncols;
            let ridx_sources = lidx_sources + ncols;

            se2tc_fat
                .slice_mut(s![.., lidx_sources..ridx_sources])
                .assign(&tmp_gram);

            se2tc_thin
                .slice_mut(s![lidx_sources..ridx_sources, ..])
                .assign(&tmp_gram);
        }

        let left: usize = 0;
        let right: usize = std::cmp::min(self.k, nrows);

        let (u, sigma, vt) = se2tc_fat.svddc(ndarray_linalg::JobSvd::Some).unwrap();

        let u = u.unwrap().slice(s![.., left..right]).to_owned();
        let sigma = Array2::from_diag(&sigma.slice(s![left..right]));
        let vt = vt.unwrap().slice(s![left..right, ..]).to_owned();

        let (_r, _gamma, st) = se2tc_thin.svddc(ndarray_linalg::JobSvd::Some).unwrap();

        let st = st.unwrap().slice(s![left..right, ..]).to_owned();

        // Store compressed M2L operators
        let mut c = Array2::zeros((self.k, self.k * self.transfer_vectors.len()));
        for i in 0..self.transfer_vectors.len() {
            let v_lidx = i * ncols;
            let v_ridx = v_lidx + ncols;
            let vt_sub = vt.slice(s![.., v_lidx..v_ridx]);
            let tmp = sigma.dot(&vt_sub.dot(&st.t()));
            let lidx = i * self.k;
            let ridx = lidx + self.k;

            c.slice_mut(s![.., lidx..ridx]).assign(&tmp);
        }

        (u, st, c)
    }
}

impl<T> FieldTranslationData<T> for SvdFieldTranslationNaiveKiFmm<T>
where
    T: Kernel + Default,
{
    type TransferVector = Vec<TransferVector>;
    type M2LOperators = (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    );
    type Domain = Domain;

    fn compute_transfer_vectors(&self) -> Self::TransferVector {
        compute_transfer_vectors()
    }

    fn ncoeffs(&self, expansion_order: usize) -> usize {
        6 * (expansion_order - 1).pow(2) + 2
    }

    fn compute_m2l_operators(
        &self,
        expansion_order: usize,
        domain: Self::Domain,
    ) -> Self::M2LOperators {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)
        // let (targets, sources, transfer_vectors) = find_unique_v_list_interactions();

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = self.ncoeffs(expansion_order);
        let ncols = self.ncoeffs(expansion_order);

        let alpha_inner = 1.05;

        let mut se2tc_fat: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
            Array2::zeros((nrows, ncols * self.transfer_vectors.len()));

        let mut se2tc_thin: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
            Array2::zeros((ncols * self.transfer_vectors.len(), nrows));

        for (i, t) in self.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface = t
                .source
                .compute_surface(&domain, expansion_order, alpha_inner)
                .into_iter()
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            let target_check_surface = t
                .target
                .compute_surface(&domain, expansion_order, alpha_inner)
                .into_iter()
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            let mut tmp_gram = Vec::new();
            self.kernel.gram(
                &source_equivalent_surface[..],
                &target_check_surface[..],
                &mut tmp_gram,
            );

            let tmp_gram = Array::from_shape_vec((nrows, ncols), tmp_gram).unwrap();
            let lidx_sources = i * ncols;
            let ridx_sources = lidx_sources + ncols;

            se2tc_fat
                .slice_mut(s![.., lidx_sources..ridx_sources])
                .assign(&tmp_gram);

            se2tc_thin
                .slice_mut(s![lidx_sources..ridx_sources, ..])
                .assign(&tmp_gram);
        }

        let left: usize = 0;
        let right: usize = std::cmp::min(self.k, nrows);

        let (u, sigma, vt) = se2tc_fat.svddc(ndarray_linalg::JobSvd::Some).unwrap();

        let u = u.unwrap().slice(s![.., left..right]).to_owned();
        let sigma = Array2::from_diag(&sigma.slice(s![left..right]));
        let vt = vt.unwrap().slice(s![left..right, ..]).to_owned();

        let (_r, _gamma, st) = se2tc_thin.svddc(ndarray_linalg::JobSvd::Some).unwrap();

        let st = st.unwrap().slice(s![left..right, ..]).to_owned();

        // Store compressed M2L operators
        let mut c = Array2::zeros((self.k, self.k * self.transfer_vectors.len()));
        for i in 0..self.transfer_vectors.len() {
            let v_lidx = i * ncols;
            let v_ridx = v_lidx + ncols;
            let vt_sub = vt.slice(s![.., v_lidx..v_ridx]);
            let tmp = sigma.dot(&vt_sub.dot(&st.t()));
            let lidx = i * self.k;
            let ridx = lidx + self.k;

            c.slice_mut(s![.., lidx..ridx]).assign(&tmp);
        }

        (u, st, c)
    }
}

// impl FftFieldTranslation {
//     pub fn new() -> Self {
//         FftFieldTranslation::default()
//     }
// }

// impl SvdFieldTranslation {
//     pub fn new() -> Self {
//         SvdFieldTranslation::default()
//     }
// }

impl FftFieldTranslationNaive {
    pub fn new() -> Self {
        FftFieldTranslationNaive::default()
    }
}

impl<T> SvdFieldTranslationNaiveKiFmm<T>
where
    T: Kernel + Default,
{
    pub fn new(kernel: T, k: Option<usize>, expansion_order: usize, domain: Domain) -> Self {
        let mut result = SvdFieldTranslationNaiveKiFmm::default();

        if let Some(k) = k {
            result.k = k;
        } else {
            // TODO: Should be data driven
            result.k = 50;
        }

        result.transfer_vectors = result.compute_transfer_vectors();
        result.m2l = result.compute_m2l_operators(expansion_order, domain);

        result
    }
}

impl<T> SvdFieldTranslationKiFmm<T>
where
    T: Kernel + Default,
{
    pub fn new(kernel: T, k: Option<usize>, expansion_order: usize, domain: Domain) -> Self {
        let mut result = SvdFieldTranslationKiFmm::default();

        if let Some(k) = k {
            result.k = k;
        } else {
            // TODO: Should be data driven
            result.k = 50;
        }

        result.transfer_vectors = result.compute_transfer_vectors();
        result.m2l = result.compute_m2l_operators(expansion_order, domain);

        result
    }
}
