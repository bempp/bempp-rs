use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use ndarray::*;
use ndarray_linalg::SVDDC;
use ndarray_ndimage::{pad, PadMode};
use num::Float;
use ndrustfft::{Complex, FftHandler, ndfft};

use bempp_traits::{field::FieldTranslationData, kernel::Kernel};
use bempp_tree::types::{domain::Domain, morton::MortonKey};

#[derive(Default)]
pub struct FftFieldTranslationNaiveKiFmm<T>
where
    T: Kernel + Default,
{
    // Maps between convolution and surface grids
    pub surf_to_conv_map: HashMap<usize, usize>,
    pub conv_to_surf_map: HashMap<usize, usize>,

    // Map from potentials to surface grid
    pub potentials_to_surf: ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,

    // Precomputed FFT of unique kernel interactions placed on
    // convolution grid.
    pub m2l: Vec<ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 3]>>>,

    // Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    pub kernel: T,
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

#[derive(Debug)]
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

impl<T> FieldTranslationData<T> for FftFieldTranslationNaiveKiFmm<T>
where
    T: Kernel + Default,
{
    type Domain = Domain;
    type M2LOperators = Vec<ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 3]>>>;
    type TransferVector = Vec<TransferVector>;

    fn compute_m2l_operators(
        &self,
        expansion_order: usize,
        domain: Self::Domain,
    ) -> Self::M2LOperators {
        
        let alpha_inner = 1.05;

        let mut result: Vec<ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 3]>>> = Vec::new();
        
        for t in self.transfer_vectors.iter() {

            let source_equivalent_surface = t
                .source
                .compute_surface(&domain, expansion_order, alpha_inner);
            
            let conv_grid_sources = t
                .source.convolution_grid(expansion_order, &domain, &source_equivalent_surface);
            
            let target_check_surface = t
                .target
                .compute_surface(&domain, expansion_order, alpha_inner);
            // Find min target
            let sums: Vec<f64> = target_check_surface.iter().map(|point| point.iter().sum()).collect_vec();
            let min_index = sums
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(index, _)| index)
                .unwrap();
            let min_target = target_check_surface[min_index];


            let kernel = self.compute_kernel(expansion_order, &conv_grid_sources, min_target);
            let m = kernel.len();
            let n = kernel[0].len();
            let k = kernel[0][0].len();

            // Precompute and store the FFT of each unique kernel interaction
            let kernel = Array3::from_shape_vec((m, n, k), kernel.into_iter().flatten().flatten().collect()).unwrap();

            // Begin by calculating pad lengths along each dimension
            let p = 2*m;
            let q = 2*n;
            let r = 2*k;

            let padding = [
                [0, p-m],
                [0, q-n],
                [0, r-k],
            ];

            let padded_kernel = pad(&kernel, &padding, PadMode::Constant(0.));

            // Map to complex for FFT
            let padded_kernel = padded_kernel.map(|&x| Complex::new(x, 0.0));

            let mut padded_kernel_hat: Array3<Complex<f64>> = Array3::zeros((p, q, r));

            // Compute FFT of kernel for this transfer vector
            { 
                // 1. Init the handlers for FFTs along each axis
                let mut handler_ax0 = FftHandler::<f64>::new(p);
                let mut handler_ax1 = FftHandler::<f64>::new(q);
                let mut handler_ax2 = FftHandler::<f64>::new(r);

                // 2. Compute the transform along each axis
                let mut tmp1: Array3<Complex<f64>> = Array3::zeros((p, q, r));
                ndfft(&padded_kernel, &mut tmp1, &mut handler_ax2, 2);
                let mut tmp2: Array3<Complex<f64>> = Array3::zeros((p, q, r));
                ndfft(&tmp1, &mut tmp2, &mut handler_ax1, 1);
                ndfft(&tmp2, &mut padded_kernel_hat, &mut handler_ax0, 0);

            }

            // Store FFT of kernel for this transfer vector
            {
                result.push(padded_kernel_hat);

            }

        }

        result
    }

    fn compute_transfer_vectors(&self) -> Self::TransferVector {
        compute_transfer_vectors()
    }

    fn ncoeffs(&self, expansion_order: usize) -> usize {
        6 * (expansion_order - 1).pow(2) + 2
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

impl<T> SvdFieldTranslationNaiveKiFmm<T>
where
    T: Kernel + Default,
{
    pub fn new(kernel: T, k: Option<usize>, expansion_order: usize, domain: Domain) -> Self {
        let mut result = SvdFieldTranslationNaiveKiFmm::default();

        if let Some(k) = k {
            // Compression rank <= number of coefficients
            let ncoeffs = result.ncoeffs(expansion_order);
            if k <= ncoeffs {
                result.k = k
            } else {
                result.k = ncoeffs;
            }
        } else {
            // TODO: Should be data driven if nothing is provided by the user
            result.k = 50;
        }

        result.kernel = kernel;
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
            // Compression rank <= number of coefficients
            let ncoeffs = result.ncoeffs(expansion_order);
            if k <= ncoeffs {
                result.k = k
            } else {
                result.k = ncoeffs;
            }
        } else {
            // TODO: Should be data driven if nothing is provided by the user
            result.k = 50;
        }

        result.kernel = kernel;
        result.transfer_vectors = result.compute_transfer_vectors();
        result.m2l = result.compute_m2l_operators(expansion_order, domain);

        result
    }
}

impl<T> FftFieldTranslationNaiveKiFmm<T>
where
    T: Kernel + Default,
{
    pub fn new(kernel: T, expansion_order: usize, domain: Domain) -> Self {
        let mut result = FftFieldTranslationNaiveKiFmm::default();

        // Create maps between surface and convolution grids
        let (surf_to_conv, conv_to_surf) =
            FftFieldTranslationNaiveKiFmm::<T>::compute_surf_to_conv_map(expansion_order);
        result.surf_to_conv_map = surf_to_conv;
        result.conv_to_surf_map = conv_to_surf;

        result.kernel = kernel;

        result.transfer_vectors = result.compute_transfer_vectors();
        result.m2l = result.compute_m2l_operators(expansion_order, domain);

        result
    }

    pub fn compute_surf_to_conv_map(
        expansion_order: usize,
    ) -> (HashMap<usize, usize>, HashMap<usize, usize>) {
        let n = 2 * expansion_order - 1;

        // Index maps between surface and convolution grids
        let mut surf_to_conv: HashMap<usize, usize> = HashMap::new();
        let mut conv_to_surf: HashMap<usize, usize> = HashMap::new();

        // Initialise surface grid index
        let mut surf_index = 0;

        // The boundaries of the surface grid
        let lower = expansion_order - 1;
        let upper = 2 * expansion_order - 2;

        // Iterate through the entire convolution grid marking the boundaries
        // This makes the map much easier to understand and debug
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let conv_idx = i * n * n + j * n + k;
                    if (i >= lower && j >= lower && k == lower)
                        || (i >= lower && j >= lower && k == upper)
                        || (j >= lower && k >= lower && i == upper)
                        || (j >= lower && k >= lower && i == lower)
                        || (k >= lower && i >= lower && j == lower)
                        || (k >= lower && i >= lower && j == upper)
                    {
                        surf_to_conv.insert(surf_index, conv_idx);
                        conv_to_surf.insert(conv_idx, surf_index);
                        surf_index += 1;
                    }
                }
            }
        }

        (surf_to_conv, conv_to_surf)
    }

    pub fn compute_kernel(
        &self,
        expansion_order: usize,
        convolution_grid: &Vec<[f64; 3]>,
        min_target: [f64; 3],
    ) -> Vec<Vec<Vec<f64>>> {
        let n = 2 * expansion_order - 1;
        let mut result = vec![vec![vec![0f64; n]; n]; n];

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let conv_idx = i * n * n + j * n + k;
                    let src = convolution_grid[conv_idx];
                    result[i][j][k] = self.kernel.kernel(&src[..], &min_target[..]);
                }
            }
        }
        result
    }

    pub fn compute_signal(
        &self,
        expansion_order: usize,
        convolution_grid: &Vec<[f64; 3]>,
        charges: &Vec<f64>,
    ) -> Vec<Vec<Vec<f64>>> {
        let n = 2 * expansion_order - 1;
        let mut result = vec![vec![vec![0f64; n]; n]; n];

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let conv_idx = i * n * n + j * n + k;
                    if self.conv_to_surf_map.contains_key(&conv_idx) {
                        let surf_idx = self.conv_to_surf_map.get(&conv_idx).unwrap();
                        result[i][j][k] = charges[*surf_idx]
                    }
                }
            }
        }

        result
    }
}
