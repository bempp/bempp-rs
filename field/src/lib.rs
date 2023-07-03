use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use ndarray::*;
use ndarray_linalg::SVDDC;
use ndarray_ndimage::{pad, PadMode};
use ndrustfft::{ndfft, ndfft_r2c, Complex, FftHandler, R2cFftHandler};

use rlst;
use rlst::algorithms::linalg::LinAlg;
use rlst::algorithms::traits::svd::{Mode, Svd};
use rlst::common::traits::{NewLikeSelf, NewLikeTranspose, Transpose};
use rlst::common::{
    tools::PrettyPrint,
    traits::{Copy, Eval},
};
use rlst::dense::{base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix};
use rlst::dense::{rlst_fixed_mat, rlst_mat, rlst_pointer_mat, traits::*, Dot, Shape};

use bempp_traits::{
    field::FieldTranslationData,
    kernel::{EvalType, Kernel, KernelType},
    types::Scalar,
};
use bempp_tree::types::{domain::Domain, morton::MortonKey};

type FftM2LEntry = ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 3]>>;
type SvdM2lEntry =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

#[derive(Default)]
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
    pub m2l: Vec<FftM2LEntry>,

    // Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    pub kernel: T,
}

// #[derive(Default)]
pub struct SvdFieldTranslationNaiveKiFmm<T>
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

// #[derive(Default)]
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

// impl<T> FieldTranslationData<T> for FftFieldTranslationNaiveKiFmm<T>
// where
//     T: Kernel + Default,
// {
//     type Domain = Domain;
//     type M2LOperators = Vec<ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 3]>>>;
//     type TransferVector = Vec<TransferVector>;

//     fn compute_m2l_operators(
//         &self,
//         expansion_order: usize,
//         domain: Self::Domain,
//     ) -> Self::M2LOperators {
//         type TranslationType = ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 3]>>;
//         let mut result: Vec<TranslationType> = Vec::new();

//         for t in self.transfer_vectors.iter() {
//             let source_equivalent_surface =
//                 t.source
//                     .compute_surface(&domain, expansion_order, self.alpha);

//             let conv_grid_sources = t.source.convolution_grid(
//                 expansion_order,
//                 &domain,
//                 &source_equivalent_surface,
//                 self.alpha,
//             );

//             let target_check_surface = t.target.compute_surface(&domain, expansion_order, self.alpha);

//             // TODO: Remove dim
//             let dim = 3;
//             // Find min target
//             let ncoeffs: usize = target_check_surface.len() / dim;
//             let sums: Vec<_> = (0..ncoeffs)
//                 .map(|i| target_check_surface[i] + target_check_surface[ncoeffs + i] + target_check_surface[2*ncoeffs + i])
//                 .collect();

//             let min_index = sums
//                 .iter()
//                 .enumerate()
//                 .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
//                 .map(|(index, _)| index)
//                 .unwrap();

//             let min_target = [
//                 target_check_surface[min_index],
//                 target_check_surface[min_index + ncoeffs],
//                 target_check_surface[min_index + 2 * ncoeffs],
//             ];

//             // TODO: Fix compute_kernel to work with new kernel
//             let kernel = self.compute_kernel(expansion_order, &conv_grid_sources, min_target);
//             let m = kernel.len();
//             let n = kernel[0].len();
//             let k = kernel[0][0].len();

//             // Precompute and store the FFT of each unique kernel interaction
//             let kernel =
//                 Array3::from_shape_vec((m, n, k), kernel.into_iter().flatten().flatten().collect())
//                     .unwrap();

//             // Begin by calculating pad lengths along each dimension
//             let p = 2 * m;
//             let q = 2 * n;
//             let r = 2 * k;

//             let padding = [[0, p - m], [0, q - n], [0, r - k]];

//             let padded_kernel = pad(&kernel, &padding, PadMode::Constant(0.));

//             // Flip the kernel
//             let padded_kernel = padded_kernel.slice(s![..;-1,..;-1,..;-1]).to_owned();
//             let mut padded_kernel_hat: Array3<Complex<f64>> = Array3::zeros((p, q, r / 2 + 1));

//             // Compute FFT of kernel for this transfer vector
//             {
//                 // 1. Init the handlers for FFTs along each axis
//                 let mut handler_ax0 = FftHandler::<f64>::new(p);
//                 let mut handler_ax1 = FftHandler::<f64>::new(q);
//                 let mut handler_ax2 = R2cFftHandler::<f64>::new(r);

//                 // 2. Compute the transform along each axis
//                 let mut tmp1: Array3<Complex<f64>> = Array3::zeros((p, q, r / 2 + 1));
//                 ndfft_r2c(&padded_kernel, &mut tmp1, &mut handler_ax2, 2);
//                 let mut tmp2: Array3<Complex<f64>> = Array3::zeros((p, q, r / 2 + 1));
//                 ndfft(&tmp1, &mut tmp2, &mut handler_ax1, 1);
//                 ndfft(&tmp2, &mut padded_kernel_hat, &mut handler_ax0, 0);
//             }

//             // Store FFT of kernel for this transfer vector
//             {
//                 result.push(padded_kernel_hat);
//             }
//         }

//         result
//     }

//     fn compute_transfer_vectors(&self) -> Self::TransferVector {
//         compute_transfer_vectors()
//     }

//     fn ncoeffs(&self, expansion_order: usize) -> usize {
//         6 * (expansion_order - 1).pow(2) + 2
//     }
// }

impl<T> FieldTranslationData<T> for SvdFieldTranslationKiFmm<T>
where
    T: Kernel<T = f64> + Default,
{
    type TransferVector = Vec<TransferVector>;
    type M2LOperators = (SvdM2lEntry, SvdM2lEntry, SvdM2lEntry);
    type Domain = Domain;

    fn compute_transfer_vectors(&self) -> Self::TransferVector {
        compute_transfer_vectors()
    }

    fn ncoeffs(&self, expansion_order: usize) -> usize {
        6 * (expansion_order - 1).pow(2) + 2
    }

    fn compute_m2l_operators<'a>(
        &self,
        expansion_order: usize,
        domain: Self::Domain,
    ) -> Self::M2LOperators {
        // ){
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = self.ncoeffs(expansion_order);
        let ncols = self.ncoeffs(expansion_order);

        // let mut se2tc_fat: SvdM2lEntry =
        //     Array2::zeros((nrows, ncols * self.transfer_vectors.len()));
        let ntransfer_vectors = self.transfer_vectors.len();
        let mut se2tc_fat = rlst_mat![f64, (nrows, ncols * ntransfer_vectors)];

        // let mut se2tc_thin: SvdM2lEntry =
        //     Array2::zeros((ncols * self.transfer_vectors.len(), nrows));
        let mut se2tc_thin = rlst_mat![f64, (nrows * ntransfer_vectors, ncols)];

        for (i, t) in self.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface =
                t.source
                    .compute_surface(&domain, expansion_order, self.alpha);
            let nsources = source_equivalent_surface.len() / self.kernel.space_dimension();
            let source_equivalent_surface = unsafe {
                rlst_pointer_mat!['a, f64, source_equivalent_surface.as_ptr(), (nsources, self.kernel.space_dimension()), (1, nsources)]
            };

            let target_check_surface =
                t.target
                    .compute_surface(&domain, expansion_order, self.alpha);
            let ntargets = target_check_surface.len() / self.kernel.space_dimension();
            let target_check_surface = unsafe {
                rlst_pointer_mat!['a, f64, target_check_surface.as_ptr(), (ntargets, self.kernel.space_dimension()), (1, ntargets)]
            };

            let mut tmp_gram = rlst_mat![f64, (ntargets, nsources)];

            self.kernel.gram(
                EvalType::Value,
                source_equivalent_surface.data(),
                target_check_surface.data(),
                tmp_gram.data_mut(),
            );

            // // let tmp_gram = Array::from_shape_vec((nrows, ncols), tmp_gram).unwrap();
            let lidx_sources = i * ncols;
            let ridx_sources = lidx_sources + ncols;

            let block_size = nrows * ncols;
            let start_idx = i * block_size;
            let end_idx = start_idx + block_size;
            let mut block = se2tc_fat.get_slice_mut(start_idx, end_idx);
            block.copy_from_slice(tmp_gram.data_mut());

            // // se2tc_fat
            // //     .slice_mut(s![.., lidx_sources..ridx_sources])
            // //     .assign(&tmp_gram);

            // se2tc_fat.block_mut((0, lidx_sources), tmp_gram.shape());
            for j in 0..ncols {
                let start_idx = j * ntransfer_vectors * nrows + i * nrows;
                let end_idx = start_idx + nrows;
                let mut block_column = se2tc_thin.get_slice_mut(start_idx, end_idx);
                let mut gram_column = tmp_gram.get_slice_mut(j * ncols, j * ncols + ncols);
                block_column.copy_from_slice(gram_column);
            }
        }

        let left: usize = 0;
        let right: usize = self.k;
        let (sigma, u, vt) = se2tc_fat.linalg().svd(Mode::All, Mode::Slim).unwrap();

        let u = u.unwrap();
        let vt = vt.unwrap();

        // Keep 'k' singular values
        let mut sigma_mat = rlst_mat![f64, (self.k, self.k)];
        for i in 0..self.k {
            sigma_mat[[i, i]] = sigma[i]
        }

        let (mu, nu) = u.shape();
        let u = u.block((0, 0), (mu, self.k)).eval();

        let (mvt, nvt) = vt.shape();
        let vt = vt.block((0, 0), (right, nvt)).eval();
        // println!("u {:?} {:?} {:?}", u.shape(), sigma_mat.shape(), vt.shape());

        // let (u, sigma, vt) = se2tc_fat.svddc(ndarray_linalg::JobSvd::Some).unwrap();
        // let u = u.unwrap().slice(s![.., left..right]).to_owned();
        // let sigma = Array2::from_diag(&sigma.slice(s![left..right]));
        // let vt = vt.unwrap().slice(s![left..right, ..]).to_owned();

        let (_gamma, _r, st) = se2tc_thin.linalg().svd(Mode::All, Mode::All).unwrap();
        let st = st.unwrap();
        let (mst, nst) = st.shape();
        let st_block = st.block((0, 0), (self.k, nst));
        let s = st_block.transpose().eval();
        let st = st.block((0, 0), (self.k, nst)).eval();

        // let (_r, _gamma, st) = se2tc_thin.svddc(ndarray_linalg::JobSvd::Some).unwrap();
        // let st = st.unwrap().slice(s![left..right, ..]).to_owned();

        // Store compressed M2L operators
        // let mut c = Array2::zeros((self.k, self.k * self.transfer_vectors.len()));
        let mut c = rlst_mat![f64, (self.k, self.k * ntransfer_vectors)];

        for i in 0..self.transfer_vectors.len() {
            // let v_lidx = i * ncols;
            // let v_ridx = v_lidx + ncols;
            // let vt_sub = vt.slice(s![.., v_lidx..v_ridx]);

            // let block_size = right*ncols;
            // let start_idx = i * block_size;
            // let end_idx = start_idx+block_size;
            let top_left = (0, i * ncols);
            let dim = (self.k, ncols);
            let vt_block = vt.block(top_left, dim);

            let tmp = sigma_mat.dot(&vt_block.dot(&s));
            //     let tmp = sigma.dot(&vt_sub.dot(&st.t()));
            //     let lidx = i * self.k;
            //     let ridx = lidx + self.k;

            let top_left = (0, i * self.k);
            let dim = (self.k, ncols);
            // let mut c_block =;
            c.block_mut(top_left, dim)
                .data_mut()
                .copy_from_slice(tmp.data());

            //     c.slice_mut(s![.., lidx..ridx]).assign(&tmp);
        }

        (u, st, c)
        // assert!(false)
    }
}

// impl<T> FieldTranslationData<T> for SvdFieldTranslationNaiveKiFmm<T>
// where
//     T: Kernel + Default,
// {
//     type TransferVector = Vec<TransferVector>;
//     type M2LOperators = (SvdM2lEntry, SvdM2lEntry, SvdM2lEntry);
//     type Domain = Domain;

//     fn compute_transfer_vectors(&self) -> Self::TransferVector {
//         compute_transfer_vectors()
//     }

//     fn ncoeffs(&self, expansion_order: usize) -> usize {
//         6 * (expansion_order - 1).pow(2) + 2
//     }

//     fn compute_m2l_operators(
//         &self,
//         expansion_order: usize,
//         domain: Self::Domain,
//     ) -> Self::M2LOperators {
//         // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)

//         // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
//         let nrows = self.ncoeffs(expansion_order);
//         let ncols = self.ncoeffs(expansion_order);

//         let mut se2tc_fat: SvdM2lEntry =
//             Array2::zeros((nrows, ncols * self.transfer_vectors.len()));

//         let mut se2tc_thin: SvdM2lEntry =
//             Array2::zeros((ncols * self.transfer_vectors.len(), nrows));

//         for (i, t) in self.transfer_vectors.iter().enumerate() {
//             let source_equivalent_surface =
//                 t.source.compute_surface(&domain, expansion_order, self.alpha);

//             let target_check_surface =
//                 t.target.compute_surface(&domain, expansion_order, self.alpha);

//             let mut tmp_gram = Vec::new();
//             self.kernel.gram(
//                 &source_equivalent_surface[..],
//                 &target_check_surface[..],
//                 &mut tmp_gram,
//             );

//             let tmp_gram = Array::from_shape_vec((nrows, ncols), tmp_gram).unwrap();
//             let lidx_sources = i * ncols;
//             let ridx_sources = lidx_sources + ncols;

//             se2tc_fat
//                 .slice_mut(s![.., lidx_sources..ridx_sources])
//                 .assign(&tmp_gram);

//             se2tc_thin
//                 .slice_mut(s![lidx_sources..ridx_sources, ..])
//                 .assign(&tmp_gram);
//         }

//         let left: usize = 0;
//         let right: usize = std::cmp::min(self.k, nrows);

//         let (u, sigma, vt) = se2tc_fat.svddc(ndarray_linalg::JobSvd::Some).unwrap();

//         let u = u.unwrap().slice(s![.., left..right]).to_owned();
//         let sigma = Array2::from_diag(&sigma.slice(s![left..right]));
//         let vt = vt.unwrap().slice(s![left..right, ..]).to_owned();

//         let (_r, _gamma, st) = se2tc_thin.svddc(ndarray_linalg::JobSvd::Some).unwrap();

//         let st = st.unwrap().slice(s![left..right, ..]).to_owned();

//         // Store compressed M2L operators
//         let mut c = Array2::zeros((self.k, self.k * self.transfer_vectors.len()));
//         for i in 0..self.transfer_vectors.len() {
//             let v_lidx = i * ncols;
//             let v_ridx = v_lidx + ncols;
//             let vt_sub = vt.slice(s![.., v_lidx..v_ridx]);
//             let tmp = sigma.dot(&vt_sub.dot(&st.t()));
//             let lidx = i * self.k;
//             let ridx = lidx + self.k;

//             c.slice_mut(s![.., lidx..ridx]).assign(&tmp);
//         }

//         (u, st, c)
//     }
// }

// impl<T> SvdFieldTranslationNaiveKiFmm<T>
// where
//     T: Kernel + Default,
// {
//     pub fn new(
//         kernel: T,
//         k: Option<usize>,
//         expansion_order: usize,
//         domain: Domain,
//         alpha: f64,
//     ) -> Self {
//         let mut result = SvdFieldTranslationNaiveKiFmm::default();

//         if let Some(k) = k {
//             // Compression rank <= number of coefficients
//             let ncoeffs = result.ncoeffs(expansion_order);
//             if k <= ncoeffs {
//                 result.k = k
//             } else {
//                 result.k = ncoeffs;
//             }
//         } else {
//             // TODO: Should be data driven if nothing is provided by the user
//             result.k = 50;
//         }

//         result.alpha = alpha;
//         result.kernel = kernel;
//         result.transfer_vectors = result.compute_transfer_vectors();
//         result.m2l = result.compute_m2l_operators(expansion_order, domain);

//         result
//     }
// }

impl<T> SvdFieldTranslationKiFmm<T>
where
    T: Kernel<T = f64> + Default,
{
    pub fn new(
        kernel: T,
        k: Option<usize>,
        expansion_order: usize,
        domain: Domain,
        alpha: f64,
    ) -> Self {
        let dummy = rlst_mat![f64, (1, 1)];

        // TODO: There should be a default for matrices to make code cleaner.
        let mut result = SvdFieldTranslationKiFmm {
            alpha,
            k: 100,
            kernel,
            m2l: (
                dummy.new_like_self().eval(),
                dummy.new_like_self().eval(),
                dummy.new_like_self().eval(),
            ),
            transfer_vectors: Vec::new(),
        };

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

        result.transfer_vectors = result.compute_transfer_vectors();
        result.m2l = result.compute_m2l_operators(expansion_order, domain);

        result
    }
}

// impl<T> FftFieldTranslationNaiveKiFmm<T>
// where
//     T: Kernel + Default,
// {
//     pub fn new(kernel: T, expansion_order: usize, domain: Domain, alpha: f64) -> Self {
//         let mut result = FftFieldTranslationNaiveKiFmm::default();

//         // Create maps between surface and convolution grids
//         let (surf_to_conv, conv_to_surf) =
//             FftFieldTranslationNaiveKiFmm::<T>::compute_surf_to_conv_map(expansion_order);
//         result.surf_to_conv_map = surf_to_conv;
//         result.conv_to_surf_map = conv_to_surf;

//         result.kernel = kernel;

//         result.alpha = alpha;
//         result.transfer_vectors = result.compute_transfer_vectors();
//         result.m2l = result.compute_m2l_operators(expansion_order, domain);

//         result
//     }

//     pub fn compute_surf_to_conv_map(
//         expansion_order: usize,
//     ) -> (HashMap<usize, usize>, HashMap<usize, usize>) {
//         let n = 2 * expansion_order - 1;

//         // Index maps between surface and convolution grids
//         let mut surf_to_conv: HashMap<usize, usize> = HashMap::new();
//         let mut conv_to_surf: HashMap<usize, usize> = HashMap::new();

//         // Initialise surface grid index
//         let mut surf_index = 0;

//         // The boundaries of the surface grid
//         let lower = expansion_order - 1;
//         let upper = 2 * expansion_order - 2;

//         // Iterate through the entire convolution grid marking the boundaries
//         // This makes the map much easier to understand and debug
//         for i in 0..n {
//             for j in 0..n {
//                 for k in 0..n {
//                     let conv_idx = i * n * n + j * n + k;
//                     if (i >= lower && j >= lower && (k == lower || k == upper))
//                         || (j >= lower && k >= lower && (i == lower || i == upper))
//                         || (k >= lower && i >= lower && (j == lower || j == upper))
//                     {
//                         surf_to_conv.insert(surf_index, conv_idx);
//                         conv_to_surf.insert(conv_idx, surf_index);
//                         surf_index += 1;
//                     }
//                 }
//             }
//         }

//         (surf_to_conv, conv_to_surf)
//     }

//     pub fn compute_kernel(
//         &self,
//         expansion_order: usize,
//         convolution_grid: &[[f64; 3]],
//         min_target: [f64; 3],
//     ) -> Vec<Vec<Vec<f64>>> {
//         let n = 2 * expansion_order - 1;
//         let mut result = vec![vec![vec![0f64; n]; n]; n];

//         for (i, result_i) in result.iter_mut().enumerate() {
//             for (j, result_ij) in result_i.iter_mut().enumerate() {
//                 for (k, result_ijk) in result_ij.iter_mut().enumerate() {
//                     let conv_idx = i * n * n + j * n + k;
//                     let src = convolution_grid[conv_idx];
//                     *result_ijk = self.kernel.kernel(&src[..], &min_target[..]);
//                 }
//             }
//         }
//         result
//     }

//     pub fn compute_signal(&self, expansion_order: usize, charges: &[f64]) -> Vec<Vec<Vec<f64>>> {
//         let n = 2 * expansion_order - 1;
//         let mut result = vec![vec![vec![0f64; n]; n]; n];

//         for (i, result_i) in result.iter_mut().enumerate() {
//             for (j, result_ij) in result_i.iter_mut().enumerate() {
//                 for (k, result_ijk) in result_ij.iter_mut().enumerate() {
//                     let conv_idx = i * n * n + j * n + k;
//                     if self.conv_to_surf_map.contains_key(&conv_idx) {
//                         let surf_idx = self.conv_to_surf_map.get(&conv_idx).unwrap();
//                         *result_ijk = charges[*surf_idx]
//                     }
//                 }
//             }
//         }

//         result
//     }
// }

use std::any::type_name;

fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

mod test {

    use super::*;
    use bempp_kernel::laplace_3d::Laplace3dKernel;

    #[test]
    fn test_svd() {
        let kernel = Laplace3dKernel::<f64>::default();
        let k = 100;
        let order = 2;
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let alpha_inner = 1.05;

        let m2l_data_svd =
            SvdFieldTranslationKiFmm::new(kernel, Some(k), order, domain, alpha_inner);
    }
}
