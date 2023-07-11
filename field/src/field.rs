use std::{collections::HashMap, hash::Hash};

use num::Complex;
use rlst::{
    algorithms::{
        linalg::LinAlg,
        traits::svd::{Mode, Svd},
    },
    common::traits::{Eval, NewLikeSelf, Transpose},
    dense::{rlst_mat, traits::*, Dot, Shape},
};

use bempp_traits::{field::FieldTranslationData, kernel::Kernel, types::EvalType, arrays::Array3DAccess, fmm::Fmm};
use bempp_tree::types::domain::Domain;
use bempp_tools::Array3D;

use crate::{
    helpers::{compute_transfer_vectors, pad3, flip3, rfft3},
    types::{SvdFieldTranslationKiFmm, FftFieldTranslationNaiveKiFmm, SvdM2lEntry, FftM2lEntry, TransferVector},
};

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
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = self.ncoeffs(expansion_order);
        let ncols = self.ncoeffs(expansion_order);

        let ntransfer_vectors = self.transfer_vectors.len();
        let mut se2tc_fat = rlst_mat![f64, (nrows, ncols * ntransfer_vectors)];

        let mut se2tc_thin = rlst_mat![f64, (nrows * ntransfer_vectors, ncols)];

        for (i, t) in self.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface =
                t.source
                    .compute_surface(&domain, expansion_order, self.alpha);
            let nsources = source_equivalent_surface.len() / self.kernel.space_dimension();

            let target_check_surface =
                t.target
                    .compute_surface(&domain, expansion_order, self.alpha);
            let ntargets = target_check_surface.len() / self.kernel.space_dimension();

            let mut tmp_gram = rlst_mat![f64, (ntargets, nsources)];

            self.kernel.assemble_st(
                EvalType::Value,
                &source_equivalent_surface[..],
                &target_check_surface[..],
                tmp_gram.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut tmp_gram = tmp_gram.transpose().eval();

            let block_size = nrows * ncols;
            let start_idx = i * block_size;
            let end_idx = start_idx + block_size;
            let block = se2tc_fat.get_slice_mut(start_idx, end_idx);
            block.copy_from_slice(tmp_gram.data_mut());

            for j in 0..ncols {
                let start_idx = j * ntransfer_vectors * nrows + i * nrows;
                let end_idx = start_idx + nrows;
                let block_column = se2tc_thin.get_slice_mut(start_idx, end_idx);
                let gram_column = tmp_gram.get_slice_mut(j * ncols, j * ncols + ncols);
                block_column.copy_from_slice(gram_column);
            }
        }

        let (sigma, u, vt) = se2tc_fat.linalg().svd(Mode::All, Mode::Slim).unwrap();

        let u = u.unwrap();
        let vt = vt.unwrap();

        // Keep 'k' singular values
        let mut sigma_mat = rlst_mat![f64, (self.k, self.k)];
        for i in 0..self.k {
            sigma_mat[[i, i]] = sigma[i]
        }

        let (mu, _) = u.shape();
        let u = u.block((0, 0), (mu, self.k)).eval();

        let (_, nvt) = vt.shape();
        let vt = vt.block((0, 0), (self.k, nvt)).eval();

        // Store compressed M2L operators
        let (_gamma, _r, st) = se2tc_thin.linalg().svd(Mode::Slim, Mode::All).unwrap();
        let st = st.unwrap();
        let (_, nst) = st.shape();
        let st_block = st.block((0, 0), (self.k, nst));
        let s_block = st_block.transpose().eval();

        let mut c = rlst_mat![f64, (self.k, self.k * ntransfer_vectors)];

        for i in 0..self.transfer_vectors.len() {
            let top_left = (0, i * ncols);
            let dim = (self.k, ncols);
            let vt_block = vt.block(top_left, dim);

            let tmp = sigma_mat.dot(&vt_block.dot(&s_block));

            let top_left = (0, i * self.k);
            let dim = (self.k, self.k);

            c.block_mut(top_left, dim)
                .data_mut()
                .copy_from_slice(tmp.data());
        }

        let st_block = s_block.transpose().eval();
        (u, st_block, c)
    }
}

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
            k: 0,
            kernel,
            m2l: (
                dummy.new_like_self().eval(),
                dummy.new_like_self().eval(),
                dummy.new_like_self().eval(),
            ),
            transfer_vectors: Vec::new(),
        };

        let ncoeffs = result.ncoeffs(expansion_order);
        if let Some(k) = k {
            // Compression rank <= number of coefficients
            if k <= ncoeffs {
                result.k = k;
            } else {
                result.k = ncoeffs
            }
        } else {
            result.k = 50;
        }

        result.transfer_vectors = result.compute_transfer_vectors();
        result.m2l = result.compute_m2l_operators(expansion_order, domain);

        result
    }
}

impl<T> FieldTranslationData<T> for FftFieldTranslationNaiveKiFmm<T>
where
    T: Kernel<T = f64> + Default,
{
    type Domain = Domain;
    type M2LOperators = Vec<FftM2lEntry>;
    type TransferVector = Vec<TransferVector>;

    fn compute_m2l_operators(
        &self,
        expansion_order: usize,
        domain: Self::Domain,
    ) -> Self::M2LOperators {
        let mut result = Vec::new();

        for t in self.transfer_vectors.iter() {
            let source_equivalent_surface =
                t.source
                    .compute_surface(&domain, expansion_order, self.alpha);

            let conv_grid_sources = t.source.convolution_grid(
                expansion_order,
                &domain,
                &source_equivalent_surface,
                self.alpha,
            );

            let target_check_surface = t.target.compute_surface(&domain, expansion_order, self.alpha);

            // TODO: Remove dim
            let dim = 3;
            // Find min target
            let ncoeffs: usize = target_check_surface.len() / dim;
            let sums: Vec<_> = (0..ncoeffs)
                .map(|i| target_check_surface[i] + target_check_surface[ncoeffs + i] + target_check_surface[2*ncoeffs + i])
                .collect();

            let min_index = sums
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(index, _)| index)
                .unwrap();

            let min_target = [
                target_check_surface[min_index],
                target_check_surface[min_index + ncoeffs],
                target_check_surface[min_index + 2 * ncoeffs],
            ];

            let kernel = self.compute_kernel(expansion_order, &conv_grid_sources, min_target);

            let &(m, n, o) = kernel.shape();

            // Precompute and store the FFT of each unique kernel interaction

            // Begin by calculating pad lengths along each dimension
            let p = 2 * m;
            let q = 2 * n;
            let r = 2 * o;

            let padded_kernel = pad3(&kernel, (p-m, q-n, r-o), (0, 0, 0));

            // Flip the kernel
            let padded_kernel = flip3(&padded_kernel);

            // Compute FFT of kernel for this transfer vector
            let padded_kernel_hat = rfft3(&padded_kernel);

            // Store FFT of kernel for this transfer vector            
            result.push(padded_kernel_hat)
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

impl<T> FftFieldTranslationNaiveKiFmm<T>
where
    T: Kernel<T = f64> + Default,
{
    pub fn new(kernel: T, expansion_order: usize, domain: Domain, alpha: f64) -> Self {

        let mut result = FftFieldTranslationNaiveKiFmm {
            alpha,
            kernel,
            surf_to_conv_map: HashMap::default(),
            conv_to_surf_map: HashMap::default(),
            m2l: Vec::default(), 
            transfer_vectors: Vec::default(),
        };

        // Create maps between surface and convolution grids
        let (surf_to_conv, conv_to_surf) =
            FftFieldTranslationNaiveKiFmm::<T>::compute_surf_to_conv_map(expansion_order);
        
        result.surf_to_conv_map = surf_to_conv;
        result.conv_to_surf_map = conv_to_surf;
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
                    if (i >= lower && j >= lower && (k == lower || k == upper))
                        || (j >= lower && k >= lower && (i == lower || i == upper))
                        || (k >= lower && i >= lower && (j == lower || j == upper))
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
        convolution_grid: &[f64],
        min_target: [f64; 3],
    )  -> Array3D<f64>
     {
        let n = 2 * expansion_order - 1;
        let mut result = Array3D::<f64>::new((n, n, n));
        let nconv = n.pow(3);

        let mut kernel_evals = vec![0f64; nconv];

        self.kernel.assemble_st(
            EvalType::Value, 
            convolution_grid,
            &min_target[..], 
            &mut kernel_evals[..]
        );

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let conv_idx = i * n * n + j * n + k;
                    *result.get_mut(i, j, k).unwrap() = kernel_evals[conv_idx];
                }
            }
        } 

        result
    }

    pub fn compute_signal(
        &self, 
        expansion_order: usize, 
        charges: &[f64]
    ) 
    -> Array3D<f64>
    {
        let n = 2 * expansion_order - 1;
        let mut result = Array3D::new((n,n,n));

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let conv_idx = i*n*n+j*n+k;
                    if self.conv_to_surf_map.contains_key(&conv_idx) {
                        let surf_idx = self.conv_to_surf_map.get(&conv_idx).unwrap();
                        *result.get_mut(i, j, k).unwrap() = charges[*surf_idx];
                    } 
                }
            }
        }
       
        result
    }
}

#[cfg(test)]
mod test {

    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::types::morton::MortonKey;

    use super::*;

    #[test]
    fn test_fft() {

        let order = 2;
        let alpha = 1.05;
        let level = 2;
        let kernel = Laplace3dKernel::<f64>::new();

        let domain = Domain { origin: [0., 0., 0.], diameter: [1.0, 1.0, 1.0] };
        let fft = FftFieldTranslationNaiveKiFmm::new(kernel, order, domain, alpha);

        let domain = Domain { origin: [0., 0., 0.], diameter: [1., 1., 1.] };
        let key = MortonKey::from_point(&[0.5, 0.5, 0.5], &domain, level);
        let surface_grid = key.compute_surface(&domain, order, alpha);
        let convolution_grid = key.convolution_grid(order, &domain, &surface_grid, alpha);
        let min_target = [0.8, 0.8, 0.8];

        let k = fft.compute_kernel(order, &convolution_grid, min_target);

        let &(m, n, o) = k.shape();

        for i in 0..m {
            println!("{:?}", k.get(i, 1, 1));
        }
    }
}