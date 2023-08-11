use std::{collections::HashMap, hash::Hash};

use num::Complex;
use fftw::types::*;
use rlst::{
    algorithms::{
        linalg::LinAlg,
        traits::svd::{Mode, Svd},
    },
    common::traits::{Eval, NewLikeSelf, Transpose},
    dense::{rlst_mat, traits::*, Dot, Shape},
};

use bempp_traits::{field::FieldTranslationData, kernel::Kernel, types::EvalType, arrays::Array3DAccess, fmm::Fmm};
use bempp_tree::{types::domain::Domain, implementations::helpers::{map_corners_to_surface, find_corners}};
use bempp_tools::Array3D;

use crate::{
    helpers::{compute_transfer_vectors_unique, pad3, flip3, rfft3, rfft3_fftw, axial_reflection_surface, reflect_transfer_vector, reflect_transfer_vector_axial, reflect_transfer_vector_diagonal, diagonal_reflection, axial_reflection_convolution},
    types::{SvdFieldTranslationKiFmm, FftFieldTranslationNaiveKiFmm, FftFieldTranslationKiFmm, SvdM2lEntry, FftM2lEntry, TransferVector},
};

// impl<T> FieldTranslationData<T> for SvdFieldTranslationKiFmm<T>
// where
//     T: Kernel<T = f64> + Default,
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

//     fn compute_m2l_operators<'a>(
//         &self,
//         expansion_order: usize,
//         domain: Self::Domain,
//     ) -> Self::M2LOperators {
//         // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)

//         // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
//         let nrows = self.ncoeffs(expansion_order);
//         let ncols = self.ncoeffs(expansion_order);

//         let ntransfer_vectors = self.transfer_vectors.len();
//         let mut se2tc_fat = rlst_mat![f64, (nrows, ncols * ntransfer_vectors)];

//         let mut se2tc_thin = rlst_mat![f64, (nrows * ntransfer_vectors, ncols)];

//         for (i, t) in self.transfer_vectors.iter().enumerate() {
//             let source_equivalent_surface =
//                 t.source
//                     .compute_surface(&domain, expansion_order, self.alpha);
//             let nsources = source_equivalent_surface.len() / self.kernel.space_dimension();

//             let target_check_surface =
//                 t.target
//                     .compute_surface(&domain, expansion_order, self.alpha);
//             let ntargets = target_check_surface.len() / self.kernel.space_dimension();

//             let mut tmp_gram = rlst_mat![f64, (ntargets, nsources)];

//             self.kernel.assemble_st(
//                 EvalType::Value,
//                 &source_equivalent_surface[..],
//                 &target_check_surface[..],
//                 tmp_gram.data_mut(),
//             );

//             // Need to transpose so that rows correspond to targets, and columns to sources
//             let mut tmp_gram = tmp_gram.transpose().eval();

//             let block_size = nrows * ncols;
//             let start_idx = i * block_size;
//             let end_idx = start_idx + block_size;
//             let block = se2tc_fat.get_slice_mut(start_idx, end_idx);
//             block.copy_from_slice(tmp_gram.data_mut());

//             for j in 0..ncols {
//                 let start_idx = j * ntransfer_vectors * nrows + i * nrows;
//                 let end_idx = start_idx + nrows;
//                 let block_column = se2tc_thin.get_slice_mut(start_idx, end_idx);
//                 let gram_column = tmp_gram.get_slice_mut(j * ncols, j * ncols + ncols);
//                 block_column.copy_from_slice(gram_column);
//             }
//         }

//         let (sigma, u, vt) = se2tc_fat.linalg().svd(Mode::All, Mode::Slim).unwrap();

//         let u = u.unwrap();
//         let vt = vt.unwrap();

//         // Keep 'k' singular values
//         let mut sigma_mat = rlst_mat![f64, (self.k, self.k)];
//         for i in 0..self.k {
//             sigma_mat[[i, i]] = sigma[i]
//         }

//         let (mu, _) = u.shape();
//         let u = u.block((0, 0), (mu, self.k)).eval();

//         let (_, nvt) = vt.shape();
//         let vt = vt.block((0, 0), (self.k, nvt)).eval();

//         // Store compressed M2L operators
//         let (_gamma, _r, st) = se2tc_thin.linalg().svd(Mode::Slim, Mode::All).unwrap();
//         let st = st.unwrap();
//         let (_, nst) = st.shape();
//         let st_block = st.block((0, 0), (self.k, nst));
//         let s_block = st_block.transpose().eval();

//         let mut c = rlst_mat![f64, (self.k, self.k * ntransfer_vectors)];

//         for i in 0..self.transfer_vectors.len() {
//             let top_left = (0, i * ncols);
//             let dim = (self.k, ncols);
//             let vt_block = vt.block(top_left, dim);

//             let tmp = sigma_mat.dot(&vt_block.dot(&s_block));

//             let top_left = (0, i * self.k);
//             let dim = (self.k, self.k);

//             c.block_mut(top_left, dim)
//                 .data_mut()
//                 .copy_from_slice(tmp.data());
//         }

//         let st_block = s_block.transpose().eval();
//         (u, st_block, c)
//     }
// }

// impl<T> SvdFieldTranslationKiFmm<T>
// where
//     T: Kernel<T = f64> + Default,
// {
//     pub fn new(
//         kernel: T,
//         k: Option<usize>,
//         expansion_order: usize,
//         domain: Domain,
//         alpha: f64,
//     ) -> Self {
//         let dummy = rlst_mat![f64, (1, 1)];

//         // TODO: There should be a default for matrices to make code cleaner.
//         let mut result = SvdFieldTranslationKiFmm {
//             alpha,
//             k: 0,
//             kernel,
//             m2l: (
//                 dummy.new_like_self().eval(),
//                 dummy.new_like_self().eval(),
//                 dummy.new_like_self().eval(),
//             ),
//             transfer_vectors: Vec::new(),
//         };

//         let ncoeffs = result.ncoeffs(expansion_order);
//         if let Some(k) = k {
//             // Compression rank <= number of coefficients
//             if k <= ncoeffs {
//                 result.k = k;
//             } else {
//                 result.k = ncoeffs
//             }
//         } else {
//             result.k = 50;
//         }

//         result.transfer_vectors = result.compute_transfer_vectors();
//         result.m2l = result.compute_m2l_operators(expansion_order, domain);

//         result
//     }
// }

// impl<T> FieldTranslationData<T> for FftFieldTranslationNaiveKiFmm<T>
// where
//     T: Kernel<T = f64> + Default,
// {
//     type Domain = Domain;
//     type M2LOperators = Vec<FftM2lEntry>;
//     type TransferVector = Vec<TransferVector>;

//     fn compute_m2l_operators(
//         &self,
//         expansion_order: usize,
//         domain: Self::Domain,
//     ) -> Self::M2LOperators {
//         let mut result = Vec::new();

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

//             let kernel = self.compute_kernel(expansion_order, &conv_grid_sources, min_target);

//             let &(m, n, o) = kernel.shape();

//             // Precompute and store the FFT of each unique kernel interaction

//             // Begin by calculating pad lengths along each dimension
//             // let p = 2_f64.powf((m as f64).log2().ceil()) as usize;
//             // let q = 2_f64.powf((n as f64).log2().ceil()) as usize;
//             // let r = 2_f64.powf((o as f64).log2().ceil()) as usize;

//             // let p = p.max(4);
//             // let q = q.max(4);
//             // let r = r.max(4);
//             let p = m + 1;
//             let q = n + 1;
//             let r = o + 1;

//             let padded_kernel = pad3(&kernel, (p-m, q-n, r-o), (0, 0, 0));

//             // Flip the kernel
//             let padded_kernel = flip3(&padded_kernel);

//             // Compute FFT of kernel for this transfer vector
//             let padded_kernel_hat = rfft3(&padded_kernel);

//             // Store FFT of kernel for this transfer vector            
//             result.push(padded_kernel_hat)
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


// impl<T> FftFieldTranslationNaiveKiFmm<T>
// where
//     T: Kernel<T = f64> + Default,
// {
//     pub fn new(kernel: T, expansion_order: usize, domain: Domain, alpha: f64) -> Self {

//         let mut result = FftFieldTranslationNaiveKiFmm {
//             alpha,
//             kernel,
//             surf_to_conv_map: HashMap::default(),
//             conv_to_surf_map: HashMap::default(),
//             m2l: Vec::default(), 
//             transfer_vectors: Vec::default(),
//         };

//         // Create maps between surface and convolution grids
//         let (surf_to_conv, conv_to_surf) =
//             FftFieldTranslationNaiveKiFmm::<T>::compute_surf_to_conv_map(expansion_order);
        
//         result.surf_to_conv_map = surf_to_conv;
//         result.conv_to_surf_map = conv_to_surf;
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
//         convolution_grid: &[f64],
//         min_target: [f64; 3],
//     )  -> Array3D<f64>
//      {
//         let n = 2 * expansion_order - 1;
//         let mut result = Array3D::<f64>::new((n, n, n));
//         let nconv = n.pow(3);

//         let mut kernel_evals = vec![0f64; nconv];

//         self.kernel.assemble_st(
//             EvalType::Value, 
//             convolution_grid,
//             &min_target[..], 
//             &mut kernel_evals[..]
//         );

//         result.get_data_mut().copy_from_slice(&kernel_evals[..]);

//         result
//     }

//     pub fn compute_signal(
//         &self, 
//         expansion_order: usize, 
//         charges: &[f64]
//     ) 
//     -> Array3D<f64>
//     {
//         let n = 2 * expansion_order - 1;
//         let mut result = Array3D::new((n,n,n));

//         let mut tmp = vec![0f64; n*n*n];

//         for i in 0..n {
//             for j in 0..n {
//                 for k in 0..n {
//                     let conv_idx = i * n * n + j * n + k;
//                     if let Some(surf_idx) = self.conv_to_surf_map.get(&conv_idx) {
//                         tmp[conv_idx] = charges[*surf_idx];
//                     } else {
//                         tmp[conv_idx] = 0f64;
//                     }
//                 }
//             }
//         }
        
//         result.get_data_mut().copy_from_slice(&tmp[..]);

//         result
//     }
// }

impl<T> FieldTranslationData<T> for FftFieldTranslationKiFmm<T>
where
    T: Kernel<T = f64> + Default,
{
    type Domain = Domain;
    type M2LOperators = (
        Vec<FftM2lEntry>, // M2L matrices
        Vec<HashMap<usize, usize>>, // Surface to Reflected Map
        Vec<HashMap<usize, usize>>, // Inverse Surface to Reflected Map
        Vec<HashMap<usize, usize>>, // Conv Grid to Reflected Map
        Vec<HashMap<usize, usize>>, // Inverse Conv Grid to Reflected Map
    );
    type TransferVector = Vec<TransferVector>;
    type TransferVectorMap = HashMap<usize, usize>;

    fn compute_m2l_operators(
        &self,
        expansion_order: usize,
        domain: Self::Domain,
    ) -> Self::M2LOperators {
        let mut result = Vec::new();

        let mut surface_maps = Vec::new();
        let mut inv_surface_maps = Vec::new();
        let mut conv_maps = Vec::new();
        let mut inv_conv_maps = Vec::new();

        // Create a map between corner indices and surface indices
        let (map_corner_to_surface, inv_map_corner_to_surface) = map_corners_to_surface(expansion_order);

        for t in self.transfer_vectors.iter() {
            let source_equivalent_surface =
                t.source
                    .compute_surface(&domain, expansion_order, self.alpha);
            let nsources = source_equivalent_surface.len() / 3;

            // Find multi-index after axial reflections
            let (_, source_multi_index) = t.source.surface_grid(expansion_order);

            let mut source_multi_index_axial = vec![0usize; source_multi_index.len()];

            for i in 0..nsources {
                let m = [
                    source_multi_index[i],
                    source_multi_index[nsources+i],
                    source_multi_index[2*nsources+i],
                ];

                let m_refl = axial_reflection_surface(&m[..], &t.components[..], expansion_order);

                source_multi_index_axial[i] = m_refl[0];
                source_multi_index_axial[nsources+i] = m_refl[1];
                source_multi_index_axial[2*nsources+i] = m_refl[2];
            }

            // Find the transfer vector after reflection into reference octant
            let axial_transfer_vector = reflect_transfer_vector_axial(&t.components[..]);

            // Find the transfer vector after reflection into reference cone
            let diag_axial_transfer_vector = reflect_transfer_vector_diagonal(&axial_transfer_vector[..]);

            // Find multi-index after diagonal reflections
            let mut source_multi_index_axial_diag = vec![0usize; source_multi_index.len()];

            for i in 0..nsources {
                let m = [
                    source_multi_index_axial[i],
                    source_multi_index_axial[nsources+i],
                    source_multi_index_axial[2*nsources+i],
                ];

                let m_refl = diagonal_reflection(&m[..], &t.components[..]);

                source_multi_index_axial_diag[i] = m_refl[0];
                source_multi_index_axial_diag[nsources+i] = m_refl[1];
                source_multi_index_axial_diag[2*nsources+i] = m_refl[2];
            } 
            
            // Need to create a map between original surface multi-indices and reflected surface multi-indices
            let mut map_surface = HashMap::new();
            let mut inv_map_surface = HashMap::new();

            for i in 0..nsources {
                let original = [
                    source_multi_index[i],
                    source_multi_index[nsources+i],
                    source_multi_index[2*nsources+i]
                ];
                for j in 0..nsources {
                    let reflected = [
                        source_multi_index_axial_diag[i],
                        source_multi_index_axial_diag[nsources+i],
                        source_multi_index_axial_diag[2*nsources+i]
                    ];
                    
                    if (original[0] == reflected[0]) && (original[1] == reflected[1]) && (original[2] == reflected[2]) {
                        map_surface.insert(i, j);
                        inv_map_surface.insert(j, i);
                    }
                }
            }


            // Find convolution grid
            let conv_point_corner_index = 7;
            let corners = find_corners(&source_equivalent_surface[..]);
            let conv_point = [
                corners[conv_point_corner_index],
                corners[8+conv_point_corner_index],
                corners[16+conv_point_corner_index]
            ];
            let conv_point_index = inv_map_corner_to_surface.get(&conv_point_corner_index).unwrap();

            let (conv_grid_sources, conv_grid_multi_index) = t.source.convolution_grid(
                expansion_order,
                &domain,
                self.alpha,
                &conv_point[..],
                conv_point_corner_index
            );

            // TODO: 1 Based indexing for multi-indices needs to be fixed in Python first
            // Find multi-indices after axial reflections
            let nconv = conv_grid_multi_index.len() / 3;
            let mut conv_grid_multi_index_axial = vec![0usize; conv_grid_multi_index.len()];

            for i in 0..nconv {
                let m = [
                    conv_grid_multi_index[i],
                    conv_grid_multi_index[nconv+i],
                    conv_grid_multi_index[2*nconv+i],
                ];

                let m_refl = axial_reflection_convolution(&m[..], &t.components[..], expansion_order);

                conv_grid_multi_index_axial[i] = m_refl[0];
                conv_grid_multi_index_axial[nconv+i] = m_refl[1];
                conv_grid_multi_index_axial[2*nconv+i] = m_refl[2];
            }

            // Find multi-indices after diagonal reflections
            let mut conv_grid_multi_index_axial_diag = vec![0usize; conv_grid_multi_index.len()];

            for i in 0..nconv {
                let m = [
                    conv_grid_multi_index_axial[i],
                    conv_grid_multi_index_axial[nconv+i],
                    conv_grid_multi_index_axial[2*nconv+i],
                ];

                let m_refl = diagonal_reflection(&m[..], &t.components[..]);

                conv_grid_multi_index_axial_diag[i] = m_refl[0];
                conv_grid_multi_index_axial_diag[nconv+i] = m_refl[1];
                conv_grid_multi_index_axial_diag[2*nconv+i] = m_refl[2];
            }

              // Need to create a map between original conv grid multi-indices and reflected conv grid multi-indices
              let mut map_conv = HashMap::new();
              let mut inv_map_conv = HashMap::new();
  
              for i in 0..nconv {
                  let original = [
                      conv_grid_multi_index[i],
                      conv_grid_multi_index[nconv+i],
                      conv_grid_multi_index[2*nconv+i]
                  ];
                  for j in 0..nsources {
                      let reflected = [
                          conv_grid_multi_index_axial_diag[i],
                          conv_grid_multi_index_axial_diag[nconv+i],
                          conv_grid_multi_index_axial_diag[2*nconv+i]
                      ];

                      if (original[0] == reflected[0]) && (original[1] == reflected[1]) && (original[2] == reflected[2]) {
                          map_conv.insert(i, j);
                          inv_map_conv.insert(j, i);
                      }
                  }
              }
            
            // Find reflected convolution grid
            let &conv_point_index_reflected = map_surface.get(&conv_point_index).unwrap();
            let conv_point_reflected = [
              source_equivalent_surface[conv_point_index_reflected],
              source_equivalent_surface[nsources+conv_point_index_reflected],
              source_equivalent_surface[2*nsources+conv_point_index_reflected],
            ];
            let &conv_point_corner_index_reflected = map_corner_to_surface.get(&conv_point_index_reflected).unwrap();

            let (conv_grid_reflected, _) = t.source.convolution_grid(
                expansion_order, &domain, self.alpha, &conv_point_reflected[..], conv_point_corner_index_reflected
            );

            // Compute target check surface
            let target_check_surface = t.target.compute_surface(&domain, expansion_order, self.alpha);
            let ntargets = target_check_surface.len() / 3;
            
            // Find target with which to calculate kernel wrt
            let kernel_point_index = 0;
            let kernel_point = [
                target_check_surface[kernel_point_index],
                target_check_surface[ntargets+kernel_point_index],
                target_check_surface[2*ntargets+kernel_point_index]
            ];
            
            let &kernel_point_index_reflected = map_surface.get(&kernel_point_index).unwrap();
            let kernel_point_reflected = [
                target_check_surface[kernel_point_index_reflected],
                target_check_surface[ntargets+kernel_point_index_reflected],
                target_check_surface[2*ntargets+kernel_point_index_reflected]
            ];

            let reflected_kernel = self.compute_kernel_reflected(
                expansion_order, &conv_grid_reflected[..], &map_conv, &kernel_point[..]
            );

            let &(m, n, o) = reflected_kernel.shape();

            // Precompute and store the FFT of each unique kernel interaction
            let p = m + 1;
            let q = n + 1;
            let r = o + 1;
            let padded_reflected_kernel = pad3(&reflected_kernel, (p-m, q-n, r-o), (0, 0, 0));

            // Flip the kernel
            let mut padded_reflected_kernel = flip3(&padded_reflected_kernel);

            // Compute FFT of kernel for this transfer vector
            let mut padded_reflected_kernel_hat = Array3D::<c64>::new((p, q, r/2 + 1));
            rfft3_fftw(padded_reflected_kernel.get_data_mut(), padded_reflected_kernel_hat.get_data_mut(), &[p, q, r]);

            // Store FFT of kernel for this transfer vector            
            result.push(padded_reflected_kernel_hat);

            // Save surface and convolution maps
            surface_maps.push(map_surface);
            inv_surface_maps.push(inv_map_surface);
            conv_maps.push(map_conv);
            inv_conv_maps.push(inv_map_conv);
        }

        (result, surface_maps, inv_surface_maps, conv_maps, inv_conv_maps)
    }

    fn compute_transfer_vectors(&self) -> (Self::TransferVector, Self::TransferVectorMap) {
        compute_transfer_vectors_unique()
    }

    fn ncoeffs(&self, expansion_order: usize) -> usize {
        6 * (expansion_order - 1).pow(2) + 2
    }
}


impl<T> FftFieldTranslationKiFmm<T>
where
    T: Kernel<T = f64> + Default,
{
    pub fn new(kernel: T, expansion_order: usize, domain: Domain, alpha: f64) -> Self {

        let mut result = FftFieldTranslationKiFmm {
            alpha,
            kernel,
            surf_to_conv_map: HashMap::default(),
            conv_to_surf_map: HashMap::default(),
            m2l: Vec::default(), 
            transfer_vectors: Vec::default(),
            transfer_vector_map: HashMap::default(),
            surf_grid_maps: Vec::default(),
            inv_surf_grid_maps: Vec::default(),
            conv_grid_maps: Vec::new(),
            inv_conv_grid_maps: Vec::new(),
        };

        // Create maps between surface and convolution grids
        let (surf_to_conv, conv_to_surf) =
        FftFieldTranslationKiFmm::<T>::compute_surf_to_conv_map(expansion_order);
        
        result.surf_to_conv_map = surf_to_conv;
        result.conv_to_surf_map = conv_to_surf;
        (result.transfer_vectors, result.transfer_vector_map) = result.compute_transfer_vectors();
        let (m2l, surface_maps, inv_surface_maps, conv_maps, inv_conv_maps) = result.compute_m2l_operators(expansion_order, domain);
        result.m2l = m2l;
        result.surf_grid_maps = surface_maps;
        result.inv_surf_grid_maps = inv_surface_maps;
        result.conv_grid_maps= conv_maps;
        result.inv_conv_grid_maps = inv_conv_maps;

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

        result.get_data_mut().copy_from_slice(&kernel_evals[..]);

        result
    }

    pub fn compute_kernel_reflected(
        &self,
        expansion_order: usize,
        convolution_grid_reflected: &[f64],
        convolution_map: &HashMap<usize, usize>,
        kernel_point: &[f64]
    ) -> Array3D<f64> {
        let n = 2 * expansion_order - 1;
        let nconv = n.pow(3);
        let mut result = Array3D::<f64>::new((n, n, n));

        for i in 0..n {
            for j in 0..n {
                for k in 0.. n {
                    let original_conv_idx = i*n*n+j*n+k;
                    let &reflected_conv_index = convolution_map.get(&original_conv_idx).unwrap();
                    let conv_point = [
                        convolution_grid_reflected[reflected_conv_index],
                        convolution_grid_reflected[nconv+reflected_conv_index],
                        convolution_grid_reflected[2*nconv+reflected_conv_index],
                    ];
                    let mut res = vec![0f64];
                    self.kernel.evaluate_st(EvalType::Value, &conv_point[..], &kernel_point[..], &[1.0], &mut res[..]);

                    *result.get_mut(i, j, k).unwrap() = res[0];
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

        let mut tmp = vec![0f64; n*n*n];

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let conv_idx = i * n * n + j * n + k;
                    if let Some(surf_idx) = self.conv_to_surf_map.get(&conv_idx) {
                        tmp[conv_idx] = charges[*surf_idx];
                    } else {
                        tmp[conv_idx] = 0f64;
                    }
                }
            }
        }
        
        result.get_data_mut().copy_from_slice(&tmp[..]);

        result
    }
}


#[cfg(test)]
mod test {

    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::types::morton::MortonKey;
    use itertools::Itertools;

    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_fft() {

        let order = 2;
        let alpha = 1.05;
        let level = 2;
        let kernel = Laplace3dKernel::<f64>::new();

        let domain = Domain { origin: [0., 0., 0.], diameter: [1.0, 1.0, 1.0] };
        let fft = FftFieldTranslationKiFmm::new(kernel, order, domain, alpha);

        
        println!("FFT {:?}", fft.transfer_vectors.len());
        println!("FFT {:?}", fft.transfer_vector_map.keys().len());
        let tmp: HashSet<usize> = fft.transfer_vector_map.values().cloned().collect();
        println!("FFT {:?}", tmp.len());
        
        assert!(false);
    }
}