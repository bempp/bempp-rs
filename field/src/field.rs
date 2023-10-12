//! Implementation of traits for field translations via the FFT and SVD.
use std::collections::{HashMap, HashSet};

use fftw::types::*;
use itertools::Itertools;
use rlst::{
    algorithms::{
        linalg::LinAlg,
        traits::svd::{Mode, Svd},
    },
    common::traits::{Eval, Transpose},
    dense::{rlst_mat, Dot, RandomAccessMut, RawAccess, RawAccessMut, Shape},
};

use bempp_tools::Array3D;
use bempp_traits::{
    arrays::Array3DAccess, field::FieldTranslationData, kernel::Kernel, types::EvalType,
};
use bempp_tree::{
    implementations::helpers::find_corners, types::domain::Domain, types::morton::MortonKey,
};

use crate::{
    array::{flip3, pad3},
    fft::rfft3_fftw,
    surface::{axial_reflection_surface, diagonal_reflection},
    transfer_vector::{
        axially_reflect_components, compute_transfer_vectors, compute_transfer_vectors_unique,
        diagonally_reflect_components,
    },
    types::{
        FftFieldTranslationKiFmm, FftM2lOperatorData, SvdFieldTranslationKiFmm, SvdM2lOperatorData,
        TransferVector,
    },
};

impl<T> FieldTranslationData<T> for SvdFieldTranslationKiFmm<T>
where
    T: Kernel<T = f64> + Default,
{
    type TransferVector = Vec<TransferVector>;
    type TransferVectorMap = HashMap<usize, usize>;
    type M2LOperators = SvdM2lOperatorData;
    type Domain = Domain;

    fn compute_transfer_vectors(&self) -> (Self::TransferVector, Self::TransferVectorMap) {
        compute_transfer_vectors()
    }

    fn ncoeffs(&self, order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }

    fn compute_m2l_operators<'a>(&self, order: usize, domain: Self::Domain) -> Self::M2LOperators {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = self.ncoeffs(order);
        let ncols = self.ncoeffs(order);

        let ntransfer_vectors = self.transfer_vectors.len();
        let mut se2tc_fat = rlst_mat![f64, (nrows, ncols * ntransfer_vectors)];

        let mut se2tc_thin = rlst_mat![f64, (nrows * ntransfer_vectors, ncols)];

        for (i, t) in self.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface = t.source.compute_surface(&domain, order, self.alpha);
            let nsources = source_equivalent_surface.len() / self.kernel.space_dimension();

            let target_check_surface = t.target.compute_surface(&domain, order, self.alpha);
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

        SvdM2lOperatorData { u, st_block, c }
    }
}

impl<T> SvdFieldTranslationKiFmm<T>
where
    T: Kernel<T = f64> + Default,
{
    /// Constructor for SVD field translation struct for the kernel independent FMM (KiFMM).
    ///
    /// # Arguments
    /// * `kernel` - The kernel being used, only compatible with homogenous, translationally invariant kernels.
    /// * `k` - The maximum rank to be used in SVD compression for the translation operators, if none is specified will be taken as  max({50, max_column_rank})
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `domain` - Domain associated with the global point set.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    pub fn new(kernel: T, k: Option<usize>, order: usize, domain: Domain, alpha: f64) -> Self {
        let mut result = SvdFieldTranslationKiFmm {
            alpha,
            k: 0,
            kernel,
            operator_data: SvdM2lOperatorData::default(),
            transfer_vectors: Vec::new(),
        };

        let ncoeffs = result.ncoeffs(order);
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

        (result.transfer_vectors, _) = result.compute_transfer_vectors();
        result.operator_data = result.compute_m2l_operators(order, domain);

        result
    }
}

impl<T> FieldTranslationData<T> for FftFieldTranslationKiFmm<T>
where
    T: Kernel<T = f64> + Default,
{
    type Domain = Domain;

    type M2LOperators = FftM2lOperatorData;

    type TransferVector = Vec<TransferVector>;
    type TransferVectorMap = HashMap<usize, usize>;

    fn compute_m2l_operators(&self, order: usize, domain: Self::Domain) -> Self::M2LOperators {
        let mut kernel_data = HashMap::new();

        let mut permutation_matrices = HashMap::new();
        let mut permuted_multi_indices = HashMap::new();

        // Calculate all transfer vectors (316)
        let (transfer_vectors, _) = compute_transfer_vectors();

        // Store a set of considered vectors, to avoid redundant computations
        let mut considered = HashSet::new();

        for t in transfer_vectors.iter() {
            // Find transfer vector after it's been reflected in reference octant
            let axial_transfer_vector = axially_reflect_components(&t.components);
            // Find transfer vector after reflection into reference cone
            let diag_axial_transfer_vector =
                diagonally_reflect_components(&axial_transfer_vector[..]);
            // Compute reflected checksum
            let t_refl =
                MortonKey::find_transfer_vector_from_components(&diag_axial_transfer_vector);

            // Find multi-index after axial reflections
            let (_, source_multi_index) = MortonKey::surface_grid(order);

            // Find multi-indices after axial reflection
            let mut source_multi_index_axial = vec![0usize; source_multi_index.len()];

            let nsources = 6 * (order - 1).pow(2) + 2;

            for i in 0..nsources {
                let m = [
                    source_multi_index[i],
                    source_multi_index[nsources + i],
                    source_multi_index[2 * nsources + i],
                ];
                let m_refl = axial_reflection_surface(&m[..], &t.components[..], order);

                source_multi_index_axial[i] = m_refl[0];
                source_multi_index_axial[nsources + i] = m_refl[1];
                source_multi_index_axial[2 * nsources + i] = m_refl[2];
            }

            // Find multi-index after diagonal and axial reflections
            let mut source_multi_index_axial_diag = vec![0usize; source_multi_index.len()];

            for i in 0..nsources {
                let m = [
                    source_multi_index_axial[i],
                    source_multi_index_axial[nsources + i],
                    source_multi_index_axial[2 * nsources + i],
                ];

                let m_refl = diagonal_reflection(&m, &axial_transfer_vector);

                source_multi_index_axial_diag[i] = m_refl[0];
                source_multi_index_axial_diag[nsources + i] = m_refl[1];
                source_multi_index_axial_diag[2 * nsources + i] = m_refl[2];
            }

            // Need a map between between reflected/unreflected surfaces multiindices in terms of linear index
            let mut map_surface = rlst_mat![f64, (nsources, nsources)];

            for i in 0..nsources {
                let original = [
                    source_multi_index[i],
                    source_multi_index[nsources + i],
                    source_multi_index[2 * nsources + i],
                ];
                for j in 0..nsources {
                    let reflected = [
                        source_multi_index_axial_diag[j],
                        source_multi_index_axial_diag[nsources + j],
                        source_multi_index_axial_diag[2 * nsources + j],
                    ];

                    if (original[0] == reflected[0])
                        & (original[1] == reflected[1])
                        & (original[2] == reflected[2])
                    {
                        // map_surface.insert(i, j);
                        *map_surface.get_mut(i, j).unwrap() = 1.0;
                    }
                }
            }

            // let source_multi_index_axial_diag = source_multi_index_axial_diag.iter().map(|e| e -1 ).collect_vec();
            permutation_matrices.insert(t.hash, map_surface);
            permuted_multi_indices.insert(t.hash, source_multi_index_axial_diag);

            if !considered.contains(&t_refl) {
                // Add reflected checksum to checked set.
                considered.insert(t_refl);

                // Continue with algorithm
                let source_equivalent_surface =
                    t.source.compute_surface(&domain, order, self.alpha);
                let nsources = source_equivalent_surface.len() / 3;

                let ntargets = source_equivalent_surface.len() / 3;

                // Find multi-index after axial reflections
                let (_, source_multi_index) = MortonKey::surface_grid(order);

                // Find multi-indices after axial reflection
                let mut source_multi_index_axial = vec![0usize; source_multi_index.len()];

                for i in 0..nsources {
                    let m = [
                        source_multi_index[i],
                        source_multi_index[nsources + i],
                        source_multi_index[2 * nsources + i],
                    ];
                    let m_refl = axial_reflection_surface(&m[..], &t.components[..], order);

                    source_multi_index_axial[i] = m_refl[0];
                    source_multi_index_axial[nsources + i] = m_refl[1];
                    source_multi_index_axial[2 * nsources + i] = m_refl[2];
                }

                // Find multi-index after diagonal and axial reflections
                let mut source_multi_index_axial_diag = vec![0usize; source_multi_index.len()];

                for i in 0..nsources {
                    let m = [
                        source_multi_index_axial[i],
                        source_multi_index_axial[nsources + i],
                        source_multi_index_axial[2 * nsources + i],
                    ];

                    let m_refl = diagonal_reflection(&m, &axial_transfer_vector);

                    source_multi_index_axial_diag[i] = m_refl[0];
                    source_multi_index_axial_diag[nsources + i] = m_refl[1];
                    source_multi_index_axial_diag[2 * nsources + i] = m_refl[2];
                }

                // Find representative source/target pair
                let r_idx = transfer_vectors
                    .iter()
                    .enumerate()
                    .filter_map(|(i, t)| if t.hash == t_refl { Some(i) } else { None })
                    .collect_vec();
                let r_idx = r_idx[0];
                let r_t = &transfer_vectors[r_idx];

                let r_source_equivalent_surface =
                    r_t.source.compute_surface(&domain, order, self.alpha);
                let r_target_check_surface = r_t.target.compute_surface(&domain, order, self.alpha);

                // Find the representative convolution point, i.e. furthest corner.
                let r_conv_point_corner_index = 7;
                let r_corners = find_corners(&r_source_equivalent_surface[..]);
                let r_conv_point_corner = [
                    r_corners[r_conv_point_corner_index],
                    r_corners[8 + r_conv_point_corner_index],
                    r_corners[16 + r_conv_point_corner_index],
                ];

                let (r_conv_grid, _) = t.source.convolution_grid(
                    order,
                    &domain,
                    self.alpha,
                    &r_conv_point_corner,
                    r_conv_point_corner_index,
                );

                // Compute representative kernel
                let r_kernel_point_index = 0;
                let r_kernel_point = [
                    r_target_check_surface[r_kernel_point_index],
                    r_target_check_surface[ntargets + r_kernel_point_index],
                    r_target_check_surface[2 * ntargets + r_kernel_point_index],
                ];

                // Compute the kernel.
                let r_kernel = self.compute_kernel(order, &r_conv_grid, r_kernel_point);
                let &(m, n, o) = r_kernel.shape();
                let p = m + 1;
                let q = n + 1;
                let r = o + 1;
                let r_padded_kernel = pad3(&r_kernel, (p - m, q - n, r - o), (0, 0, 0));

                let mut r_padded_kernel = flip3(&r_padded_kernel);

                // Compute FFT of kernel for this transfer vector
                let mut r_padded_kernel_hat = Array3D::<c64>::new((p, q, r / 2 + 1));
                rfft3_fftw(
                    r_padded_kernel.get_data_mut(),
                    r_padded_kernel_hat.get_data_mut(),
                    &[p, q, r],
                );

                // Store FFT of kernel for this transfer vector
                kernel_data.insert(t_refl, r_padded_kernel_hat);
            }
        }

        assert!(considered.len() == 16);

        FftM2lOperatorData {
            kernel_data,
            permutation_matrices,
            permuted_multi_indices,
        }
    }

    fn compute_transfer_vectors(&self) -> (Self::TransferVector, Self::TransferVectorMap) {
        compute_transfer_vectors_unique()
    }

    fn ncoeffs(&self, order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }
}

impl<T> FftFieldTranslationKiFmm<T>
where
    T: Kernel<T = f64> + Default,
{
    /// Constructor for FFT field translation struct for the kernel independent FMM (KiFMM).
    ///
    /// # Arguments
    /// * `kernel` - The kernel being used, only compatible with homogenous, translationally invariant kernels.
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `domain` - Domain associated with the global point set.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    pub fn new(kernel: T, order: usize, domain: Domain, alpha: f64) -> Self {
        let mut result = FftFieldTranslationKiFmm {
            alpha,
            kernel,
            surf_to_conv_map: HashMap::default(),
            conv_to_surf_map: HashMap::default(),
            operator_data: FftM2lOperatorData::default(),
            transfer_vectors: Vec::default(),
            transfer_vector_map: HashMap::default(),
        };

        // Create maps between surface and convolution grids
        let (surf_to_conv, conv_to_surf) =
            FftFieldTranslationKiFmm::<T>::compute_surf_to_conv_map(order);

        result.surf_to_conv_map = surf_to_conv;
        result.conv_to_surf_map = conv_to_surf;
        (result.transfer_vectors, result.transfer_vector_map) = result.compute_transfer_vectors();

        result.operator_data = result.compute_m2l_operators(order, domain);

        result
    }

    /// Compute map between convolution grid indices and surface indices, return mapping and inverse mapping.
    ///
    /// # Arguments
    /// * `order` - The expansion order for the multipole and local expansions.
    pub fn compute_surf_to_conv_map(
        order: usize,
    ) -> (HashMap<usize, usize>, HashMap<usize, usize>) {
        // Number of points along each axis of convolution grid
        let n = 2 * order - 1;

        // Index maps between surface and convolution grids
        let mut surf_to_conv: HashMap<usize, usize> = HashMap::new();
        let mut conv_to_surf: HashMap<usize, usize> = HashMap::new();

        // Initialise surface grid index
        let mut surf_index = 0;

        // The boundaries of the surface grid when embedded within the convolution grid
        let lower = order - 1;
        let upper = 2 * order - 2;

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let conv_index = i + n * j + n * n * k;
                    if (i >= lower && j >= lower && (k == lower || k == upper))
                        || (j >= lower && k >= lower && (i == lower || i == upper))
                        || (k >= lower && i >= lower && (j == lower || j == upper))
                    {
                        surf_to_conv.insert(surf_index, conv_index);
                        conv_to_surf.insert(conv_index, surf_index);
                        surf_index += 1;
                    }
                }
            }
        }

        (surf_to_conv, conv_to_surf)
    }

    /// Computes the unique kernel evaluations and places them on a convolution grid on the source box wrt to a given target point on the target box surface grid.
    ///
    /// # Arguments
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `convolution_grid` - Cartesian coordinates of points on the convolution grid at a source box, expected in row major order.
    /// * `target_pt` - The point on the target box's surface grid, with which kernels are being evaluated with respect to.
    pub fn compute_kernel(
        &self,
        order: usize,
        convolution_grid: &[f64],
        target_pt: [f64; 3],
    ) -> Array3D<f64> {
        let n = 2 * order - 1;
        let mut result = Array3D::<f64>::new((n, n, n));
        let nconv = n.pow(3);

        let mut kernel_evals = vec![0f64; nconv];

        self.kernel.assemble_st(
            EvalType::Value,
            convolution_grid,
            &target_pt[..],
            &mut kernel_evals[..],
        );

        result.get_data_mut().copy_from_slice(&kernel_evals[..]);

        result
    }

    /// Place charge data on the convolution grid.
    ///
    /// # Arguments
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `charges` - A vector of charges.
    pub fn compute_signal(&self, order: usize, charges: &[f64]) -> Array3D<f64> {
        let n = 2 * order - 1;
        let n_tot = n * n * n;
        let mut result = Array3D::new((n, n, n));

        let mut tmp = vec![0f64; n_tot];

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let conv_index = i + n * j + n * n * k;
                    if let Some(surf_index) = self.conv_to_surf_map.get(&conv_index) {
                        tmp[conv_index] = charges[*surf_index];
                    } else {
                        tmp[conv_index] = 0f64;
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
    use crate::fft::irfft3_fftw;

    use super::*;

    use bempp_kernel::laplace_3d::Laplace3dKernel;

    #[test]
    pub fn test_svd_operator_data() {
        let kernel = Laplace3dKernel::new();
        let order = 5;
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let alpha = 1.05;
        let k = 60;
        let ntransfer_vectors = 316;
        let svd = SvdFieldTranslationKiFmm::new(kernel.clone(), Some(k), order, domain, alpha);
        let m2l = svd.compute_m2l_operators(order, domain);

        // Test that the rank cutoff has been taken correctly (k < ncoeffs)
        assert_eq!(m2l.st_block.shape(), (k, svd.ncoeffs(order)));
        assert_eq!(m2l.c.shape(), (k, k * ntransfer_vectors));
        assert_eq!(m2l.u.shape(), (svd.ncoeffs(order), k));

        // Test that the rank cutoff has been taken correctly (k > ncoeffs)
        let k = 100;
        let svd = SvdFieldTranslationKiFmm::new(kernel.clone(), Some(k), order, domain, alpha);
        let m2l = svd.compute_m2l_operators(order, domain);
        assert_eq!(
            m2l.st_block.shape(),
            (svd.ncoeffs(order), svd.ncoeffs(order))
        );
        assert_eq!(
            m2l.c.shape(),
            (svd.ncoeffs(order), svd.ncoeffs(order) * ntransfer_vectors)
        );
        assert_eq!(m2l.u.shape(), (svd.ncoeffs(order), svd.ncoeffs(order)));

        // Test that the rank cutoff has been taken correctly (k unspecified)
        let k = None;
        let default_k = 50;
        let svd = SvdFieldTranslationKiFmm::new(kernel, k, order, domain, alpha);
        let m2l = svd.compute_m2l_operators(order, domain);
        assert_eq!(m2l.st_block.shape(), (default_k, svd.ncoeffs(order)));
        assert_eq!(m2l.c.shape(), (default_k, default_k * ntransfer_vectors));
        assert_eq!(m2l.u.shape(), (svd.ncoeffs(order), default_k));
    }

    #[test]
    pub fn test_fft_operator_data() {
        let kernel = Laplace3dKernel::new();
        let order = 5;
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let alpha = 1.05;

        let fft = FftFieldTranslationKiFmm::new(kernel, order, domain, alpha);

        // Create a random point in the middle of the domain
        let m2l = fft.compute_m2l_operators(order, domain);

        // Test that the number of precomputed kernel interactions matches the number of transfer vectors
        assert_eq!(m2l.kernel_data.keys().len(), 16);
    }

    #[test]
    fn test_fft_field_translation() {
        let kernel = Laplace3dKernel::new();
        let order: usize = 3;

        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let alpha = 1.05;

        // Some random expansion data
        let ncoeffs = 6 * (order - 1).pow(2) + 2;
        let mut multipole = rlst_mat![f64, (ncoeffs, 1)];

        for i in 0..ncoeffs {
            *multipole.get_mut(i, 0).unwrap() = i as f64;
        }

        // Create field translation object
        let fft = FftFieldTranslationKiFmm::new(kernel, order, domain, alpha);

        // Compute all M2L operators
        let m2l = fft.compute_m2l_operators(order, domain);

        // Pick a random source/target pair
        // let idx = 29;
        let idx = 153;
        let (all_transfer_vectors, _) = compute_transfer_vectors();

        let transfer_vector = &all_transfer_vectors[idx];
        let unique_transfer_vector = fft.transfer_vector_map.get(&transfer_vector.hash).unwrap();

        // Place charges on the convolution grid
        let surface_map = fft
            .operator_data
            .permutation_matrices
            .get(&transfer_vector.hash)
            .unwrap();

        let r_multipole = surface_map.dot(&multipole).eval();

        // println!("HERE {:?} {:?}", multipole.data(), r_multipole.data());

        // Compute FFT of the representative signal
        let r_signal = fft.compute_signal(order, r_multipole.data());
        let &(m, n, o) = r_signal.shape();
        let p = m + 1;
        let q = n + 1;
        let r = o + 1;
        let pad_size = (p - m, q - n, r - o);
        let pad_index = (p - m, q - n, r - o);
        let mut r_padded_signal = pad3(&r_signal, pad_size, pad_index);
        let mut r_padded_signal_hat = Array3D::<c64>::new((p, q, r / 2 + 1));

        rfft3_fftw(
            r_padded_signal.get_data_mut(),
            r_padded_signal_hat.get_data_mut(),
            &[p, q, r],
        );

        // Lookup appropriate FFT of Kernel matrix from precomputatoins
        let r_padded_kernel_hat = m2l.kernel_data.get(unique_transfer_vector).unwrap();

        // Compute convolution
        let hadamard_product = r_padded_signal_hat
            .get_data()
            .iter()
            .zip(r_padded_kernel_hat.get_data().iter())
            .map(|(a, b)| a * b)
            .collect_vec();
        let mut hadamard_product = Array3D::from_data(hadamard_product, (p, q, r / 2 + 1));

        let mut r_potentials = Array3D::new((p, q, r));
        irfft3_fftw(
            hadamard_product.get_data_mut(),
            r_potentials.get_data_mut(),
            &[p, q, r],
        );

        // Unpermute the coefficients
        let surface_multi_index_axial_diag = fft
            .operator_data
            .permuted_multi_indices
            .get(&transfer_vector.hash)
            .unwrap();

        let mut tmp = Vec::new();
        let ntargets = surface_multi_index_axial_diag.len() / 3;
        let xs = &surface_multi_index_axial_diag[0..ntargets];
        let ys = &surface_multi_index_axial_diag[ntargets..2 * ntargets];
        let zs = &surface_multi_index_axial_diag[2 * ntargets..];

        for i in 0..ntargets {
            let val = r_potentials.get(zs[i], ys[i], xs[i]).unwrap();
            tmp.push(*val);
        }

        // Find source and target surfaces
        let sources = transfer_vector
            .source
            .compute_surface(&domain, order, alpha);
        let targets = transfer_vector
            .target
            .compute_surface(&domain, order, alpha);

        // Get direct evaluations for testing
        let mut direct = vec![0f64; ncoeffs];
        fft.kernel.evaluate_st(
            EvalType::Value,
            &sources[..],
            &targets[..],
            multipole.data(),
            &mut direct[..],
        );

        let abs_error: f64 = tmp
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

        assert!(rel_error < 1e-15);
    }
}
