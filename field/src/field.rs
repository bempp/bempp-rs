//! Implementation of traits for field translations via the FFT and SVD.
use bempp_traits::kernel::ScaleInvariantKernel;
use itertools::Itertools;
use num::Zero;
use num::{Complex, Float};
use rlst_blis::interface::gemm::Gemm;
use rlst_common::types::Scalar;
use rlst_dense::{
    array::{empty_array, Array},
    base_array::BaseArray,
    data_container::VectorContainer,
    linalg::svd::SvdMode,
    rlst_dynamic_array2, rlst_dynamic_array3,
    traits::{
        MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, Shape, UnsafeRandomAccessByRef,
        UnsafeRandomAccessMut,
    },
};
use std::collections::HashSet;

use bempp_traits::{field::FieldTranslationData, kernel::Kernel, types::EvalType};
use bempp_tree::{
    implementations::helpers::find_corners, types::domain::Domain, types::morton::MortonKey,
};

use crate::types::{
    SvdFieldTranslationKiFmmIA, SvdFieldTranslationKiFmmRcmp, SvdM2lOperatorDataIA,
    SvdM2lOperatorDataRcmp,
};
use crate::{
    array::flip3,
    fft::Fft,
    transfer_vector::compute_transfer_vectors,
    types::{
        FftFieldTranslationKiFmm, FftM2lOperatorData, SvdFieldTranslationKiFmm, SvdM2lOperatorData,
        TransferVector,
    },
};

impl<T, U> FieldTranslationData<U> for SvdFieldTranslationKiFmm<T, U>
where
    T: Float + Default,
    T: Scalar<Real = T> + Gemm,
    U: Kernel<T = T> + Default,
    Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>: MatrixSvd<Item = T>,
{
    type TransferVector = Vec<TransferVector>;
    type M2LOperators = SvdM2lOperatorData<T>;
    type Domain = Domain<T>;

    fn ncoeffs(&self, order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }

    fn compute_m2l_operators<'a>(
        &self,
        order: usize,
        domain: Self::Domain,
        _depth: u64,
    ) -> Self::M2LOperators {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = self.ncoeffs(order);
        let ncols = self.ncoeffs(order);

        let ntransfer_vectors = self.transfer_vectors.len();
        let mut se2tc_fat = rlst_dynamic_array2!(T, [nrows, ncols * ntransfer_vectors]);
        let mut se2tc_thin = rlst_dynamic_array2!(T, [nrows * ntransfer_vectors, ncols]);

        for (i, t) in self.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface = t.source.compute_surface(&domain, order, self.alpha);
            let nsources = source_equivalent_surface.len() / self.kernel.space_dimension();

            let target_check_surface = t.target.compute_surface(&domain, order, self.alpha);
            let ntargets = target_check_surface.len() / self.kernel.space_dimension();

            let mut tmp_gram_t = rlst_dynamic_array2!(T, [ntargets, nsources]);

            self.kernel.assemble_st(
                EvalType::Value,
                &source_equivalent_surface[..],
                &target_check_surface[..],
                tmp_gram_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut tmp_gram = rlst_dynamic_array2!(T, [nsources, ntargets]);
            tmp_gram.fill_from(tmp_gram_t.transpose());

            let mut block = se2tc_fat
                .view_mut()
                .into_subview([0, i * ncols], [nrows, ncols]);
            block.fill_from(tmp_gram.view());

            let mut block_column = se2tc_thin
                .view_mut()
                .into_subview([i * nrows, 0], [nrows, ncols]);
            block_column.fill_from(tmp_gram.view());
        }

        let mu = se2tc_fat.shape()[0];
        let nvt = se2tc_fat.shape()[1];
        let k = std::cmp::min(mu, nvt);

        let mut u_big = rlst_dynamic_array2!(T, [mu, k]);
        let mut sigma = vec![T::zero(); k];
        let mut vt_big = rlst_dynamic_array2!(T, [k, nvt]);

        se2tc_fat
            .into_svd_alloc(
                u_big.view_mut(),
                vt_big.view_mut(),
                &mut sigma[..],
                SvdMode::Reduced,
            )
            .unwrap();

        let mut u = rlst_dynamic_array2!(T, [mu, self.k]);
        let mut sigma_mat = rlst_dynamic_array2!(T, [self.k, self.k]);
        let mut vt = rlst_dynamic_array2!(T, [self.k, nvt]);

        u.fill_from(u_big.into_subview([0, 0], [mu, self.k]));
        vt.fill_from(vt_big.into_subview([0, 0], [self.k, nvt]));
        for (j, s) in sigma.iter().enumerate().take(self.k) {
            unsafe {
                *sigma_mat.get_unchecked_mut([j, j]) = T::from(*s).unwrap();
            }
        }

        // Store compressed M2L operators
        let thin_nrows = se2tc_thin.shape()[0];
        let nst = se2tc_thin.shape()[1];
        let k = std::cmp::min(thin_nrows, nst);
        let mut _gamma = rlst_dynamic_array2!(T, [thin_nrows, k]);
        let mut _r = vec![T::zero(); k];
        let mut st = rlst_dynamic_array2!(T, [k, nst]);

        se2tc_thin
            .into_svd_alloc(
                _gamma.view_mut(),
                st.view_mut(),
                &mut _r[..],
                SvdMode::Reduced,
            )
            .unwrap();

        let mut s_block = rlst_dynamic_array2!(T, [nst, self.k]);
        for j in 0..self.k {
            for i in 0..nst {
                unsafe { *s_block.get_unchecked_mut([i, j]) = *st.get_unchecked([j, i]) }
            }
        }

        let mut c = rlst_dynamic_array2!(T, [self.k, self.k * ntransfer_vectors]);

        for i in 0..self.transfer_vectors.len() {
            let vt_block = vt.view().into_subview([0, i * ncols], [self.k, ncols]);

            let tmp = empty_array::<T, 2>().simple_mult_into_resize(
                sigma_mat.view(),
                empty_array::<T, 2>().simple_mult_into_resize(vt_block.view(), s_block.view()),
            );

            c.view_mut()
                .into_subview([0, i * self.k], [self.k, self.k])
                .fill_from(tmp);
        }

        let mut st_block = rlst_dynamic_array2!(T, [self.k, nst]);
        st_block.fill_from(s_block.transpose());

        SvdM2lOperatorData { u, st_block, c }
    }
}

impl<T, U> FieldTranslationData<U> for SvdFieldTranslationKiFmmRcmp<T, U>
where
    T: Float + Default,
    T: Scalar<Real = T> + Gemm,
    U: Kernel<T = T> + Default,
    Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>: MatrixSvd<Item = T>,
{
    type TransferVector = Vec<TransferVector>;
    type M2LOperators = SvdM2lOperatorDataRcmp<T>;
    type Domain = Domain<T>;

    fn ncoeffs(&self, order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }

    fn compute_m2l_operators<'a>(
        &self,
        order: usize,
        domain: Self::Domain,
        _depth: u64,
    ) -> Self::M2LOperators {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = self.ncoeffs(order);
        let ncols = self.ncoeffs(order);

        let ntransfer_vectors = self.transfer_vectors.len();
        let mut se2tc_fat = rlst_dynamic_array2!(T, [nrows, ncols * ntransfer_vectors]);
        let mut se2tc_thin = rlst_dynamic_array2!(T, [nrows * ntransfer_vectors, ncols]);

        for (i, t) in self.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface = t.source.compute_surface(&domain, order, self.alpha);
            let nsources = source_equivalent_surface.len() / self.kernel.space_dimension();

            let target_check_surface = t.target.compute_surface(&domain, order, self.alpha);
            let ntargets = target_check_surface.len() / self.kernel.space_dimension();

            let mut tmp_gram_t = rlst_dynamic_array2!(T, [ntargets, nsources]);

            self.kernel.assemble_st(
                EvalType::Value,
                &source_equivalent_surface[..],
                &target_check_surface[..],
                tmp_gram_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut tmp_gram = rlst_dynamic_array2!(T, [nsources, ntargets]);
            tmp_gram.fill_from(tmp_gram_t.transpose());

            let mut block = se2tc_fat
                .view_mut()
                .into_subview([0, i * ncols], [nrows, ncols]);
            block.fill_from(tmp_gram.view());

            let mut block_column = se2tc_thin
                .view_mut()
                .into_subview([i * nrows, 0], [nrows, ncols]);
            block_column.fill_from(tmp_gram.view());
        }

        let mu = se2tc_fat.shape()[0];
        let nvt = se2tc_fat.shape()[1];
        let k = std::cmp::min(mu, nvt);

        let mut u_big = rlst_dynamic_array2!(T, [mu, k]);
        let mut sigma = vec![T::zero(); k];
        let mut vt_big = rlst_dynamic_array2!(T, [k, nvt]);

        se2tc_fat
            .into_svd_alloc(
                u_big.view_mut(),
                vt_big.view_mut(),
                &mut sigma[..],
                SvdMode::Reduced,
            )
            .unwrap();

        let mut u = rlst_dynamic_array2!(T, [mu, self.k]);
        let mut sigma_mat = rlst_dynamic_array2!(T, [self.k, self.k]);
        let mut vt = rlst_dynamic_array2!(T, [self.k, nvt]);

        u.fill_from(u_big.into_subview([0, 0], [mu, self.k]));
        vt.fill_from(vt_big.into_subview([0, 0], [self.k, nvt]));
        for (j, s) in sigma.iter().enumerate().take(self.k) {
            unsafe {
                *sigma_mat.get_unchecked_mut([j, j]) = T::from(*s).unwrap();
            }
        }

        // Store compressed M2L operators
        let thin_nrows = se2tc_thin.shape()[0];
        let nst = se2tc_thin.shape()[1];
        let k = std::cmp::min(thin_nrows, nst);
        let mut _gamma = rlst_dynamic_array2!(T, [thin_nrows, k]);
        let mut _r = vec![T::zero(); k];
        let mut st = rlst_dynamic_array2!(T, [k, nst]);

        se2tc_thin
            .into_svd_alloc(
                _gamma.view_mut(),
                st.view_mut(),
                &mut _r[..],
                SvdMode::Reduced,
            )
            .unwrap();

        let mut s_block = rlst_dynamic_array2!(T, [nst, self.k]);
        for j in 0..self.k {
            for i in 0..nst {
                unsafe { *s_block.get_unchecked_mut([i, j]) = *st.get_unchecked([j, i]) }
            }
        }

        let mut c_u = Vec::new();
        let mut c_vt = Vec::new();

        for i in 0..self.transfer_vectors.len() {
            let vt_block = vt.view().into_subview([0, i * ncols], [self.k, ncols]);

            let tmp = empty_array::<T, 2>().simple_mult_into_resize(
                sigma_mat.view(),
                empty_array::<T, 2>().simple_mult_into_resize(vt_block.view(), s_block.view()),
            );

            let mut u_i = rlst_dynamic_array2!(T, [self.k, self.k]);
            let mut sigma_i = vec![T::zero(); self.k];
            let mut vt_i = rlst_dynamic_array2!(T, [self.k, self.k]);

            tmp.into_svd_alloc(u_i.view_mut(), vt_i.view_mut(), &mut sigma_i, SvdMode::Full)
                .unwrap();

            let rank = retain_energy(&sigma_i, self.threshold);

            let mut u_i_compressed = rlst_dynamic_array2!(T, [self.k, rank]);
            let mut vt_i_compressed_ = rlst_dynamic_array2!(T, [rank, self.k]);

            let mut sigma_mat_i_compressed = rlst_dynamic_array2!(T, [rank, rank]);

            u_i_compressed.fill_from(u_i.into_subview([0, 0], [self.k, rank]));
            vt_i_compressed_.fill_from(vt_i.into_subview([0, 0], [rank, self.k]));

            for (j, s) in sigma_i.iter().enumerate().take(rank) {
                unsafe {
                    *sigma_mat_i_compressed.get_unchecked_mut([j, j]) = T::from(*s).unwrap();
                }
            }

            let vt_i_compressed = empty_array::<T, 2>()
                .simple_mult_into_resize(sigma_mat_i_compressed.view(), vt_i_compressed_.view());

            c_u.push(u_i_compressed);
            c_vt.push(vt_i_compressed);
        }

        let mut st_block = rlst_dynamic_array2!(T, [self.k, nst]);
        st_block.fill_from(s_block.transpose());

        SvdM2lOperatorDataRcmp {
            u,
            st_block,
            c_u,
            c_vt,
        }
    }
}

impl<T, U> FieldTranslationData<U> for SvdFieldTranslationKiFmmIA<T, U>
where
    T: Float + Default,
    T: Scalar<Real = T> + Gemm,
    U: Kernel<T = T> + ScaleInvariantKernel<T = T> + Default,
    Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>: MatrixSvd<Item = T>,
{
    type TransferVector = Vec<TransferVector>;
    type M2LOperators = SvdM2lOperatorDataIA<T>;
    type Domain = Domain<T>;

    fn ncoeffs(&self, order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }

    fn compute_m2l_operators<'a>(
        &self,
        order: usize,
        domain: Self::Domain,
        depth: u64,
    ) -> Self::M2LOperators {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = self.ncoeffs(order);
        let ncols = self.ncoeffs(order);

        let mut u = Vec::new();

        let mut vt = Vec::new();

        for _i in 2..=depth {
            let mut tmp_i = Vec::new();
            for _j in 0..316 {
                let tmp_ij = rlst_dynamic_array2!(T, [1, 1]);
                tmp_i.push(tmp_ij);
            }
            vt.push(tmp_i)
        }
        // let mut vt = vec![vec![tmp; 316]; (depth - 1) as usize];

        for (c_idx, t) in self.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface = t.source.compute_surface(&domain, order, self.alpha);
            let target_check_surface = t.target.compute_surface(&domain, order, self.alpha);

            let mut tmp_gram_t = rlst_dynamic_array2!(T, [nrows, ncols]);

            self.kernel.assemble_st(
                EvalType::Value,
                &target_check_surface[..],
                &source_equivalent_surface[..],
                tmp_gram_t.data_mut(),
            );

            let mut u_i = rlst_dynamic_array2!(T, [nrows, self.k]);
            let mut sigma_i = vec![T::zero(); self.k];
            let mut vt_i = rlst_dynamic_array2!(T, [self.k, ncols]);

            tmp_gram_t
                .into_svd_alloc(u_i.view_mut(), vt_i.view_mut(), &mut sigma_i, SvdMode::Full)
                .unwrap();

            // Retain such that 95% of energy of singular values is retained.
            let rank = retain_energy(&sigma_i, self.threshold);

            let mut u_i_compressed = rlst_dynamic_array2!(T, [nrows, rank]);
            let mut vt_i_compressed_ = rlst_dynamic_array2!(T, [rank, ncols]);

            let mut sigma_mat_i_compressed = rlst_dynamic_array2!(T, [rank, rank]);

            u_i_compressed.fill_from(u_i.into_subview([0, 0], [nrows, rank]));
            vt_i_compressed_.fill_from(vt_i.into_subview([0, 0], [rank, ncols]));

            for (j, s) in sigma_i.iter().enumerate().take(rank) {
                unsafe {
                    *sigma_mat_i_compressed.get_unchecked_mut([j, j]) = T::from(*s).unwrap();
                }
            }

            let vt_i_compressed = empty_array::<T, 2>()
                .simple_mult_into_resize(sigma_mat_i_compressed.view(), vt_i_compressed_.view());

            for (level_idx, level) in (2..=depth).enumerate() {
                let scale = self.kernel.scale(level) * m2l_scale(level);

                // let mut vt_i_compressed_scaled = vec![T::zero(); vt_i_compressed.data().len()];
                let mut vt_i_compressed_scaled = rlst_dynamic_array2!(T, vt_i_compressed.shape());
                vt_i_compressed_scaled
                    .data_mut()
                    .iter_mut()
                    .zip(vt_i_compressed.data())
                    .for_each(|(v, v_)| *v = scale * *v_);
                // println!("HERE {:?} {:?}", vt_i_compressed.shape(), scale_mat.shape());

                vt[level_idx][c_idx] = vt_i_compressed_scaled
            }

            // Store compressed M2L oeprators
            u.push(u_i_compressed);
            // vt.push(vt_i_compressed);
        }

        SvdM2lOperatorDataIA { u, vt }
    }
}

fn m2l_scale<T>(level: u64) -> T
where
    T: Float + Default,
    T: Scalar<Real = T> + Gemm,
{
    if level < 2 {
        panic!("M2L only perfomed on level 2 and below")
    }

    if level == 2 {
        T::from(1. / 2.).unwrap()
    } else {
        let two = T::from(2.0).unwrap();
        Scalar::powf(two, T::from(level - 3).unwrap())
    }
}

fn retain_energy<T: Float + Default + Scalar<Real = T> + Gemm>(
    singular_values: &[T],
    percentage: T,
) -> usize {
    // Calculate the total energy.
    let total_energy: T = singular_values.iter().map(|&s| s * s).sum();

    // Calculate the threshold energy to retain.
    let threshold_energy = total_energy * (percentage / T::one());

    // Iterate over singular values to find the minimum set that retains the desired energy.
    let mut cumulative_energy = T::zero();
    let mut significant_values = Vec::new();

    for (i, &value) in singular_values.iter().enumerate() {
        cumulative_energy += value * value;
        significant_values.push(value);
        if cumulative_energy >= threshold_energy {
            return i + 1;
        }
    }

    significant_values.len()
}

impl<T, U> SvdFieldTranslationKiFmm<T, U>
where
    T: Float + Default,
    T: Scalar<Real = T> + rlst_blis::interface::gemm::Gemm,
    U: Kernel<T = T> + Default,
    Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>: MatrixSvd<Item = T>,
{
    /// Constructor for SVD field translation struct for the kernel independent FMM (KiFMM).
    ///
    /// # Arguments
    /// * `kernel` - The kernel being used, only compatible with homogenous, translationally invariant kernels.
    /// * `k` - The maximum rank to be used in SVD compression for the translation operators, if none is specified will be taken as  max({50, max_column_rank})
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `domain` - Domain associated with the global point set.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    pub fn new(kernel: U, k: Option<usize>, order: usize, domain: Domain<T>, alpha: T) -> Self {
        let mut result = SvdFieldTranslationKiFmm {
            alpha,
            k: 0,
            kernel,
            operator_data: SvdM2lOperatorData::default(),
            transfer_vectors: vec![],
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
        result.transfer_vectors = compute_transfer_vectors();
        result.operator_data = result.compute_m2l_operators(order, domain, 0);

        result
    }
}

impl<T, U> SvdFieldTranslationKiFmmRcmp<T, U>
where
    T: Float + Default,
    T: Scalar<Real = T> + rlst_blis::interface::gemm::Gemm,
    U: Kernel<T = T> + Default,
    Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>: MatrixSvd<Item = T>,
{
    /// Constructor for SVD field translation struct for the kernel independent FMM (KiFMM).
    ///
    /// # Arguments
    /// * `kernel` - The kernel being used, only compatible with homogenous, translationally invariant kernels.
    /// * `k` - The maximum rank to be used in SVD compression for the translation operators before recompression, if none is specified will be taken as  max({50, max_column_rank})
    /// * `threshold` - Percentage of energy to be retained from SVD of a given M2L operator, calculated from sum of squares of singular values during recompression.
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `domain` - Domain associated with the global point set.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    pub fn new(
        kernel: U,
        k: Option<usize>,
        threshold: T,
        order: usize,
        domain: Domain<T>,
        alpha: T,
    ) -> Self {
        let mut result = SvdFieldTranslationKiFmmRcmp {
            alpha,
            k: 0,
            threshold,
            kernel,
            operator_data: SvdM2lOperatorDataRcmp::default(),
            transfer_vectors: vec![],
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
        result.transfer_vectors = compute_transfer_vectors();
        result.operator_data = result.compute_m2l_operators(order, domain, 0);

        result
    }
}

impl<T, U> SvdFieldTranslationKiFmmIA<T, U>
where
    T: Float + Default,
    T: Scalar<Real = T> + rlst_blis::interface::gemm::Gemm,
    U: Kernel<T = T> + Default + ScaleInvariantKernel<T = T>,
    Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>: MatrixSvd<Item = T>,
{
    /// Constructor for SVD field translation struct for the kernel independent FMM (KiFMM).
    ///
    /// # Arguments
    /// * `kernel` - The kernel being used, only compatible with homogenous, translationally invariant kernels.
    /// * `threshold` - Percentage of energy to be retained from SVD of a given M2L operator, calculated from sum of squares of singular values.
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `domain` - Domain associated with the global point set.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    pub fn new(
        kernel: U,
        threshold: T,
        order: usize,
        domain: Domain<T>,
        alpha: T,
        depth: u64,
    ) -> Self {
        let mut result = SvdFieldTranslationKiFmmIA {
            alpha,
            k: 0,
            threshold,
            kernel,
            operator_data: SvdM2lOperatorDataIA::default(),
            transfer_vectors: vec![],
        };

        let ncoeffs = result.ncoeffs(order);
        result.k = ncoeffs;
        result.transfer_vectors = compute_transfer_vectors();
        result.operator_data = result.compute_m2l_operators(order, domain, depth);

        result
    }
}

impl<T, U> FieldTranslationData<U> for FftFieldTranslationKiFmm<T, U>
where
    T: Scalar<Real = T> + Float + Default + Fft,
    Complex<T>: Scalar,
    U: Kernel<T = T> + Default,
{
    type Domain = Domain<T>;

    type M2LOperators = FftM2lOperatorData<Complex<T>>;

    type TransferVector = Vec<TransferVector>;

    fn compute_m2l_operators(
        &self,
        order: usize,
        domain: Self::Domain,
        _depth: u64,
    ) -> Self::M2LOperators {
        // Parameters related to the FFT and Tree
        let m = 2 * order - 1; // Size of each dimension of 3D kernel/signal
        let pad_size = 1;
        let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        let size_real = p * p * (p / 2 + 1); // Number of Fourier coefficients when working with real data
        let nsiblings = 8; // Number of siblings for a given tree node
        let nconvolutions = nsiblings * nsiblings; // Number of convolutions computed for each node

        // Pick a point in the middle of the domain
        let two = T::from(2.0).unwrap();
        let midway = domain.diameter.iter().map(|d| *d / two).collect_vec();
        let point = midway
            .iter()
            .zip(domain.origin)
            .map(|(m, o)| *m + o)
            .collect_vec();
        let point = [point[0], point[1], point[2]];

        // Encode point in centre of domain and compute halo of parent, and their resp. children
        let key = MortonKey::from_point(&point, &domain, 3);
        let siblings = key.siblings();
        let parent = key.parent();
        let halo = parent.neighbors();
        let halo_children = halo.iter().map(|h| h.children()).collect_vec();

        // The child boxes in the halo of the sibling set
        let mut sources = vec![];
        // The sibling set
        let mut targets = vec![];
        // The transfer vectors corresponding to source->target translations
        let mut transfer_vectors = vec![];
        // Green's function evaluations for each source, target pair interaction
        let mut kernel_data_vec = vec![];

        for _ in &halo_children {
            sources.push(vec![]);
            targets.push(vec![]);
            transfer_vectors.push(vec![]);
            kernel_data_vec.push(vec![]);
        }

        // Each set of 64 M2L operators will correspond to a point in the halo
        // Computing transfer of potential from sibling set to halo
        for (i, halo_child_set) in halo_children.iter().enumerate() {
            let mut tmp_transfer_vectors = vec![];
            let mut tmp_targets = vec![];
            let mut tmp_sources = vec![];

            // Consider all halo children for a given sibling at a time
            for sibling in siblings.iter() {
                for halo_child in halo_child_set.iter() {
                    tmp_transfer_vectors.push(halo_child.find_transfer_vector(sibling));
                    tmp_targets.push(sibling);
                    tmp_sources.push(halo_child);
                }
            }

            // From source to target
            transfer_vectors[i] = tmp_transfer_vectors;
            targets[i] = tmp_targets;
            sources[i] = tmp_sources;
        }

        let n_source_equivalent_surface = 6 * (order - 1).pow(2) + 2;
        let n_target_check_surface = n_source_equivalent_surface;
        let n_corners = 8;

        // Iterate over each set of convolutions in the halo (26)
        for i in 0..transfer_vectors.len() {
            // Iterate over each unique convolution between sibling set, and halo siblings (64)
            for j in 0..transfer_vectors[i].len() {
                // Translating from sibling set to boxes in its M2L halo
                let target = targets[i][j];
                let source = sources[i][j];

                let source_equivalent_surface = source.compute_surface(&domain, order, self.alpha);
                let target_check_surface = target.compute_surface(&domain, order, self.alpha);

                let v_list: HashSet<MortonKey> = target
                    .parent()
                    .neighbors()
                    .iter()
                    .flat_map(|pn| pn.children())
                    .filter(|pnc| !target.is_adjacent(pnc))
                    .collect();

                if v_list.contains(source) {
                    // Compute convolution grid around the source box
                    let conv_point_corner_index = 7;
                    let corners = find_corners(&source_equivalent_surface[..]);
                    let conv_point_corner = [
                        corners[conv_point_corner_index],
                        corners[n_corners + conv_point_corner_index],
                        corners[2 * n_corners + conv_point_corner_index],
                    ];

                    let (conv_grid, _) = source.convolution_grid(
                        order,
                        &domain,
                        self.alpha,
                        &conv_point_corner,
                        conv_point_corner_index,
                    );

                    // Calculate Green's fct evaluations with respect to a 'kernel point' on the target box
                    let kernel_point_index = 0;
                    let kernel_point = [
                        target_check_surface[kernel_point_index],
                        target_check_surface[n_target_check_surface + kernel_point_index],
                        target_check_surface[2 * n_target_check_surface + kernel_point_index],
                    ];

                    // Compute Green's fct evaluations
                    let kernel = self.compute_kernel(order, &conv_grid, kernel_point);

                    let mut kernel = flip3(&kernel);

                    // Compute FFT of padded kernel
                    let mut kernel_hat = rlst_dynamic_array3!(Complex<T>, [p, p, p / 2 + 1]);

                    // TODO: is kernel_hat the transpose of what it used to be?
                    T::rfft3_fftw(kernel.data_mut(), kernel_hat.data_mut(), &[p, p, p]);

                    kernel_data_vec[i].push(kernel_hat);
                } else {
                    // Fill with zeros when interaction doesn't exist
                    let n = 2 * order - 1;
                    let p = n + 1;
                    let kernel_hat_zeros = rlst_dynamic_array3!(Complex<T>, [p, p, p / 2 + 1]);
                    kernel_data_vec[i].push(kernel_hat_zeros);
                }
            }
        }

        // Each element corresponds to all evaluations for each sibling (in order) at that halo position
        let mut kernel_data =
            vec![vec![Complex::<T>::zero(); nconvolutions * size_real]; halo_children.len()];

        // For each halo position
        for i in 0..halo_children.len() {
            // For each unique interaction
            for j in 0..nconvolutions {
                let offset = j * size_real;
                kernel_data[i][offset..offset + size_real]
                    .copy_from_slice(kernel_data_vec[i][j].data())
            }
        }

        // We want to use this data by frequency in the implementation of FFT M2L
        // Rearrangement: Grouping by frequency, then halo child, then sibling
        let mut kernel_data_f = vec![];
        for _ in &halo_children {
            kernel_data_f.push(vec![]);
        }
        for i in 0..halo_children.len() {
            let current_vector = &kernel_data[i];
            for l in 0..size_real {
                // halo child
                for k in 0..8 {
                    // sibling
                    for j in 0..8 {
                        let index = j * size_real * 8 + k * size_real + l;
                        kernel_data_f[i].push(current_vector[index]);
                    }
                }
            }
        }

        FftM2lOperatorData {
            kernel_data,
            kernel_data_f,
        }
    }

    fn ncoeffs(&self, order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }
}

impl<T, U> FftFieldTranslationKiFmm<T, U>
where
    T: Float + Scalar<Real = T> + Default + Fft,
    Complex<T>: Scalar,
    U: Kernel<T = T> + Default,
{
    /// Constructor for FFT field translation struct for the kernel independent FMM (KiFMM).
    ///
    /// # Arguments
    /// * `kernel` - The kernel being used, only compatible with homogenous, translationally invariant kernels.
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `domain` - Domain associated with the global point set.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    pub fn new(kernel: U, order: usize, domain: Domain<T>, alpha: T) -> Self {
        let mut result = FftFieldTranslationKiFmm {
            alpha,
            kernel,
            surf_to_conv_map: Vec::default(),
            conv_to_surf_map: Vec::default(),
            operator_data: FftM2lOperatorData::default(),
            transfer_vectors: Vec::default(),
        };

        // Create maps between surface and convolution grids
        let (surf_to_conv, conv_to_surf) =
            FftFieldTranslationKiFmm::<T, U>::compute_surf_to_conv_map(order);

        result.surf_to_conv_map = surf_to_conv;
        result.conv_to_surf_map = conv_to_surf;
        result.transfer_vectors = compute_transfer_vectors();

        result.operator_data = result.compute_m2l_operators(order, domain, 0);

        result
    }

    /// Compute map between convolution grid indices and surface indices, return mapping and inverse mapping.
    ///
    /// # Arguments
    /// * `order` - The expansion order for the multipole and local expansions.
    pub fn compute_surf_to_conv_map(order: usize) -> (Vec<usize>, Vec<usize>) {
        // Number of points along each axis of convolution grid
        let n = 2 * order - 1;
        let npad = n + 1;

        let nsurf_grid = 6 * (order - 1).pow(2) + 2;

        // Index maps between surface and convolution grids
        let mut surf_to_conv = vec![0usize; nsurf_grid];
        let mut conv_to_surf = vec![0usize; nsurf_grid];

        // Initialise surface grid index
        let mut surf_index = 0;

        // The boundaries of the surface grid when embedded within the convolution grid
        let lower = order;
        let upper = 2 * order - 1;

        for k in 0..npad {
            for j in 0..npad {
                for i in 0..npad {
                    let conv_index = i + npad * j + npad * npad * k;
                    if (i >= lower && j >= lower && (k == lower || k == upper))
                        || (j >= lower && k >= lower && (i == lower || i == upper))
                        || (k >= lower && i >= lower && (j == lower || j == upper))
                    {
                        surf_to_conv[surf_index] = conv_index;
                        surf_index += 1;
                    }
                }
            }
        }

        let lower = 0;
        let upper = order - 1;
        let mut surf_index = 0;

        for k in 0..npad {
            for j in 0..npad {
                for i in 0..npad {
                    let conv_index = i + npad * j + npad * npad * k;
                    if (i <= upper && j <= upper && (k == lower || k == upper))
                        || (j <= upper && k <= upper && (i == lower || i == upper))
                        || (k <= upper && i <= upper && (j == lower || j == upper))
                    {
                        conv_to_surf[surf_index] = conv_index;
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
    /// * `convolution_grid` - Cartesian coordinates of points on the convolution grid at a source box, expected in column major order.
    /// * `target_pt` - The point on the target box's surface grid, with which kernels are being evaluated with respect to.
    pub fn compute_kernel(
        &self,
        order: usize,
        convolution_grid: &[T],
        target_pt: [T; 3],
    ) -> Array<T, BaseArray<T, VectorContainer<T>, 3>, 3> {
        let n = 2 * order - 1;
        let npad = n + 1;

        let mut result = rlst_dynamic_array3!(T, [npad, npad, npad]);

        let nconv = n.pow(3);
        let mut kernel_evals = vec![T::zero(); nconv];
        self.kernel.assemble_st(
            EvalType::Value,
            convolution_grid,
            &target_pt[..],
            &mut kernel_evals[..],
        );

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let conv_idx = i + j * n + k * n * n;
                    let save_idx = i + j * npad + k * npad * npad;
                    result.data_mut()[save_idx..(save_idx + 1)]
                        .copy_from_slice(&kernel_evals[(conv_idx)..(conv_idx + 1)]);
                }
            }
        }

        result
    }

    /// Place charge data on the convolution grid.
    ///
    /// # Arguments
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `charges` - A vector of charges.
    pub fn compute_signal(
        &self,
        order: usize,
        charges: &[T],
    ) -> Array<T, BaseArray<T, VectorContainer<T>, 3>, 3> {
        let n = 2 * order - 1;
        let npad = n + 1;

        let mut result = rlst_dynamic_array3!(T, [npad, npad, npad]);

        for (i, &j) in self.surf_to_conv_map.iter().enumerate() {
            result.data_mut()[j] = charges[i];
        }

        result
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::fft::Fft;
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use cauchy::{c32, c64};
    use num::complex::Complex;
    use rlst_dense::traits::{RandomAccessByRef, RandomAccessMut};

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
        let m2l = svd.compute_m2l_operators(order, domain, 0);

        // Test that the rank cutoff has been taken correctly (k < ncoeffs)
        assert_eq!(m2l.st_block.shape(), [k, svd.ncoeffs(order)]);
        assert_eq!(m2l.c.shape(), [k, k * ntransfer_vectors]);
        assert_eq!(m2l.u.shape(), [svd.ncoeffs(order), k]);

        // Test that the rank cutoff has been taken correctly (k > ncoeffs)
        let k = 100;
        let svd = SvdFieldTranslationKiFmm::new(kernel.clone(), Some(k), order, domain, alpha);
        let m2l = svd.compute_m2l_operators(order, domain, 0);
        assert_eq!(
            m2l.st_block.shape(),
            [svd.ncoeffs(order), svd.ncoeffs(order)]
        );
        assert_eq!(
            m2l.c.shape(),
            [svd.ncoeffs(order), svd.ncoeffs(order) * ntransfer_vectors]
        );
        assert_eq!(m2l.u.shape(), [svd.ncoeffs(order), svd.ncoeffs(order)]);

        // Test that the rank cutoff has been taken correctly (k unspecified)
        let k = None;
        let default_k = 50;
        let svd = SvdFieldTranslationKiFmm::new(kernel, k, order, domain, alpha);
        let m2l = svd.compute_m2l_operators(order, domain, 0);
        assert_eq!(m2l.st_block.shape(), [default_k, svd.ncoeffs(order)]);
        assert_eq!(m2l.c.shape(), [default_k, default_k * ntransfer_vectors]);
        assert_eq!(m2l.u.shape(), [svd.ncoeffs(order), default_k]);
    }

    #[test]
    pub fn test_fft_operator_data() {
        let kernel: Laplace3dKernel<f32> = Laplace3dKernel::<f32>::new();
        let order = 5;
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let alpha = 1.05;

        let fft = FftFieldTranslationKiFmm::new(kernel, order, domain, alpha);

        // Create a random point in the middle of the domain
        let m2l: FftM2lOperatorData<c32> = fft.compute_m2l_operators(order, domain, 0);
        let m = 2 * order - 1; // Size of each dimension of 3D kernel/signal
        let pad_size = 1;
        let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        let size_real = p * p * (p / 2 + 1); // Number of Fourier coefficients when working with real data

        // Test that the number of precomputed kernel interactions matches the number of halo postitions
        assert_eq!(m2l.kernel_data.len(), 26);

        // Test that each halo position has exactly 8x8 kernels associated with it
        for i in 0..26 {
            assert_eq!(m2l.kernel_data[i].len() / size_real, 64)
        }
    }

    #[test]
    fn test_svd_field_translation() {
        let kernel = Laplace3dKernel::new();
        let order: usize = 2;

        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let alpha = 1.05;

        // Some expansion data
        let ncoeffs = 6 * (order - 1).pow(2) + 2;
        let mut multipole = rlst_dynamic_array2!(f64, [ncoeffs, 1]);

        for i in 0..ncoeffs {
            *multipole.get_mut([i, 0]).unwrap() = i as f64;
        }

        // Create field translation object
        let svd = SvdFieldTranslationKiFmm::new(kernel, Some(1000), order, domain, alpha);

        // Pick a random source/target pair
        let idx = 153;
        let all_transfer_vectors = compute_transfer_vectors();

        let transfer_vector = &all_transfer_vectors[idx];

        // Lookup correct components of SVD compressed M2L operator matrix
        let c_idx = svd
            .transfer_vectors
            .iter()
            .position(|x| x.hash == transfer_vector.hash)
            .unwrap();

        let [nrows, _] = svd.operator_data.c.shape();
        let c_sub = svd
            .operator_data
            .c
            .into_subview([0, c_idx * svd.k], [nrows, svd.k]);

        let compressed_multipole = empty_array::<f64, 2>()
            .simple_mult_into_resize(svd.operator_data.st_block.view(), multipole.view());

        let compressed_check_potential = empty_array::<f64, 2>()
            .simple_mult_into_resize(c_sub.view(), compressed_multipole.view());

        // Post process to find check potential
        let check_potential = empty_array::<f64, 2>().simple_mult_into_resize(
            svd.operator_data.u.view(),
            compressed_check_potential.view(),
        );

        let sources = transfer_vector
            .source
            .compute_surface(&domain, order, alpha);
        let targets = transfer_vector
            .target
            .compute_surface(&domain, order, alpha);
        let mut direct = vec![0f64; ncoeffs];
        svd.kernel.evaluate_st(
            EvalType::Value,
            &sources[..],
            &targets[..],
            multipole.data(),
            &mut direct[..],
        );

        let abs_error: f64 = check_potential
            .data()
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

        assert!(rel_error < 1e-14);
    }

    fn m2l_scale(level: u64) -> f64 {
        if level < 2 {
            panic!("M2L only performed on level 2 and below")
        }
        if level == 2 {
            1. / 2.
        } else {
            2_f64.powf((level - 3) as f64)
        }
    }

    #[test]
    fn test_fft_operator_data_kernels() {
        let kernel = Laplace3dKernel::new();
        let order: usize = 2;

        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let alpha = 1.05;

        // Some expansion data998
        let ncoeffs = 6 * (order - 1).pow(2) + 2;
        let mut multipole = rlst_dynamic_array2!(f64, [ncoeffs, 1]);

        for i in 0..ncoeffs {
            *multipole.get_mut([i, 0]).unwrap() = i as f64;
        }

        let level = 2;
        // Create field translation object
        let fft = FftFieldTranslationKiFmm::new(kernel, order, domain, alpha);

        let kernels = &fft.operator_data.kernel_data;

        let key = MortonKey::from_point(&[0.5, 0.5, 0.5], &domain, level);

        let parent_neighbours = key.parent().neighbors();
        let mut v_list_structured = vec![];
        for _ in 0..26 {
            v_list_structured.push(vec![]);
        }
        for (i, pn) in parent_neighbours.iter().enumerate() {
            for child in pn.children() {
                if !key.is_adjacent(&child) {
                    v_list_structured[i].push(Some(child));
                } else {
                    v_list_structured[i].push(None)
                }
            }
        }

        // pick a halo position
        let halo_idx = 0;
        // pick a halo child position
        let halo_child_idx = 2;
        let n = 2 * order - 1;
        let p = n + 1;
        let size_real = p * p * (p / 2 + 1);

        // Find kernel from precomputation;
        let kernel_hat =
            &kernels[halo_idx][halo_child_idx * size_real..(halo_child_idx + 1) * size_real];

        // Apply scaling
        let scale = m2l_scale(level);
        let kernel_hat = kernel_hat.iter().map(|k| *k * scale).collect_vec();

        let target = key;
        let source = v_list_structured[halo_idx][halo_child_idx].unwrap();
        let source_equivalent_surface = source.compute_surface(&domain, order, fft.alpha);
        let target_check_surface = target.compute_surface(&domain, order, fft.alpha);
        let ntargets = target_check_surface.len() / 3;

        // Compute conv grid
        let conv_point_corner_index = 7;
        let corners = find_corners(&source_equivalent_surface[..]);
        let conv_point_corner = [
            corners[conv_point_corner_index],
            corners[8 + conv_point_corner_index],
            corners[16 + conv_point_corner_index],
        ];

        let (conv_grid, _) = source.convolution_grid(
            order,
            &domain,
            fft.alpha,
            &conv_point_corner,
            conv_point_corner_index,
        );

        let kernel_point_index = 0;
        let kernel_point = [
            target_check_surface[kernel_point_index],
            target_check_surface[ntargets + kernel_point_index],
            target_check_surface[2 * ntargets + kernel_point_index],
        ];

        // Compute kernel from source/target pair
        let test_kernel = fft.compute_kernel(order, &conv_grid, kernel_point);
        let [m, n, o] = test_kernel.shape();

        let mut test_kernel = flip3(&test_kernel);

        // Compute FFT of padded kernel
        let mut test_kernel_hat = rlst_dynamic_array3!(c64, [m, n, o / 2 + 1]);
        f64::rfft3_fftw(
            test_kernel.data_mut(),
            test_kernel_hat.data_mut(),
            &[m, n, o],
        );

        for (p, t) in test_kernel_hat.data().iter().zip(kernel_hat.iter()) {
            assert!((p - t).norm() < 1e-6)
        }
    }

    #[test]
    fn test_kernel_rearrangement() {
        // Dummy data mirroring unrearranged kernels
        // here each '1000' corresponds to a sibling index
        // each '100' to a child in a given halo element
        // and each '1' to a frequency
        let mut kernel_data_mat = vec![];
        for _ in 0..26 {
            kernel_data_mat.push(vec![]);
        }
        let size_real = 10;

        for elem in kernel_data_mat.iter_mut().take(26) {
            // sibling index
            for j in 0..8 {
                // halo child index
                for k in 0..8 {
                    // frequency
                    for l in 0..size_real {
                        elem.push(Complex::new((1000 * j + 100 * k + l) as f64, 0.))
                    }
                }
            }
        }

        // We want to use this data by frequency in the implementation of FFT M2L
        // Rearrangement: Grouping by frequency, then halo child, then sibling
        let mut rearranged = vec![];
        for _ in 0..26 {
            rearranged.push(vec![]);
        }
        for i in 0..26 {
            let current_vector = &kernel_data_mat[i];
            for l in 0..size_real {
                // halo child
                for k in 0..8 {
                    // sibling
                    for j in 0..8 {
                        let index = j * size_real * 8 + k * size_real + l;
                        rearranged[i].push(current_vector[index]);
                    }
                }
            }
        }

        // We expect the first 64 elements to correspond to the first frequency components of all
        // siblings with all elements in a given halo position
        let freq = 4;
        let offset = freq * 64;
        let result = &rearranged[0][offset..offset + 64];

        // For each halo child
        for i in 0..8 {
            // for each sibling
            for j in 0..8 {
                let expected = (i * 100 + j * 1000 + freq) as f64;
                assert!(expected == result[i * 8 + j].re)
            }
        }
    }

    #[test]
    fn test_fft_field_translation() {
        let kernel = Laplace3dKernel::new();
        let order: usize = 2;

        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [5., 5., 5.],
        };

        let alpha = 1.05;

        // Some expansion data
        let ncoeffs = 6 * (order - 1).pow(2) + 2;
        let mut multipole = rlst_dynamic_array2!(f64, [ncoeffs, 1]);

        for i in 0..ncoeffs {
            *multipole.get_mut([i, 0]).unwrap() = i as f64;
        }

        // Create field translation object
        let fft = FftFieldTranslationKiFmm::new(kernel, order, domain, alpha);

        // Compute all M2L operators

        // Pick a random source/target pair
        let idx = 123;
        let all_transfer_vectors = compute_transfer_vectors();

        let transfer_vector = &all_transfer_vectors[idx];

        // Compute FFT of the representative signal
        let mut signal = fft.compute_signal(order, multipole.data());
        let [m, n, o] = signal.shape();
        let mut signal_hat = rlst_dynamic_array3!(c64, [m, n, o / 2 + 1]);

        f64::rfft3_fftw(signal.data_mut(), signal_hat.data_mut(), &[m, n, o]);

        let source_equivalent_surface = transfer_vector
            .source
            .compute_surface(&domain, order, fft.alpha);
        let target_check_surface = transfer_vector
            .target
            .compute_surface(&domain, order, fft.alpha);
        let ntargets = target_check_surface.len() / 3;

        // Compute conv grid
        let conv_point_corner_index = 7;
        let corners = find_corners(&source_equivalent_surface[..]);
        let conv_point_corner = [
            corners[conv_point_corner_index],
            corners[8 + conv_point_corner_index],
            corners[16 + conv_point_corner_index],
        ];

        let (conv_grid, _) = transfer_vector.source.convolution_grid(
            order,
            &domain,
            fft.alpha,
            &conv_point_corner,
            conv_point_corner_index,
        );

        let kernel_point_index = 0;
        let kernel_point = [
            target_check_surface[kernel_point_index],
            target_check_surface[ntargets + kernel_point_index],
            target_check_surface[2 * ntargets + kernel_point_index],
        ];

        // Compute kernel
        let kernel = fft.compute_kernel(order, &conv_grid, kernel_point);
        let [m, n, o] = kernel.shape();

        let mut kernel = flip3(&kernel);

        // Compute FFT of padded kernel
        let mut kernel_hat = rlst_dynamic_array3!(c64, [m, n, o / 2 + 1]);
        f64::rfft3_fftw(kernel.data_mut(), kernel_hat.data_mut(), &[m, n, o]);

        let mut hadamard_product = rlst_dynamic_array3!(c64, [m, n, o / 2 + 1]);
        for k in 0..o / 2 + 1 {
            for j in 0..n {
                for i in 0..m {
                    *hadamard_product.get_mut([i, j, k]).unwrap() =
                        kernel_hat.get([i, j, k]).unwrap() * signal_hat.get([i, j, k]).unwrap();
                }
            }
        }

        let mut potentials = rlst_dynamic_array3!(f64, [m, n, o]);

        f64::irfft3_fftw(
            hadamard_product.data_mut(),
            potentials.data_mut(),
            &[m, n, o],
        );

        let mut result = vec![0f64; ntargets];
        for (i, &idx) in fft.conv_to_surf_map.iter().enumerate() {
            result[i] = potentials.data()[idx];
        }

        // Get direct evaluations for testing
        let mut direct = vec![0f64; ncoeffs];
        fft.kernel.evaluate_st(
            EvalType::Value,
            &source_equivalent_surface[..],
            &target_check_surface[..],
            multipole.data(),
            &mut direct[..],
        );

        let abs_error: f64 = result
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

        assert!(rel_error < 1e-15);
    }
}
