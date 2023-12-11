//! Implementation of traits for field translations via the FFT and SVD.
use cauchy::Scalar;
use itertools::Itertools;
use num::Zero;
use num::{Complex, Float};
use rlst_dense::{
    array::{empty_array, Array},
    base_array::BaseArray,
    data_container::SliceContainer,
    linalg::svd::SvdMode,
    rlst_dynamic_array1, rlst_dynamic_array2, rlst_dynamic_array3,
    traits::{MultIntoResize, RandomAccessByRef, RawAccess, RawAccessMut, Shape, RandomAccessMut},
};
use std::collections::{HashMap, HashSet};

use bempp_tools::Array3D;
use bempp_traits::{field::FieldTranslationData, kernel::Kernel, types::EvalType};
use bempp_tree::{
    implementations::helpers::find_corners, types::domain::Domain, types::morton::MortonKey,
};

use crate::{
    array::{flip3, pad3},
    fft::Fft,
    transfer_vector::compute_transfer_vectors,
    types::{
        FftFieldTranslationKiFmm, FftM2lOperatorData, FftMatrix, SvdFieldTranslationKiFmm,
        SvdM2lOperatorData, TransferVector,
    },
};

type T = f64;

// impl<T, U> FieldTranslationData<U> for SvdFieldTranslationKiFmm<T, U>
impl<U> FieldTranslationData<U> for SvdFieldTranslationKiFmm<T, U>
where
    T: Float + Default + Scalar<Real = T>,
    U: Kernel<T = T> + Default,
{
    type TransferVector = Vec<TransferVector>;
    type M2LOperators = SvdM2lOperatorData<T>;
    type Domain = Domain<T>;

    fn ncoeffs(&self, order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }

    fn compute_m2l_operators<'a>(&self, order: usize, domain: Self::Domain) -> Self::M2LOperators {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = self.ncoeffs(order);
        let ncols = self.ncoeffs(order);

        let ntransfer_vectors = self.transfer_vectors.len();
        let mut se2tc_fat = rlst_dynamic_array2![T, [nrows, ncols * ntransfer_vectors]];
        let mut se2tc_thin = rlst_dynamic_array2![T, [nrows * ntransfer_vectors, ncols]];

        for (i, t) in self.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface = t.source.compute_surface(&domain, order, self.alpha);
            let nsources = source_equivalent_surface.len() / self.kernel.space_dimension();

            let target_check_surface = t.target.compute_surface(&domain, order, self.alpha);
            let ntargets = target_check_surface.len() / self.kernel.space_dimension();

            let mut tmp_gram = rlst_dynamic_array2![T, [ntargets, nsources]];

            self.kernel.assemble_st(
                EvalType::Value,
                &source_equivalent_surface[..],
                &target_check_surface[..],
                tmp_gram.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let tmp_gram_t = tmp_gram.transpose();
            for x in 0..nrows {
                for y in 0..ncols {
                    *se2tc_fat.get_mut([x, i*ncols + y]).unwrap() = *tmp_gram_t.get([x, y]).unwrap();
                }
            }

            for x in 0..nrows {
                for y in 0..ncols {
                    *se2tc_thin.get_mut([i*nrows + x, y]).unwrap() = *tmp_gram_t.get([x, y]).unwrap();
                }
            }
        }

        let mut u = rlst_dynamic_array2!(T, [nrows, self.k]);
        let mut vt = rlst_dynamic_array2!(T, [self.k, ncols * ntransfer_vectors]);
        // let mut sigma = rlst_dynamic_array1!(T::Real, [self.k]);
        let mut sigma = rlst_dynamic_array1!(T, [self.k]);
        se2tc_fat
            .into_svd_alloc(u.view_mut(), vt.view_mut(), sigma.data_mut(), SvdMode::Full)
            .unwrap();

        // Keep 'k' singular values
        let mut sigma_mat = rlst_dynamic_array2![T, [self.k, self.k]];
        for i in 0..self.k {
            sigma_mat[[i, i]] = T::from(*sigma.get([i]).unwrap());
        }

        let [mu, _] = u.shape();
        let u_block = u.into_subview([0, 1], [mu, self.k]);
        let mut u = rlst_dynamic_array2!(T, [1, self.k - mu]);
        u.fill_from(u_block);

        let [_, nvt] = vt.shape();
        let vt = vt.into_subview([0, 1], [self.k, nvt]);
        let vt: Vec<T> = vt.data().iter().map(|&x| T::from(x)).collect();
        let vt = Array::new(BaseArray::new(SliceContainer::new(&vt), [self.k, nvt]));
        // unsafe { rlst_pointer_mat!['static, T, vt.as_ptr(), (self.k, nvt), (1, self.k)] };

        // Store compressed M2L operators
        let mut _r = rlst_dynamic_array2!(T, [nrows * ntransfer_vectors, self.k]);
        let mut st = rlst_dynamic_array2!(T, [self.k, ncols]);
        // let mut _gamma = rlst_dynamic_array1!(T::Real, [self.k]);
        let mut _gamma = rlst_dynamic_array1!(T, [self.k]);
        se2tc_thin
            .into_svd_alloc(
                _r.view_mut(),
                st.view_mut(),
                _gamma.data_mut(),
                SvdMode::Full,
            )
            .unwrap();

        let [_, nst] = st.shape();
        let st_block = st.into_subview([0, 1], [self.k, nst]);
        let mut s_block = rlst_dynamic_array2!(T, [self.k, nst]);
        s_block.fill_from(st_block.transpose());
        let s_block: Vec<T> = s_block
            .data()
            .iter()
            .map(|&x| T::from(x))
            .collect();
        let s_block = Array::new(BaseArray::new(SliceContainer::new(&s_block), [nst, self.k]));
        // unsafe { rlst_pointer_mat!['static, T, s_block.as_ptr(), (nst, self.k), (1, nst)] };

        let mut c = rlst_dynamic_array2![T, [self.k, self.k * ntransfer_vectors]];

        for i in 0..self.transfer_vectors.len() {
            let top_left = [0, i * ncols];
            let dim = [self.k, ncols];
            let vt_block = vt.view().into_subview(top_left, dim);

            let tmp = empty_array::<T, 2>().simple_mult_into_resize(
                empty_array::<T, 2>().simple_mult_into_resize(sigma_mat.view(), vt_block.view()),
                s_block.view(),
            ); // sigma_mat.dot(&vt_block.dot(&s_block));

            let top_left = [0, i * self.k];
            let dim = [self.k, self.k];

            c.view_mut().into_subview(top_left, dim)
                .data_mut()
                .copy_from_slice(tmp.data());
        }

        let mut st_block = rlst_dynamic_array2!(T, [self.k, nst]);
        st_block.fill_from(s_block.transpose());

        SvdM2lOperatorData { u, st_block, c }
    }
}

// impl<T, U> SvdFieldTranslationKiFmm<T, U>
impl<U> SvdFieldTranslationKiFmm<T, U>
where
    // T: Float + Default,
    // T: Scalar<Real = T>,
    U: Kernel<T = T> + Default,
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
            operator_data: SvdM2lOperatorData::<T>::default(),
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

        result.transfer_vectors = compute_transfer_vectors();
        result.operator_data = result.compute_m2l_operators(order, domain);

        result
    }
}

impl<T, U> FieldTranslationData<U> for FftFieldTranslationKiFmm<T, U>
where
    T: Scalar<Real = T> + Float + Default + Fft<FftMatrix<T>, FftMatrix<Complex<T>>>,
    Complex<T>: Scalar,
    U: Kernel<T = T> + Default,
{
    type Domain = Domain<T>;

    type M2LOperators = FftM2lOperatorData<Complex<T>>;

    type TransferVector = Vec<TransferVector>;

    fn compute_m2l_operators(&self, order: usize, domain: Self::Domain) -> Self::M2LOperators {
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
        let mut sources = vec![vec![]; halo_children.len()];

        // The sibling set
        let mut targets = vec![vec![]; halo_children.len()];

        // The transfer vectors corresponding to source->target translations
        let mut transfer_vectors = vec![vec![]; halo_children.len()];

        // Green's function evaluations for each source, target pair interaction
        let mut kernel_data_vec = vec![];
        for _ in 0..halo_children.len() {
            kernel_data_vec.push(vec![]);
        }

        // Each set of 64 M2L operators will correspond to a point in the halo
        // Computing transfer of potential from sibling set to halo
        for (i, halo_child_set) in halo_children.iter().enumerate() {
            let mut tmp_transfer_vectors = Vec::new();
            let mut tmp_targets = Vec::new();
            let mut tmp_sources = Vec::new();

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

                    let padded_kernel = pad3(&kernel, (p - m, p - m, p - m), (0, 0, 0));
                    let mut padded_kernel = flip3(&padded_kernel);

                    // Compute FFT of padded kernel
                    let mut padded_kernel_hat = rlst_dynamic_array3!(Complex<T>, [p, p, p / 2 + 1]);

                    T::rfft3_fftw(
                        padded_kernel.data_mut(),
                        padded_kernel_hat.data_mut(),
                        &[p, p, p],
                    );

                    kernel_data_vec[i].push(padded_kernel_hat);
                } else {
                    // Fill with zeros when interaction doesn't exist
                    let n = 2 * order - 1;
                    let p = n + 1;
                    let padded_kernel_hat_zeros =
                        rlst_dynamic_array3!(Complex<T>, [p, p, p / 2 + 1]);
                    kernel_data_vec[i].push(padded_kernel_hat_zeros);
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
        let mut kernel_data_rearranged = vec![Vec::new(); halo_children.len()];
        for i in 0..halo_children.len() {
            let current_vector = &kernel_data[i];
            for l in 0..size_real {
                // halo child
                for k in 0..8 {
                    // sibling
                    for j in 0..8 {
                        let index = j * size_real * 8 + k * size_real + l;
                        kernel_data_rearranged[i].push(current_vector[index]);
                    }
                }
            }
        }

        FftM2lOperatorData {
            kernel_data,
            kernel_data_rearranged,
        }
    }

    fn ncoeffs(&self, order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }
}

impl<T, U> FftFieldTranslationKiFmm<T, U>
where
    T: Float + Scalar<Real = T> + Default + Fft<FftMatrix<T>, FftMatrix<Complex<T>>>,
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
            surf_to_conv_map: HashMap::default(),
            conv_to_surf_map: HashMap::default(),
            operator_data: FftM2lOperatorData::default(),
            transfer_vectors: Vec::default(),
        };

        // Create maps between surface and convolution grids
        let (surf_to_conv, conv_to_surf) =
            FftFieldTranslationKiFmm::<T, U>::compute_surf_to_conv_map(order);

        result.surf_to_conv_map = surf_to_conv;
        result.conv_to_surf_map = conv_to_surf;
        result.transfer_vectors = compute_transfer_vectors();

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
    /// * `convolution_grid` - Cartesian coordinates of points on the convolution grid at a source box, expected in column major order.
    /// * `target_pt` - The point on the target box's surface grid, with which kernels are being evaluated with respect to.
    pub fn compute_kernel(
        &self,
        order: usize,
        convolution_grid: &[T],
        target_pt: [T; 3],
    ) -> Array3D<T> {
        let n = 2 * order - 1;
        let mut result = rlst_dynamic_array3!(T, [n, n, n]);
        let nconv = n.pow(3);

        let mut kernel_evals = vec![T::zero(); nconv];

        self.kernel.assemble_st(
            EvalType::Value,
            convolution_grid,
            &target_pt[..],
            &mut kernel_evals[..],
        );

        result.data_mut().copy_from_slice(&kernel_evals[..]);

        result
    }

    /// Place charge data on the convolution grid.
    ///
    /// # Arguments
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `charges` - A vector of charges.
    pub fn compute_signal(&self, order: usize, charges: &[T]) -> Array3D<T> {
        let n = 2 * order - 1;
        let n_tot = n * n * n;
        let mut result = rlst_dynamic_array3!(T, [n, n, n]);

        let mut tmp = vec![T::zero(); n_tot];

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let conv_index = i + n * j + n * n * k;
                    if let Some(surf_index) = self.conv_to_surf_map.get(&conv_index) {
                        tmp[conv_index] = charges[*surf_index];
                    } else {
                        tmp[conv_index] = T::zero();
                    }
                }
            }
        }

        result.data_mut().copy_from_slice(&tmp[..]);

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
    use rlst_dense::traits::RandomAccessMut;

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
        assert_eq!(m2l.st_block.shape(), [k, svd.ncoeffs(order)]);
        assert_eq!(m2l.c.shape(), [k, k * ntransfer_vectors]);
        assert_eq!(m2l.u.shape(), [svd.ncoeffs(order), k]);

        // Test that the rank cutoff has been taken correctly (k > ncoeffs)
        let k = 100;
        let svd = SvdFieldTranslationKiFmm::new(kernel.clone(), Some(k), order, domain, alpha);
        let m2l = svd.compute_m2l_operators(order, domain);
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
        let m2l = svd.compute_m2l_operators(order, domain);
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
        let m2l: FftM2lOperatorData<c32> = fft.compute_m2l_operators(order, domain);
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
        let mut multipole = rlst_dynamic_array2![f64, [ncoeffs, 1]];

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

        let nrows = svd.operator_data.c.shape()[0];
        let top_left = [0, c_idx * svd.k];
        let dim = [nrows, svd.k];

        let c_sub = svd.operator_data.c.into_subview(top_left, dim);

        let compressed_multipole = empty_array::<f64, 2>()
            .simple_mult_into_resize(svd.operator_data.st_block.view(), multipole.view()); //svd.operator_data.st_block.dot(&multipole).eval();

        let compressed_check_potential = empty_array::<f64, 2>()
            .simple_mult_into_resize(c_sub.view(), compressed_multipole.view()); //c_sub.dot(&compressed_multipole);

        // Post process to find check potential
        let check_potential = empty_array::<f64, 2>().simple_mult_into_resize(
            svd.operator_data.u.view(),
            compressed_check_potential.view(),
        ); // svd.operator_data.u.dot(&compressed_check_potential).eval();

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
        let mut multipole = rlst_dynamic_array2![f64, [ncoeffs, 1]];

        for i in 0..ncoeffs {
            *multipole.get_mut([i, 0]).unwrap() = i as f64;
        }

        let level = 2;
        // Create field translation object
        let fft = FftFieldTranslationKiFmm::new(kernel, order, domain, alpha);

        let kernels = &fft.operator_data.kernel_data;

        let key = MortonKey::from_point(&[0.5, 0.5, 0.5], &domain, level);

        let parent_neighbours = key.parent().neighbors();
        let mut v_list_structured = vec![Vec::new(); 26];
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
        let p = m + 1;
        let q = n + 1;
        let r = o + 1;

        let padded_kernel = pad3(&test_kernel, (p - m, q - n, r - o), (0, 0, 0));
        let mut padded_kernel = flip3(&padded_kernel);

        // Compute FFT of padded kernel
        let mut padded_kernel_hat = rlst_dynamic_array3!(c64, [p, q, r / 2 + 1]);
        f64::rfft3_fftw(
            padded_kernel.data_mut(),
            padded_kernel_hat.data_mut(),
            &[p, q, r],
        );

        for (p, t) in padded_kernel_hat.data().iter().zip(kernel_hat.iter()) {
            assert!((p - t).norm() < 1e-6)
        }
    }

    #[test]
    fn test_kernel_rearrangement() {
        // Dummy data mirroring unrearranged kernels
        // here each '1000' corresponds to a sibling index
        // each '100' to a child in a given halo element
        // and each '1' to a frequency
        let mut kernel_data_mat = vec![Vec::new(); 26];
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
        let mut rearranged = vec![Vec::new(); 26];
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
        let mut multipole = rlst_dynamic_array2![f64, [ncoeffs, 1]];

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
        let signal = fft.compute_signal(order, multipole.data());
        let [m, n, o] = signal.shape();
        let p = m + 1;
        let q = n + 1;
        let r = o + 1;
        let pad_size = (p - m, q - n, r - o);
        let pad_index = (p - m, q - n, r - o);
        let mut padded_signal = pad3(&signal, pad_size, pad_index);
        let mut padded_signal_hat = rlst_dynamic_array3!(c64, [p, q, r / 2 + 1]);

        f64::rfft3_fftw(
            padded_signal.data_mut(),
            padded_signal_hat.data_mut(),
            &[p, q, r],
        );

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
        let p = m + 1;
        let q = n + 1;
        let r = o + 1;

        let padded_kernel = pad3(&kernel, (p - m, q - n, r - o), (0, 0, 0));
        let mut padded_kernel = flip3(&padded_kernel);

        // Compute FFT of padded kernel
        let mut padded_kernel_hat = rlst_dynamic_array3!(c64, [p, q, r / 2 + 1]);
        f64::rfft3_fftw(
            padded_kernel.data_mut(),
            padded_kernel_hat.data_mut(),
            &[p, q, r],
        );

        // Compute convolution
        let mut hadamard_product = padded_signal_hat
            .data()
            .iter()
            .zip(padded_kernel_hat.data().iter())
            .map(|(a, b)| a * b)
            .collect_vec();

        // let mut hadamard_product = Array::new(BaseArray::new(SliceContainer::new(&hadamard_product), [p, q, r / 2 + 1]));

        let mut potentials = rlst_dynamic_array3!(f64, [p, q, r]);

        f64::irfft3_fftw(
            &mut hadamard_product,
            potentials.data_mut(),
            &[p, q, r],
        );

        let (_, multi_indices) = MortonKey::surface_grid::<f64>(order);

        let mut tmp = Vec::new();
        let ntargets = multi_indices.len() / 3;
        let xs = &multi_indices[0..ntargets];
        let ys = &multi_indices[ntargets..2 * ntargets];
        let zs = &multi_indices[2 * ntargets..];

        for i in 0..ntargets {
            let val = potentials.get([zs[i], ys[i], xs[i]]).unwrap();
            tmp.push(*val);
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

        let abs_error: f64 = tmp
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

        assert!(rel_error < 1e-15);
    }
}
