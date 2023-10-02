use std::{
    collections::{HashMap, HashSet},
    usize,
};

use fftw::{plan::*, types::*};
use itertools::Itertools;
use num::traits::Num;
use rayon::prelude::*;

use bempp_tools::Array3D;
use bempp_traits::arrays::Array3DAccess;
use bempp_tree::types::{domain::Domain, morton::MortonKey};

use rlst::dense::{
    base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix, traits::*, Dynamic,
};

use crate::types::TransferVector;

pub type FftMatrixf64 =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

pub type FftMatrixc64 =
    Matrix<c64, BaseMatrix<c64, VectorContainer<c64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

/// Useful helper functions for handling arrays relevant to field translations.
pub mod array {
    use super::*;

    /// Return indices that sort a vec.
    ///
    /// # Arguments
    /// * `arr` - An array to be sorted.
    pub fn argsort<T: Ord>(arr: &[T]) -> Vec<usize> {
        arr.iter()
            .enumerate()
            .sorted_by(|a, b| a.1.cmp(b.1))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Pad an Array3D from a given `pad_index` with an amount of zeros specified by `pad_size` to the right of each axis.
    ///
    /// # Arguments
    /// * `arr` - An array to be padded.
    /// * `pad_size` - The amount of padding to be added along each axis.
    /// * `pad_index` - The position in the array to start the padding from.
    pub fn pad3<T>(
        arr: &Array3D<T>,
        pad_size: (usize, usize, usize),
        pad_index: (usize, usize, usize),
    ) -> Array3D<T>
    where
        T: Clone + Copy + Num,
    {
        let &(m, n, o) = arr.shape();

        let (x, y, z) = pad_index;
        let (p, q, r) = pad_size;

        // Check that there is enough space for pad
        assert!(x + p <= m + p && y + q <= n + q && z + r <= o + r);

        let mut padded = Array3D::new((p + m, q + n, r + o));

        for i in 0..m {
            for j in 0..n {
                for k in 0..o {
                    *padded.get_mut(x + i, y + j, z + k).unwrap() = *arr.get(i, j, k).unwrap();
                }
            }
        }

        padded
    }

    /// Flip an Array3D along each axis, returns a new array.
    ///
    /// # Arguments
    /// * `arr` - An array to be flipped.
    pub fn flip3<T>(arr: &Array3D<T>) -> Array3D<T>
    where
        T: Clone + Copy + Num,
    {
        let mut flipped = Array3D::new(*arr.shape());

        let &(m, n, o) = arr.shape();

        for i in 0..m {
            for j in 0..n {
                for k in 0..o {
                    *flipped.get_mut(i, j, k).unwrap() =
                        *arr.get(m - i - 1, n - j - 1, o - k - 1).unwrap();
                }
            }
        }

        flipped
    }
}

/// FFTW Wrappers
pub mod fft {

    use super::*;

    /// Compute a Real FFT of an input slice corresponding to a 3D array stored in column major format, specified by `shape` using the FFTW library.
    ///
    /// # Arguments
    /// * `input` - Input slice of real data, corresponding to a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of input data.
    pub fn rfft3_fftw(input: &mut [f64], output: &mut [c64], shape: &[usize]) {
        assert!(shape.len() == 3);
        let plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();
        let _ = plan.r2c(input, output);
    }

    /// Compute an inverse Real FFT of an input slice corresponding to the FFT of a 3D array stored in column major format, specified by `shape` using the FFTW library.
    /// This function normalises the output.
    ///
    /// # Arguments
    /// * `input` - Input slice of complex data, corresponding to an FFT of a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of input data.
    pub fn irfft3_fftw(input: &mut [c64], output: &mut [f64], shape: &[usize]) {
        assert!(shape.len() == 3);
        let size: usize = shape.iter().product();
        let plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();
        let _ = plan.c2r(input, output);
        // Normalise
        output
            .iter_mut()
            .for_each(|value| *value *= 1.0 / (size as f64));
    }

    /// Compute a Real FFT over a rlst matrix which stores data corresponding to multiple 3 dimensional arrays of shape `shape`, stored in column major order.
    /// This function is multithreaded, and uses the FFTW library.
    ///
    /// # Arguments
    /// * `input` - Input slice of real data, corresponding to a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of input data.
    pub fn rfft3_fftw_par_vec(
        input: &mut FftMatrixf64,
        output: &mut FftMatrixc64,
        shape: &[usize],
    ) {
        assert!(shape.len() == 3);

        let size: usize = shape.iter().product();
        let size_d = shape.last().unwrap();
        let size_real = (size / size_d) * (size_d / 2 + 1);

        let plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();
        let it_inp = input.data_mut().par_chunks_exact_mut(size).into_par_iter();
        let it_out = output
            .data_mut()
            .par_chunks_exact_mut(size_real)
            .into_par_iter();

        it_inp.zip(it_out).for_each(|(inp, out)| {
            let _ = plan.r2c(inp, out);
        });
    }

    /// Compute an inverse Real FFT over a rlst matrix which stores data corresponding to multiple 3 dimensional arrays of shape `shape`, stored in column major order.
    /// This function is multithreaded, and uses the FFTW library.
    ///
    /// # Arguments
    /// * `input` - Input slice of complex data, corresponding to an FFT of a 3D array stored in column major order.
    /// * `output` - Output slice.
    /// * `shape` - Shape of input data.
    pub fn irfft3_fftw_par_vec(
        input: &mut FftMatrixc64,
        output: &mut FftMatrixf64,
        shape: &[usize],
    ) {
        assert!(shape.len() == 3);
        let size: usize = shape.iter().product();
        let size_d = shape.last().unwrap();
        let size_real = (size / size_d) * (size_d / 2 + 1);
        let plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();

        let it_inp = input
            .data_mut()
            .par_chunks_exact_mut(size_real)
            .into_par_iter();
        let it_out = output.data_mut().par_chunks_exact_mut(size).into_par_iter();

        it_inp.zip(it_out).for_each(|(inp, out)| {
            let _ = plan.c2r(inp, out);
            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (size as f64));
        })
    }
}

/// Useful helper functions related to handling surfaces created for KiFMMs during field translations.
pub mod surface {
    use super::*;

    /// For a multi-index corresponding to a position on the surface grids created during the KiFMM, reflect its associated multi-index such that it
    /// matches the new position positioning of its associated transfer vector now placed in the reference octant.  
    ///
    /// # Arguments
    /// * `multi_index` - A three vector describing a surface grid point.
    /// * `transfer_vector` - A three vector describing an associated transfer vector with this surface.
    /// * `order` - The expansion order relating this surface grid created for the KiFMM.
    pub fn axial_reflection_surface(
        multi_index: &[usize],
        transfer_vector: &[i64],
        order: usize,
    ) -> Vec<usize> {
        fn helper(m: usize, t: i64, order: usize) -> usize {
            if t >= 0 {
                return m;
            } else {
                return order - (m - 1);
            }
        }

        let res = multi_index
            .iter()
            .enumerate()
            .map(|(i, &m)| helper(m, transfer_vector[i], order))
            .collect_vec();

        res
    }

    /// For a multi-index corresponding to a position on the convolution grid created for the sparsificaiton of the M2L operator in the KIFMM via an FFT,
    /// reflect its multi-index such that it matches the new position positioning of its associated transfer vector in the reference octant.  
    ///
    /// # Arguments
    /// * `multi_index` - A three vector describing a convolution grid point.
    /// * `transfer_vector` - A three vector describing an associated transfer vector with this surface.
    /// * `order` - The expansion order relating this convolution grid created for the sparsification of the KiFMM via an FFT.
    pub fn axial_reflection_convolution(
        multi_index: &[usize],
        transfer_vector: &[i64],
        order: usize,
    ) -> Vec<usize> {
        fn helper(m: usize, t: i64, order: usize) -> usize {
            if t >= 0 {
                return m;
            } else {
                return 2 * (order - 1) - (m - 1);
            }
        }

        let res = multi_index
            .iter()
            .enumerate()
            .map(|(i, &m)| helper(m, transfer_vector[i], order))
            .collect_vec();

        res
    }

    /// Perform a diagonal reflection to get a surface multi-index into reference cone. At this point the transfer vectors are assumed to correspond
    /// to translations already in the reference cone.
    ///
    /// # Arguments
    /// * `multi_index` - A three vector describing a convolution grid point.
    /// * `transfer_vector` - A three vector describing an associated transfer vector with this surface.
    pub fn diagonal_reflection(multi_index: &[usize], transfer_vector: &[i64]) -> Vec<usize> {
        let idxs = crate::helpers::array::argsort(transfer_vector);

        let res = idxs.iter().map(|&i| multi_index[i]).collect_vec();

        res
    }
}

/// Useful helper functions for handling transfer vectors.
pub mod transfer_vector {

    use super::*;

    /// Reflect a transfer vector into the reference octant combining axial and diagonal symmetries.
    pub fn reflect(components: &[i64; 3]) -> Vec<i64> {
        // Axial reflection
        let axial = axially_reflect_components(components);

        // Diagonal reflection
        let axial_diag = diagonally_reflect_components(&axial[..]);

        axial_diag
    }

    /// Apply axial reflections to Transfer Vector components to get into reference octant.
    pub fn axially_reflect_components(components: &[i64]) -> Vec<i64> {
        // Axial reflection
        let axial = |c: &i64| {
            if *c >= 0 {
                *c
            } else {
                -*c
            }
        };

        let axial = components.iter().map(axial).collect_vec();
        axial
    }

    /// Apply diagonal reference to Transfer Vector components to get into reference cone.
    /// The vector must already be in the reference octant.
    pub fn diagonally_reflect_components(components: &[i64]) -> Vec<i64> {
        // Diagonal reflection
        let idxs = crate::helpers::array::argsort(components);

        let axial_diag = idxs.iter().map(|&i| components[i].clone()).collect_vec();

        axial_diag
    }
}

/// Unique M2L interactions for homogenous, translationally invariant kernel functions (e.g. Laplace/Helmholtz).
/// There are at most 316 such interactions, corresponding to unique `transfer vectors'. Here we compute all of them
/// with respect to level 3 of an associated octree (this is the first level in which they all exist).
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
        let components = t.find_transfer_vector_components(&s);

        result.push(TransferVector {
            components,
            hash: v,
            source: s,
            target: t,
        })
    }

    result
}

/// Though there are at most 316 unique transfer vectors for homogenous, translationally invariant, kernels
/// many of these are actually redundant rotations of a subset of just 16 unique transfer vectors. This function
/// returns a Vec of these unique transfer vectors, as well as a map between the redundant and unique checksums.
pub fn compute_transfer_vectors_unique() -> (Vec<TransferVector>, HashMap<usize, usize>) {
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

    let mut transfer_vectors_component: Vec<[i64; 3]> = Vec::new();

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
        let tmp: Vec<[i64; 3]> = v_list
            .iter()
            .map(|v| key.find_transfer_vector_components(v))
            .collect_vec();

        transfer_vectors_component.extend(&mut tmp.iter().cloned());

        sources.extend(&mut v_list.iter().cloned());

        let tmp_targets = vec![**key; tmp.len()];
        targets.extend(&mut tmp_targets.iter().cloned());
    }

    // Find unique transfer vectors (316 in total hom. smooth kernels)
    let mut unique_transfer_vectors = Vec::new();
    let mut unique_transfer_vectors_component = Vec::new();
    let mut unique_indices = HashSet::new();

    let transfer_vectors = transfer_vectors_component
        .iter()
        .map(|c| MortonKey::find_transfer_vector_from_components(c))
        .collect_vec();

    for (i, (vec, comp)) in transfer_vectors
        .iter()
        .zip(transfer_vectors_component)
        .enumerate()
    {
        if !unique_transfer_vectors.contains(vec) {
            unique_transfer_vectors.push(*vec);
            unique_transfer_vectors_component.push(comp);
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

    // Now remove axial/diagonal redundancies as well resulting in 16 transfer vectors in total
    let mut reflected_transfer_vectors = Vec::new();

    // Create a map between the original transfer vectors and the reduced set.
    let mut axial_diag_map = HashMap::new();
    let mut axial_diag_map_component = HashMap::new();
    let mut unique_reflected_transfer_vectors = HashSet::new();

    for t in unique_transfer_vectors_component.iter() {
        let t_refl = crate::helpers::transfer_vector::reflect(t);
        // Get into checksum for ease of lookup
        let t_checksum = MortonKey::find_transfer_vector_from_components(t);
        let t_rot_checksum = MortonKey::find_transfer_vector_from_components(&t_refl[..]);

        axial_diag_map.insert(t_checksum, t_rot_checksum.clone());
        axial_diag_map_component.insert(t, t_refl.clone());
        unique_reflected_transfer_vectors.insert(t_rot_checksum.clone());

        reflected_transfer_vectors.push(t_refl);
    }

    // For each unique transfer vector find representative source/target pair for calculating
    // FMM matrices

    let mut result = Vec::<TransferVector>::new();

    let mut checked = HashSet::new();

    for (((&t, &s), &h), &c) in unique_targets
        .iter()
        .zip(unique_sources.iter())
        .zip(unique_transfer_vectors.iter())
        .zip(unique_transfer_vectors_component.iter())
    {
        let h_refl = axial_diag_map.get(&h).unwrap();
        let c_refl = axial_diag_map_component.get(&c).unwrap();

        if !checked.contains(h_refl) {
            result.push(TransferVector {
                components: [c_refl[0], c_refl[1], c_refl[2]],
                hash: *h_refl,
                target: t,
                source: s,
            });
            checked.insert(h_refl);
        }
    }

    // Also need to store mappings between surface/conv grid multi-indices
    // for the reduced set of sources and targets I've taken that correspond
    // to unique transfer vectors.

    (result, axial_diag_map)
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_argsort() {
        assert!(true)
    }

    #[test]
    fn test_flip3() {
        assert!(true)
    }

    #[test]
    fn test_pad3() {
        let dim = 3;
        // Initialise input data
        let mut input = Array3D::new((dim, dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    *input.get_mut(i, j, k).unwrap() = (i + j * dim + k * dim * dim + 1) as f64
                }
            }
        }

        // Test padding at edge of each axis
        let pad_size = (2, 3, 4);
        let pad_index = (0, 0, 0);
        let padded = crate::helpers::array::pad3(&input, pad_size, pad_index);

        let &(m, n, o) = padded.shape();

        // Check dimension
        assert_eq!(m, dim + pad_size.0);
        assert_eq!(n, dim + pad_size.1);
        assert_eq!(o, dim + pad_size.2);

        // Check that padding has been correctly applied
        for i in dim..m {
            for j in dim..n {
                for k in dim..o {
                    assert_eq!(*padded.get(i, j, k).unwrap(), 0f64)
                }
            }
        }

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    assert_eq!(*padded.get(i, j, k).unwrap(), *input.get(i, j, k).unwrap())
                }
            }
        }

        // Test padding at the start of each axis
        let pad_index = (2, 2, 2);

        let padded = crate::helpers::array::pad3(&input, pad_size, pad_index);

        // Check that padding has been correctly applied
        for i in 0..pad_index.0 {
            for j in 0..pad_index.1 {
                for k in 0..pad_index.2 {
                    assert_eq!(*padded.get(i, j, k).unwrap(), 0f64)
                }
            }
        }

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    assert_eq!(
                        *padded
                            .get(i + pad_index.0, j + pad_index.1, k + pad_index.2)
                            .unwrap(),
                        *input.get(i, j, k).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn test_axial_reflection_surface() {
        assert!(true)
    }

    #[test]
    fn test_axial_reflection_convolution() {
        assert!(true)
    }

    #[test]
    fn test_diagonal_reflection() {
        assert!(true)
    }

    #[test]
    fn test_diagonally_reflect_components() {
        assert!(true)
    }

    #[test]
    fn test_axially_reflect_components() {
        assert!(true)
    }

    #[test]
    fn test_compute_transfer_vectors() {
        assert!(true)
    }

    #[test]
    fn test_compute_transfer_vectors_unique() {
        assert!(true)
    }
}
