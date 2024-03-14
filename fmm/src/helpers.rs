//! Helpers
use std::collections::HashMap;

use bempp_traits::tree::Tree;
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};
use num::Float;
use rlst_dense::{
    rlst_dynamic_array2,
    traits::{RawAccess, RawAccessMut, Shape},
    types::RlstScalar,
};

use crate::types::{Charges, SendPtrMut};

/// Euclidean algorithm to find greatest divisor of `n` less than or equal to `max`
///
/// # Arguments
/// * `max` - The maximum chunk size
pub fn chunk_size(n: usize, max: usize) -> usize {
    let max_divisor = max;
    for divisor in (1..=max_divisor).rev() {
        if n % divisor == 0 {
            return divisor;
        }
    }
    1 // If no divisor is found greater than 1, return 1 as the GCD
}

/// Scaling to apply to homogenous scale invariant kernels at a given octree level.
///
/// # Arguments
/// * `level` - The octree level
pub fn homogenous_kernel_scale<T: RlstScalar<Real = T>>(level: u64) -> T {
    let numerator = T::from(1).unwrap();
    let denominator = T::from(2.).unwrap();
    let power = T::from(level).unwrap();
    let denominator = <T as RlstScalar>::powf(denominator, power);
    numerator / denominator
}

/// Scaling to apply to M2L operators calculated using homogenous scale invariant kernels at a given octree level.
///
/// # Arguments
/// * `level` - The octree level
pub fn m2l_scale<T: RlstScalar<Real = T>>(level: u64) -> T {
    if level < 2 {
        panic!("M2L only perfomed on level 2 and below")
    }

    if level == 2 {
        T::from(1. / 2.).unwrap()
    } else {
        let two = T::from(2.0).unwrap();
        <T as RlstScalar>::powf(two, T::from(level - 3).unwrap())
    }
}

/// Compute the scaling for each leaf box in a tree
///
/// # Arguments
/// * `tree`- Single node tree
/// * `ncoeffs`- Number of interpolation points on leaf box
pub fn leaf_scales<T>(tree: &SingleNodeTree<T>, ncoeffs: usize) -> Vec<T>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    let mut result = vec![T::default(); tree.nleaves().unwrap() * ncoeffs];

    for (i, leaf) in tree.all_leaves().unwrap().iter().enumerate() {
        // Assign scales
        let l = i * ncoeffs;
        let r = l + ncoeffs;
        result[l..r]
            .copy_from_slice(vec![homogenous_kernel_scale(leaf.level()); ncoeffs].as_slice());
    }
    result
}

/// Compute the surfaces for each leaf box
///
/// # Arguments
/// * `tree`- Single node tree
/// * `ncoeffs`- Number of interpolation points on leaf box
/// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
/// * `expansion_order` - Expansion order of the FMM
pub fn leaf_surfaces<T>(
    tree: &SingleNodeTree<T>,
    ncoeffs: usize,
    alpha: T,
    expansion_order: usize,
) -> Vec<T>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    let dim = 3;
    let nkeys = tree.nleaves().unwrap();
    let mut result = vec![T::default(); ncoeffs * dim * nkeys];

    for (i, key) in tree.all_leaves().unwrap().iter().enumerate() {
        let l = i * ncoeffs * dim;
        let r = l + ncoeffs * dim;
        let surface = key.compute_kifmm_surface(tree.domain(), expansion_order, alpha);

        result[l..r].copy_from_slice(&surface);
    }

    result
}

/// Create an index pointer for the coordinates in a source and a target tree
/// between the local indices for each leaf and their associated charges
pub fn coordinate_index_pointer<T>(tree: &SingleNodeTree<T>) -> Vec<(usize, usize)>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    let mut index_pointer = 0;

    let mut result = vec![(0usize, 0usize); tree.nleaves().unwrap()];

    for (i, leaf) in tree.all_leaves().unwrap().iter().enumerate() {
        let npoints = if let Some(n) = tree.ncoordinates(leaf) {
            n
        } else {
            0
        };

        // Update charge index pointer
        result[i] = (index_pointer, index_pointer + npoints);
        index_pointer += npoints;
    }

    result
}

/// Create index pointers for each key at each level of an octree
pub fn level_index_pointer<T>(tree: &SingleNodeTree<T>) -> Vec<HashMap<MortonKey, usize>>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    let mut result = vec![HashMap::new(); (tree.depth() + 1).try_into().unwrap()];

    for level in 0..=tree.depth() {
        let keys = tree.keys(level).unwrap();
        for (level_idx, key) in keys.iter().enumerate() {
            result[level as usize].insert(*key, level_idx);
        }
    }
    result
}

/// Create mutable pointers corresponding to each multipole expansion at each level of an octree
pub fn level_expansion_pointers<T>(
    tree: &SingleNodeTree<T>,
    ncoeffs: usize,
    nmatvecs: usize,
    expansions: &[T],
) -> Vec<Vec<Vec<SendPtrMut<T>>>>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    let mut result = vec![Vec::new(); (tree.depth() + 1).try_into().unwrap()];

    for level in 0..=tree.depth() {
        let mut tmp_multipoles = Vec::new();

        let keys = tree.keys(level).unwrap();
        for key in keys.iter() {
            let &key_idx = tree.index(key).unwrap();
            let key_displacement = ncoeffs * nmatvecs * key_idx;
            let mut key_multipoles = Vec::new();
            for eval_idx in 0..nmatvecs {
                let eval_displacement = ncoeffs * eval_idx;
                let raw = unsafe {
                    expansions
                        .as_ptr()
                        .add(key_displacement + eval_displacement) as *mut T
                };
                key_multipoles.push(SendPtrMut { raw });
            }
            tmp_multipoles.push(key_multipoles)
        }
        result[level as usize] = tmp_multipoles
    }

    result
}

/// Create mutable pointers for leaf expansions in a tree
pub fn leaf_expansion_pointers<T>(
    tree: &SingleNodeTree<T>,
    ncoeffs: usize,
    nmatvecs: usize,
    nleaves: usize,
    expansions: &[T],
) -> Vec<Vec<SendPtrMut<T>>>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    let mut result = vec![Vec::new(); nleaves];

    for (leaf_idx, leaf) in tree.all_leaves().unwrap().iter().enumerate() {
        let key_idx = tree.index(leaf).unwrap();
        let key_displacement = ncoeffs * nmatvecs * key_idx;
        for eval_idx in 0..nmatvecs {
            let eval_displacement = ncoeffs * eval_idx;
            let raw = unsafe {
                expansions
                    .as_ptr()
                    .add(eval_displacement + key_displacement) as *mut T
            };

            result[leaf_idx].push(SendPtrMut { raw });
        }
    }

    result
}

/// Create mutable pointers for potentials in a tree
pub fn potential_pointers<T>(
    tree: &SingleNodeTree<T>,
    nmatvecs: usize,
    nleaves: usize,
    npoints: usize,
    kernel_eval_size: usize,
    potentials: &[T],
) -> Vec<SendPtrMut<T>>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    let mut result = vec![SendPtrMut::default(); nleaves * nmatvecs];
    let dim = 3;

    let mut raw_pointers = Vec::new();
    for eval_idx in 0..nmatvecs {
        let ptr = unsafe {
            potentials
                .as_ptr()
                .add(eval_idx * npoints * kernel_eval_size) as *mut T
        };
        raw_pointers.push(ptr)
    }

    for (i, leaf) in tree.all_leaves().unwrap().iter().enumerate() {
        let npoints;
        let nevals;

        if let Some(coordinates) = tree.coordinates(leaf) {
            npoints = coordinates.len() / dim;
            nevals = npoints * kernel_eval_size;
        } else {
            nevals = 0;
        }

        for j in 0..nmatvecs {
            result[nleaves * j + i] = SendPtrMut {
                raw: raw_pointers[j],
            }
        }

        // Update raw pointer with number of points at this leaf
        for ptr in raw_pointers.iter_mut() {
            *ptr = unsafe { ptr.add(nevals) }
        }
    }

    result
}

/// Map charges to map global indices
pub fn map_charges<T: RlstScalar>(global_indices: &[usize], charges: &Charges<T>) -> Charges<T> {
    let [ncharges, nmatvecs] = charges.shape();

    let mut reordered_charges = rlst_dynamic_array2!(T, [ncharges, nmatvecs]);

    for eval_idx in 0..nmatvecs {
        let eval_displacement = eval_idx * ncharges;
        for (new_idx, old_idx) in global_indices.iter().enumerate() {
            reordered_charges.data_mut()[new_idx + eval_displacement] =
                charges.data()[old_idx + eval_displacement];
        }
    }

    reordered_charges
}
