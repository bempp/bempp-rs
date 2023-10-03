//! Tools for handling and creating transfer vectors for homogenous, translationally invariant kernels.
use itertools::Itertools;
use std::collections::{HashMap, HashSet};

use bempp_tree::types::{domain::Domain, morton::MortonKey};

use crate::{array::argsort, types::TransferVector};

/// Reflect a transfer vector into the reference octant combining axial and diagonal symmetries.
///
/// # Arguments
/// * `components` - The transfer vector in component form.
pub fn reflect(components: &[i64; 3]) -> Vec<i64> {
    // Only three vectors valid for transfer vector
    assert!(components.len() == 3);

    // Axial reflection
    let axial = axially_reflect_components(components);

    // Diagonal reflection
    let axial_diag = diagonally_reflect_components(&axial[..]);

    axial_diag
}

/// Apply axial reflections to Transfer Vector components to get into reference octant.
///
/// # Arguments
/// * `components` - The transfer vector in component form.
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
///
/// # Arguments
/// * `components` - The transfer vector in component form.
pub fn diagonally_reflect_components(components: &[i64]) -> Vec<i64> {
    // Only three vectors valid for transfer vector
    assert!(components.len() == 3);

    // Diagonal reflection
    let idxs = argsort(components);

    let axial_diag = idxs.iter().map(|&i| components[i].clone()).collect_vec();

    axial_diag
}

/// Unique M2L interactions for homogenous, translationally invariant kernel functions (e.g. Laplace/Helmholtz).
/// There are at most 316 such interactions, corresponding to unique `transfer vectors'. Here we compute all of them
/// with respect to level 3 of an associated octree (this is the first level in which they all exist). The returned map
/// is somewhat redundant, as it simply maps each unique transfer vector to itself.
pub fn compute_transfer_vectors() -> (Vec<TransferVector>, HashMap<usize, usize>) {
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

    // Filter for unique transfer vectors, and their corresponding index
    let mut unique_transfer_vectors = Vec::new();
    let mut unique_indices = HashSet::new();

    for (i, vec) in transfer_vectors.iter().enumerate() {
        if !unique_transfer_vectors.contains(vec) {
            unique_transfer_vectors.push(*vec);
            unique_indices.insert(i);
        }
    }

    // Identify sources/targets which correspond to unique transfer vectors.
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
    let mut map = HashMap::new();

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
        });

        map.insert(v, v);
    }

    (result, map)
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
        let t_refl = reflect(t);
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

    (result, axial_diag_map)
}

#[cfg(test)]
pub mod test {

    use super::*;

    #[test]
    fn test_diagonally_reflect_components() {
        // Test with mixed values
        let components1 = [3i64, 1i64, 2i64];
        let reflected1 = diagonally_reflect_components(&components1);
        assert_eq!(reflected1, vec![1i64, 2i64, 3i64]);

        // Test with all the same values (repeated numbers)
        let components2 = [2i64, 2i64, 2i64];
        let reflected2 = diagonally_reflect_components(&components2);
        assert_eq!(reflected2, vec![2i64, 2i64, 2i64]);

        // Test negative values since the vector is assumed to be in the reference octant.
        let components3 = [-1i64, -2i64, -3i64];
        let reflected3 = diagonally_reflect_components(&components3);
        assert_eq!(reflected3, vec![-3i64, -2i64, -1i64]);
    }

    #[test]
    fn test_axially_reflect_components() {
        // Mixed components in transfer vector.
        let components = [-1i64, 2i64, -3i64];
        let reflected = axially_reflect_components(&components);
        assert_eq!(reflected, vec![1i64, 2i64, 3i64]);

        // Positive components in transfer vector.
        let components = [1i64, 2i64, 3i64];
        let reflected = axially_reflect_components(&components);
        assert_eq!(reflected, vec![1i64, 2i64, 3i64]);

        // Negative components in transfer vector
        let components = [-1i64, -2i64, -3i64];
        let reflected = axially_reflect_components(&components);
        assert_eq!(reflected, vec![1i64, 2i64, 3i64]);
    }
}
