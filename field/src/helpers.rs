//! Helper modules for field translations
use std::{
    collections::{HashMap, HashSet},
    usize, 
};

use itertools::Itertools;

use bempp_tree::types::{domain::Domain, morton::MortonKey};

use crate::{
    types::TransferVector,
    array::argsort,
    transfer_vector::reflect,
};

#[cfg(test)]
mod test {

    use super::{*, array::*, surface::*, transfer_vector::*};

    #[test]
    fn test_compute_transfer_vectors() {
        
        let (tv, map) = compute_transfer_vectors();

        // Check that there are 316 unique transfer vectors
        assert_eq!(tv.len(), 316);
        
        // Check that the map has 316 unique keys, that map to 316 unique values
        let keys = map.keys().collect_vec();
        let values: HashSet<usize> = map.values().cloned().collect();
        assert_eq!(keys.len(), 316);
        assert_eq!(values.len(), 316);

    }

    #[test]
    fn test_compute_transfer_vectors_unique() {
        let (tv, map) = compute_transfer_vectors_unique();

        assert_eq!(tv.len(), 16);

        // Check that the map has 316 unique keys, that map to 16 unique values
        let keys = map.keys().collect_vec();
        let values: HashSet<usize> = map.values().cloned().collect();
        assert_eq!(keys.len(), 316);
        assert_eq!(values.len(), 16);

    }
}
