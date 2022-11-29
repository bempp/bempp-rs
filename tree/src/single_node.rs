//! Data Structures and methods to create Octrees on a single node.

use std::{
    collections::HashSet,
    ops::{Deref, DerefMut},
};

use itertools::Itertools;

use crate::{
    constants::DEEPEST_LEVEL,
    types::morton::{MortonKey, MortonKeys},
};

/// Interface for a local (non-distributed) Tree.
#[derive(Debug)]
pub struct SingleNodeTree {
    /// The nodes that span the tree, defined by its leaf nodes, not necessarily complete.
    pub keys: MortonKeys,
}

impl SingleNodeTree {
    /// Linearize (remove overlaps) a vector of keys. The input must be sorted. Algorithm 7 in [1].
    pub fn linearize_keys(keys: MortonKeys) -> MortonKeys {
        let nkeys = keys.len();

        // Then we remove the ancestors.
        let mut new_keys = Vec::<MortonKey>::with_capacity(keys.len());

        // Now check pairwise for ancestor relationship and only add to new vector if item
        // is not an ancestor of the next item. Add final element.
        keys.into_iter()
            .enumerate()
            .tuple_windows::<((_, _), (_, _))>()
            .for_each(|((_, a), (j, b))| {
                if !a.is_ancestor(&b) {
                    new_keys.push(a);
                }
                if j == (nkeys - 1) {
                    new_keys.push(b);
                }
            });

        new_keys
    }

    /// Complete the region between two keys with the minimum spanning nodes, algorithm 6 in [1].
    pub fn complete_region(a: &MortonKey, b: &MortonKey) -> MortonKeys {
        let mut a_ancestors: HashSet<MortonKey> = a.ancestors();
        let mut b_ancestors: HashSet<MortonKey> = b.ancestors();

        a_ancestors.remove(a);
        b_ancestors.remove(b);

        let mut minimal_tree: MortonKeys = Vec::new();
        let mut work_list: MortonKeys = a.finest_ancestor(b).children().into_iter().collect();

        while !work_list.is_empty() {
            let current_item = work_list.pop().unwrap();
            if (current_item > *a) & (current_item < *b) & !b_ancestors.contains(&current_item) {
                minimal_tree.push(current_item);
            } else if (a_ancestors.contains(&current_item)) | (b_ancestors.contains(&current_item))
            {
                let mut children = current_item.children();
                work_list.append(&mut children);
            }
        }

        minimal_tree.sort();
        minimal_tree
    }

    /// Complete the region between all elements in an tree that doesn't necessarily span
    /// the domain defined by its least and greatest nodes.
    pub fn complete(self: &mut SingleNodeTree) {
        let a = self.keys.iter().min().unwrap();
        let b = self.keys.iter().max().unwrap();
        let mut completion = Tree::complete_region(a, b);
        completion.push(*a);
        completion.push(*b);
        completion.sort();
        self.keys = completion;
    }

    /// Wrapper for linearize_keys over all keys in Tree.
    pub fn linearize(self: &mut SingleNodeTree) {
        self.keys.sort();
        self.keys = Tree::linearize_keys(self.keys.clone());
    }

    /// Wrapper for sorting a tree, by its keys.
    pub fn sort(self: &mut SingleNodeTree) {
        self.keys.sort();
    }

    /// Enforce a 2:1 balance for a tree, and remove any overlaps.
    pub fn balance(&self) -> SingleNodeTree {
        let mut balanced: HashSet<MortonKey> = self.keys.iter().cloned().collect();

        for level in (0..DEEPEST_LEVEL).rev() {
            let work_list: MortonKeys = balanced
                .iter()
                .filter(|key| key.level() == level)
                .cloned()
                .collect();

            for key in work_list {
                let neighbors = key.neighbors();

                for neighbor in neighbors {
                    let parent = neighbor.parent();
                    if !balanced.contains(&neighbor) && !balanced.contains(&neighbor) {
                        balanced.insert(parent);

                        if parent.level() > 0 {
                            for sibling in parent.siblings() {
                                balanced.insert(sibling);
                            }
                        }
                    }
                }
            }
        }

        let mut balanced: MortonKeys = balanced.into_iter().collect();
        balanced.sort();
        let linearized = SingleNodeTree::linearize_keys(balanced);
        SingleNodeTree { keys: linearized }
    }
}

impl Deref for SingleNodeTree {
    type Target = MortonKeys;

    fn deref(&self) -> &Self::Target {
        &self.keys
    }
}

impl DerefMut for SingleNodeTree {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::prelude::*;
    use rand::SeedableRng;

    use crate::types::{domain::Domain, morton::MortonKey, point::Point};

    fn tree_fixture() -> SingleNodeTree {
        let npoints: u64 = 1000;

        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0..1.0);
        let mut points = Vec::new();

        for _ in 0..npoints {
            points.push([
                between.sample(&mut range),
                between.sample(&mut range),
                between.sample(&mut range),
            ])
        }

        let points: Vec<Point> = points
            .iter()
            .map(|p| Point {
                coordinate: p.clone(),
                global_idx: 0,
                key: MortonKey::from_point(&p, &domain),
            })
            .collect();

        let keys: MortonKeys = points.iter().map(|p| p.key).collect();

        SingleNodeTree { keys }
    }

    #[test]
    fn test_linearize() {
        let mut tree = tree_fixture();
        tree.linearize();

        // Test that a linearized tree is sorted
        for i in 0..(tree.iter().len() - 1) {
            let a = tree[i];
            let b = tree[i + 1];
            assert!(a <= b);
        }

        // Test that elements in a linearized tree are unique
        let unique: HashSet<MortonKey> = tree.iter().cloned().collect();
        assert!(unique.len() == tree.len());

        // Test that a linearized tree contains no overlaps
        let mut copy: MortonKeys = tree.keys.iter().cloned().collect();
        for &key in tree.iter() {
            let ancestors = key.ancestors();
            copy.retain(|&k| k != key);

            for ancestor in &ancestors {
                assert!(!copy.contains(ancestor))
            }
        }
    }

    #[test]
    fn test_complete_region() {
        let a: MortonKey = MortonKey {
            anchor: [0, 0, 0],
            morton: 16,
        };
        let b: MortonKey = MortonKey {
            anchor: [65535, 65535, 65535],
            morton: 0b111111111111111111111111111111111111111111111111000000000010000,
        };

        let region = SingleNodeTree::complete_region(&a, &b);

        let fa = a.finest_ancestor(&b);

        let min = region.iter().min().unwrap();
        let max = region.iter().max().unwrap();

        // Test that bounds are satisfied
        assert!(a <= *min);
        assert!(b >= *max);

        // Test that FCA is an ancestor of all nodes in the result
        for node in region.iter() {
            let ancestors = node.ancestors();
            assert!(ancestors.contains(&fa));
        }

        // Test that completed region doesn't contain its bounds
        assert!(!region.contains(&a));
        assert!(!region.contains(&b));

        // Test that the compeleted region doesn't contain any overlaps
        for node in region.iter() {
            let mut ancestors = node.ancestors();
            ancestors.remove(node);
            for ancestor in ancestors.iter() {
                assert!(!region.contains(ancestor))
            }
        }

        // Test that the region is sorted
        for i in 0..region.iter().len() - 1 {
            let a = region[i];
            let b = region[i + 1];

            assert!(a <= b);
        }
    }
}
