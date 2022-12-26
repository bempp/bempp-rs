use itertools::Itertools;
use std::collections::{HashMap, HashSet};

use solvers_traits::tree::Tree;

use crate::{
    constants::{DEEPEST_LEVEL, LEVEL_SIZE, NCRIT, ROOT},
    implementations::impl_morton::{complete_region, encode_anchor},
    types::{
        domain::Domain,
        morton::{MortonKey, MortonKeys},
        point::{Point, PointType, Points},
        single_node::SingleNodeTree,
    },
};

/// Create a mapping between points and octree nodes, assumed to overlap.
pub fn assign_points_to_nodes(points: &Points, nodes: &MortonKeys) -> HashMap<Point, MortonKey> {
    let nodes: HashSet<MortonKey> = nodes.iter().cloned().collect();

    let mut map: HashMap<Point, MortonKey> = HashMap::new();

    for point in points.iter() {
        if nodes.contains(&point.key) {
            map.insert(*point, point.key);
        } else {
            let ancestor = point
                .key
                .ancestors()
                .into_iter()
                .sorted()
                .rev()
                .find(|a| nodes.contains(a))
                .unwrap();
            map.insert(*point, ancestor);
        };
    }
    map
}

/// Create a mapping between octree nodes and the points they contain, assumed to overlap.
pub fn assign_nodes_to_points(keys: &MortonKeys, points: &Points) -> HashMap<MortonKey, Points> {
    let keys: HashSet<MortonKey> = keys.iter().cloned().collect();
    let mut map: HashMap<MortonKey, Points> = HashMap::new();

    for point in points.iter() {
        if keys.contains(&point.key) {
            map.entry(point.key).or_default().push(*point);
        } else {
            let ancestor = point
                .key
                .ancestors()
                .into_iter()
                .sorted()
                .rev()
                .find(|a| keys.contains(a))
                .unwrap();
            map.entry(ancestor).or_default().push(*point);
        }
    }
    map
}

impl SingleNodeTree {
    /// Create a new single-node tree. In non-adaptive (uniform) trees are created, they are specified
    /// by a user defined maximum depth, if an adaptive tree is created it is specified by only by the
    /// user defined maximum leaf maximum occupancy n_crit.
    pub fn new(
        points: &[[PointType; 3]],
        adaptive: bool,
        n_crit: Option<usize>,
        depth: Option<u64>,
    ) -> SingleNodeTree {
        let domain = Domain::from_local_points(points);

        let n_crit = n_crit.unwrap_or(NCRIT);
        let depth = depth.unwrap_or(DEEPEST_LEVEL) as usize;

        if adaptive {
            SingleNodeTree::adaptive_tree(adaptive, points, &domain, n_crit)
        } else {
            SingleNodeTree::uniform_tree(adaptive, points, &domain, depth)
        }
    }

    /// Constructor for uniform trees
    pub fn uniform_tree(
        adaptive: bool,
        points: &[[PointType; 3]],
        &domain: &Domain,
        depth: usize,
    ) -> SingleNodeTree {
        // Encode points at deepest level, and map to specified depth
        let points: Points = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                key: MortonKey::from_point(p, &domain, depth as u64),
                global_idx: i,
            })
            .collect();

        // Generate complete tree at specified depth
        let diameter = 1 << (DEEPEST_LEVEL - depth as u64);

        let keys = MortonKeys {
            keys: (0..LEVEL_SIZE)
                .step_by(diameter)
                .flat_map(|i| (0..LEVEL_SIZE).step_by(diameter).map(move |j| (i, j)))
                .flat_map(|(i, j)| (0..LEVEL_SIZE).step_by(diameter).map(move |k| [i, j, k]))
                .map(|anchor| {
                    let morton = encode_anchor(&anchor, depth as u64);
                    MortonKey { anchor, morton }
                })
                .collect(),
            index: 0,
        };

        let keys_to_points = assign_nodes_to_points(&keys, &points);
        let points_to_keys = assign_points_to_nodes(&points, &keys);

        SingleNodeTree {
            adaptive,
            points,
            keys,
            domain,
            points_to_keys,
            keys_to_points,
        }
    }

    /// Constructor for adaptive trees
    pub fn adaptive_tree(
        adaptive: bool,
        points: &[[PointType; 3]],
        &domain: &Domain,
        n_crit: usize,
    ) -> SingleNodeTree {
        
        // Encode points at deepest level
        let mut points: Points = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                key: MortonKey::from_point(p, &domain, DEEPEST_LEVEL),
                global_idx: i,
            })
            .collect();

        // Find unique keys corresponding to the points
        let mut complete = MortonKeys {
            keys: points.iter().map(|p| p.key).collect_vec(),
            index: 0,
        };

        complete.linearize();

        // Complete the region spanned by the points
        complete.complete();

        // Find seeds (coarsest node(s))
        let coarsest_level = complete.iter().map(|k| k.level()).min().unwrap();

        let mut seeds: MortonKeys = MortonKeys {
            keys: complete
                .into_iter()
                .filter(|k| k.level() == coarsest_level)
                .collect_vec(),
            index: 0,
        };

        seeds.sort();

        // The tree's domain is defined by the finest first/last descendants
        let ffc_root = ROOT.finest_first_child();
        let min = seeds.iter().min().unwrap();
        let fa = ffc_root.finest_ancestor(min);
        let first_child = fa.children().into_iter().min().unwrap();
        seeds.push(first_child);

        let flc_root = ROOT.finest_last_child();
        let max = seeds.iter().max().unwrap();
        let fa = flc_root.finest_ancestor(max);
        let last_child = fa.children().into_iter().max().unwrap();
        seeds.push(last_child);

        seeds.sort();

        let mut blocktree = MortonKeys::new();

        for i in 0..(seeds.iter().len() - 1) {
            let a = seeds[i];
            let b = seeds[i + 1];
            let mut tmp = complete_region(&a, &b);
            blocktree.keys.push(a);
            blocktree.keys.append(&mut tmp);
        }

        blocktree.keys.push(seeds.last().unwrap());

        blocktree.sort();

        // Split the blocks based on the n_crit constraint
        let mut balanced;
        let mut blocks_to_points;
        loop {
            let mut new_blocktree = MortonKeys::new();

            // Map between blocks and the leaves they contain
            blocks_to_points = assign_nodes_to_points(&blocktree, &points);

            // Generate a new blocktree with a block's children if they violate the n_crit parameter
            let mut check = 0;
            for (&block, points) in blocks_to_points.iter() {
                let npoints = points.len();

                if npoints > n_crit {

                    let mut children = block.children();
                    new_blocktree.append(&mut children);
                } else {
                    new_blocktree.push(block);
                    check += 1;
                }
            }

            // Return if we cycle through all blocks without splitting
            if check == blocks_to_points.len() {
                balanced = new_blocktree;
                break;
            } else {
                blocktree = new_blocktree;
            }
        }

        balanced.sort();
        
        // Balance and linearize
        balanced.balance();
        balanced.linearize();
        
        // Find new maps between points and balanced tree
        let points_to_keys = assign_points_to_nodes(&points, &balanced);

        points = points
            .iter()
            .map(|p| Point {
                coordinate: p.coordinate,
                global_idx: p.global_idx,
                key: *points_to_keys.get(p).unwrap(),
            })
            .collect();

        let keys_to_points = assign_nodes_to_points(&balanced, &points);
        let keys = balanced;

        SingleNodeTree {
            adaptive,
            points,
            keys,
            domain,
            points_to_keys,
            keys_to_points,
        }
    }
}
impl Tree for SingleNodeTree {
    type Domain = Domain;
    type Point = Point;
    type Points = Points;
    type NodeIndex = MortonKey;
    type NodeIndices = MortonKeys;

    // Get adaptivity information
    fn get_adaptive(&self) -> bool {
        self.adaptive
    }

    // Get all keys, gets local keys in multi-node setting
    fn get_keys(&self) -> &MortonKeys {
        &self.keys
    }

    // Get all points, gets local keys in multi-node setting
    fn get_points(&self) -> &Points {
        &self.points
    }

    // Get domain, gets global domain in multi-node setting
    fn get_domain(&self) -> &Domain {
        &self.domain
    }

    // Get tree node key associated with a given point
    fn map_point_to_key(&self, point: &Point) -> Option<&MortonKey> {
        self.points_to_keys.get(point)
    }

    // Get points associated with a tree node key
    fn map_key_to_points(&self, key: &MortonKey) -> Option<&Points> {
        self.keys_to_points.get(key)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use rand::prelude::*;
    use rand::SeedableRng;

    pub fn points_fixture(npoints: i32) -> Vec<[f64; 3]> {
        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0..1.0);
        let mut points: Vec<[PointType; 3]> = Vec::new();

        for _ in 0..npoints {
            points.push([
                between.sample(&mut range),
                between.sample(&mut range),
                between.sample(&mut range),
            ])
        }

        points
    }

    #[test]
    pub fn test_uniform_tree() {
        let points = points_fixture(10000);
        let depth = 4;
        let n_crit = 15;
        let tree = SingleNodeTree::new(&points, false, Some(n_crit), Some(depth));

        // Test that particle constraint is met at this level
        for (_, (_, points)) in tree.keys_to_points.iter().enumerate() {
            assert!(points.len() <= n_crit);
        }

        // Test that the tree really is uniform
        let levels: Vec<u64> = tree.get_keys().iter().map(|key| key.level()).collect();
        let first = levels[0];
        assert!(levels.iter().all(|key| *key == first));

        // Test that max level constraint is satisfied
        assert!(first == depth);

        // Test that tree is complete
        assert_eq!(tree.get_keys().len(), 8_i64.pow(depth as u32) as usize);
    }

    #[test]
    pub fn test_adaptive_tree() {
        let points = points_fixture(1000);
        let adaptive = true;
        let n_crit = 15;
        let tree = SingleNodeTree::new(&points, adaptive, Some(n_crit), None);

        // Test that particle constraint is met
        for (_, (_, points)) in tree.keys_to_points.iter().enumerate() {
            assert!(points.len() <= n_crit);
        }

        // Test that tree is not uniform
        let levels: Vec<u64> = tree.get_keys().iter().map(|key| key.level()).collect();
        let first = levels[0];
        assert_eq!(false, levels.iter().all(|level| *level == first));

        // Test for overlaps in balanced tree
        let keys: Vec<MortonKey> = tree.keys.iter().cloned().collect();
        for key in keys.iter() {
            if !keys.iter().contains(key) {
                let mut ancestors = key.ancestors();
                ancestors.remove(key);

                for ancestor in ancestors.iter() {
                    assert!(!keys.contains(ancestor));
                }
            }
        }

        // Test that adjacent keys are 2:1 balanced
        for key in keys.iter() {
            let adjacent_levels: Vec<u64> = keys
                .iter()
                .cloned()
                .filter(|k| key.is_adjacent(k))
                .map(|a| a.level())
                .collect();

            for l in adjacent_levels.iter() {
                assert!(l.abs_diff(key.level()) <= 1);
            }
        }

        // Test that the the adaptive tree is complete
        let max_level = keys.iter().map(|k| k.level()).max().unwrap();
        let diameter = 1 << (DEEPEST_LEVEL - max_level as u64);

        let uniform = MortonKeys {
            keys: (0..LEVEL_SIZE)
                .step_by(diameter)
                .flat_map(|i| (0..LEVEL_SIZE).step_by(diameter).map(move |j| (i, j)))
                .flat_map(|(i, j)| (0..LEVEL_SIZE).step_by(diameter).map(move |k| [i, j, k]))
                .map(|anchor| {
                    let morton = encode_anchor(&anchor, max_level as u64);
                    MortonKey { anchor, morton }
                })
                .collect(),
            index: 0,
        };

        let uniform_set: HashSet<MortonKey> = uniform.keys.into_iter().collect();
        for node in uniform_set.iter() {

            let ancestors = node.ancestors();

            let int: Vec<&MortonKey> = ancestors.intersection(&uniform_set).collect();

            assert!(int.len() > 0);
        }
    
    }

    pub fn test_no_overlaps_helper(tree: &SingleNodeTree) {
        let tree_set: HashSet<MortonKey> = tree.get_keys().clone().collect();

        for node in tree_set.iter() {
            let ancestors = node.ancestors();
            let int: Vec<&MortonKey> = tree_set.intersection(&ancestors).collect();
            assert!(int.len() == 1);
        }
    }
    #[test]
    pub fn test_no_overlaps() {
        let points = points_fixture(10000);
        let uniform = SingleNodeTree::new(&points, false, Some(150), Some(4));
        let adaptive = SingleNodeTree::new(&points, true, Some(150), None);
        test_no_overlaps_helper(&uniform);
        test_no_overlaps_helper(&adaptive);
    }
}
