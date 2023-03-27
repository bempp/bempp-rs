use itertools::Itertools;
use std::{
    collections::{HashMap, HashSet},
    vec,
};

use bempp_traits::tree::{FmmInteractionLists, Tree};

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

impl SingleNodeTree {
    /// Constructor for uniform trees
    pub fn uniform_tree(
        points: &[[PointType; 3]],
        point_data: &[Vec<PointType>],
        &domain: &Domain,
        depth: u64,
    ) -> SingleNodeTree {
        // Encode points at deepest level, and map to specified depth
        let mut points: Points = points
            .iter()
            .zip(point_data)
            .enumerate()
            .map(|(i, (&p, d))| {
                let base_key = MortonKey::from_point(&p, &domain, DEEPEST_LEVEL);
                let encoded_key = MortonKey::from_point(&p, &domain, depth);
                Point {
                    coordinate: p,
                    base_key,
                    encoded_key,
                    global_idx: i,
                    data: d.clone(),
                }
            })
            .collect();
        points.sort();

        // Generate complete tree at specified depth
        let diameter = 1 << (DEEPEST_LEVEL - depth);

        let leaves = MortonKeys {
            keys: (0..LEVEL_SIZE)
                .step_by(diameter)
                .flat_map(|i| (0..LEVEL_SIZE).step_by(diameter).map(move |j| (i, j)))
                .flat_map(|(i, j)| (0..LEVEL_SIZE).step_by(diameter).map(move |k| [i, j, k]))
                .map(|anchor| {
                    let morton = encode_anchor(&anchor, depth);
                    MortonKey { anchor, morton }
                })
                .collect(),
            index: 0,
        };

        // Assign keys to points
        let unmapped = SingleNodeTree::assign_nodes_to_points(&leaves, &mut points);

        // Assign leaves to points, and collect all leaf nodes
        let leaves_to_points = points
            .points
            .iter()
            .enumerate()
            .fold(
                (HashMap::new(), 0, points.points[0].clone()),
                |(mut leaves_to_points, curr_idx, curr), (i, point)| {
                    if point.encoded_key != curr.encoded_key {
                        leaves_to_points.insert(curr.encoded_key, (curr_idx, i));

                        (leaves_to_points, i, point.clone())
                    } else {
                        (leaves_to_points, curr_idx, curr)
                    }
                },
            )
            .0;

        // Add unmapped leaves
        let leaves = MortonKeys {
            keys: leaves_to_points
                .keys()
                .cloned()
                .chain(unmapped.iter().copied())
                .collect_vec(),
            index: 0,
        };

        // Find all keys in tree
        let tmp: HashSet<MortonKey> = leaves
            .iter()
            .flat_map(|leaf| leaf.ancestors().into_iter())
            .collect();

        let mut keys = MortonKeys {
            keys: tmp.into_iter().collect_vec(),
            index: 0,
        };

        let leaves_set: HashSet<MortonKey> = leaves.iter().cloned().collect();
        let keys_set: HashSet<MortonKey> = keys.iter().cloned().collect();

        // Group by level to perform efficient lookup of nodes
        keys.sort();
        let levels_to_keys = keys
            .iter()
            .enumerate()
            .fold(
                (HashMap::new(), 0, keys[0].clone()),
                |(mut levels_to_keys, curr_idx, curr), (i, key)| {
                    if key.level() != curr.level() {
                        levels_to_keys.insert(key.level(), (curr_idx, i));

                        (levels_to_keys, i, key.clone())
                    } else {
                        (levels_to_keys, curr_idx, curr)
                    }
                },
            )
            .0;

        SingleNodeTree {
            depth,
            points,
            domain,
            leaves,
            keys,
            leaves_to_points,
            leaves_set,
            keys_set,
            levels_to_keys,
        }
    }

    /// Constructor for adaptive trees
    pub fn adaptive_tree(
        points: &[[PointType; 3]],
        point_data: &[Vec<PointType>],
        &domain: &Domain,
        n_crit: u64,
    ) -> SingleNodeTree {
        // Encode points at deepest level
        let mut points: Points = points
            .iter()
            .zip(point_data.iter())
            .enumerate()
            .map(|(i, (&p, d))| {
                let key = MortonKey::from_point(&p, &domain, DEEPEST_LEVEL);
                Point {
                    coordinate: p,
                    base_key: key,
                    encoded_key: key,
                    global_idx: i,
                    data: d.clone(),
                }
            })
            .collect();
        points.sort();

        // Complete the region spanned by the points
        let mut complete = MortonKeys {
            keys: points.points.iter().map(|p| p.encoded_key).collect_vec(),
            index: 0,
        };

        complete.linearize();
        complete.complete();

        // Find seeds (coarsest node(s))
        let mut seeds = SingleNodeTree::find_seeds(&complete);

        // The tree's domain is defined by the finest first/last descendants
        let blocktree = SingleNodeTree::complete_blocktree(&mut seeds);

        // Split the blocks based on the n_crit constraint
        let mut balanced = SingleNodeTree::split_blocks(&mut points, blocktree, n_crit as usize);

        // Balance and linearize
        balanced.sort();
        balanced.balance();
        balanced.linearize();

        // Assign leaves to points, and collect all leaf nodes
        let unmapped = SingleNodeTree::assign_nodes_to_points(&balanced, &mut points);
        let leaves_to_points = points
            .points
            .iter()
            .enumerate()
            .fold(
                (HashMap::new(), 0, points.points[0].clone()),
                |(mut leaves_to_points, curr_idx, curr), (i, point)| {
                    if point.encoded_key != curr.encoded_key {
                        leaves_to_points.insert(curr.encoded_key, (curr_idx, i));

                        (leaves_to_points, i, point.clone())
                    } else {
                        (leaves_to_points, curr_idx, curr)
                    }
                },
            )
            .0;

        // Add unmapped leaves
        let leaves = MortonKeys {
            keys: leaves_to_points
                .keys()
                .cloned()
                .chain(unmapped.iter().copied())
                .collect_vec(),
            index: 0,
        };

        // Find all keys in tree
        let tmp: HashSet<MortonKey> = leaves
            .iter()
            .flat_map(|leaf| leaf.ancestors().into_iter())
            .collect();

        let mut keys = MortonKeys {
            keys: tmp.into_iter().collect_vec(),
            index: 0,
        };

        // Find depth
        let depth = points
            .points
            .iter()
            .map(|p| p.encoded_key.level())
            .max()
            .unwrap();

        let leaves_set: HashSet<MortonKey> = leaves.iter().cloned().collect();
        let keys_set: HashSet<MortonKey> = keys.iter().cloned().collect();

        // Group by level to perform efficient lookup of nodes
        keys.sort();
        let levels_to_keys = keys
            .iter()
            .enumerate()
            .fold(
                (HashMap::new(), 0, keys[0].clone()),
                |(mut levels_to_keys, curr_idx, curr), (i, key)| {
                    if key.level() != curr.level() {
                        levels_to_keys.insert(key.level(), (curr_idx, i));

                        (levels_to_keys, i, key.clone())
                    } else {
                        (levels_to_keys, curr_idx, curr)
                    }
                },
            )
            .0;

        SingleNodeTree {
            depth,
            points,
            domain,
            leaves,
            keys,
            leaves_to_points,
            leaves_set,
            keys_set,
            levels_to_keys,
        }
    }

    pub fn complete_blocktree(seeds: &mut MortonKeys) -> MortonKeys {
        let ffc_root = ROOT.finest_first_child();
        let min = seeds.iter().min().unwrap();
        let fa = ffc_root.finest_ancestor(min);
        let first_child = fa.children().into_iter().min().unwrap();

        // Check for overlap
        if first_child < *min {
            seeds.push(first_child)
        }

        let flc_root = ROOT.finest_last_child();
        let max = seeds.iter().max().unwrap();
        let fa = flc_root.finest_ancestor(max);
        let last_child = fa.children().into_iter().max().unwrap();

        if last_child > *max
            && !max.ancestors().contains(&last_child)
            && !last_child.ancestors().contains(max)
        {
            seeds.push(last_child);
        }

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

        blocktree
    }

    pub fn find_seeds(leaves: &MortonKeys) -> MortonKeys {
        let coarsest_level = leaves.iter().map(|k| k.level()).min().unwrap();

        let mut seeds: MortonKeys = MortonKeys {
            keys: leaves
                .iter()
                .filter(|k| k.level() == coarsest_level)
                .cloned()
                .collect_vec(),
            index: 0,
        };

        seeds.sort();
        seeds
    }

    /// Split tree coarse blocks by counting how many particles they contain.
    pub fn split_blocks(
        points: &mut Points,
        mut blocktree: MortonKeys,
        n_crit: usize,
    ) -> MortonKeys {
        let split_blocktree;
        let mut blocks_to_points;
        loop {
            let mut new_blocktree = MortonKeys::new();

            // Map between blocks and the leaves they contain
            let unmapped = SingleNodeTree::assign_nodes_to_points(&blocktree, points);
            blocks_to_points = points
                .points
                .iter()
                .enumerate()
                .fold(
                    (HashMap::new(), 0, points.points[0].clone()),
                    |(mut blocks_to_points, curr_idx, curr), (i, point)| {
                        if point.encoded_key != curr.encoded_key {
                            blocks_to_points.insert(curr.encoded_key, (curr_idx, i));

                            (blocks_to_points, i, point.clone())
                        } else {
                            (blocks_to_points, curr_idx, curr)
                        }
                    },
                )
                .0;

            // Collect all blocks, including those which haven't been mapped
            let mut blocks = blocks_to_points.keys().cloned().collect_vec();
            // Add empty nodes to blocks.
            for key in unmapped.iter() {
                blocks.push(*key)
            }

            // Generate a new blocktree with a block's children if they violate the n_crit parameter
            let mut check = 0;

            for block in blocks.iter() {
                if let Some((l, r)) = blocks_to_points.get(block) {
                    if (r - l) > n_crit {
                        let mut children = block.children();
                        new_blocktree.append(&mut children);
                    } else {
                        new_blocktree.push(*block);
                        check += 1;
                    }
                } else {
                    // Retain unmapped blocks
                    new_blocktree.push(*block);
                    check += 1;
                }
            }

            // Return if we cycle through all blocks without splitting
            if check == blocks.len() {
                split_blocktree = new_blocktree;
                break;
            } else {
                blocktree = new_blocktree;
            }
        }
        split_blocktree
    }

    /// Create a mapping between octree nodes and the points they contain, assumed to overlap.
    /// Return any keys that are unmapped.
    pub fn assign_nodes_to_points(nodes: &MortonKeys, points: &mut Points) -> MortonKeys {
        let mut map: HashMap<MortonKey, bool> = HashMap::new();
        for node in nodes.iter() {
            map.insert(*node, false);
        }

        for point in points.points.iter_mut() {
            // Ancestor could be the key itself
            if let Some(ancestor) = point
                .base_key
                .ancestors()
                .into_iter()
                .sorted()
                .rev()
                .find(|a| map.contains_key(a))
            {
                point.encoded_key = ancestor;
                map.insert(ancestor, true);
            }
        }

        let mut unmapped = MortonKeys::new();

        for (node, is_mapped) in map.iter() {
            if !is_mapped {
                unmapped.push(*node)
            }
        }

        unmapped
    }
}

impl Tree for SingleNodeTree {
    type Domain = Domain;
    type NodeIndex = MortonKey;
    type NodeIndexSlice<'a> = &'a [MortonKey];
    type NodeIndices = MortonKeys;
    type Point = Point;
    type PointSlice<'a> = &'a [Point];
    type PointData = Vec<f64>;
    type PointDataSlice<'a> = &'a [Vec<f64>];

    /// Create a new single-node tree. If non-adaptive (uniform) trees are created, they are specified
    /// by a user defined maximum depth, if an adaptive tree is created it is specified by only by the
    /// user defined maximum leaf maximum occupancy n_crit.
    fn new<'a>(
        points: Self::PointSlice<'a>,
        point_data: Self::PointDataSlice<'a>,
        adaptive: bool,
        n_crit: Option<u64>,
        depth: Option<u64>,
    ) -> SingleNodeTree {
        // HACK: Come back and reconcile a runtime point dimension detector
        let points = points.into_iter().map(|p| p.coordinate).collect_vec();

        let domain = Domain::from_local_points(&points[..]);

        let n_crit = n_crit.unwrap_or(NCRIT);
        let depth = depth.unwrap_or(DEEPEST_LEVEL);

        if adaptive {
            SingleNodeTree::adaptive_tree(&points[..], point_data, &domain, n_crit)
        } else {
            SingleNodeTree::uniform_tree(&points[..], point_data, &domain, depth)
        }
    }

    fn get_depth(&self) -> u64 {
        self.depth
    }

    fn get_domain<'a>(&'a self) -> &'a Self::Domain {
        &self.domain
    }

    fn get_keys<'a>(&'a self, level: u64) -> Option<Self::NodeIndexSlice<'a>> {
        if let Some(&(l, r)) = self.levels_to_keys.get(&level) {
            Some(&self.keys[l..r])
        } else {
            None
        }
    }

    fn get_all_keys<'a>(&'a self) -> Option<Self::NodeIndexSlice<'a>> {
        Some(&self.keys)
    }

    fn get_leaves<'a>(&'a self) -> Self::NodeIndexSlice<'a> {
        &self.leaves
    }

    fn get_points<'a>(&'a self, key: &Self::NodeIndex) -> Option<Self::PointSlice<'a>> {
        if let Some(&(l, r)) = self.leaves_to_points.get(key) {
            Some(&self.points.points[l..r])
        } else {
            None
        }
    }

    fn is_leaf(&self, key: &Self::NodeIndex) -> bool {
        self.leaves_set.contains(key)
    }

    fn is_node(&self, key: &Self::NodeIndex) -> bool {
        self.keys_set.contains(key)
    }
}

impl FmmInteractionLists for SingleNodeTree {
    type Tree = Self;

    fn get_v_list<'a>(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices> {
        if key.level() >= 2 {
            let v_list = key
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| self.keys_set.contains(pnc) && !key.is_adjacent(pnc))
                .collect_vec();

            if !v_list.is_empty() {
                return Some(MortonKeys {
                    keys: v_list,
                    index: 0,
                });
            } else {
                return None;
            }
        }
        None
    }

    fn get_u_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices> {
        let mut u_list = Vec::<MortonKey>::new();
        let neighbours = key.neighbors();

        // Child level
        let mut neighbors_children_adj: Vec<MortonKey> = neighbours
            .iter()
            .flat_map(|n| n.children())
            .filter(|nc| self.keys_set.contains(nc) && key.is_adjacent(nc))
            .collect();

        // Key level
        let mut neighbors_adj: Vec<MortonKey> = neighbours
            .iter()
            .filter(|n| self.keys_set.contains(n) && key.is_adjacent(n))
            .cloned()
            .collect();

        // Parent level
        let mut parent_neighbours_adj: Vec<MortonKey> = key
            .parent()
            .neighbors()
            .into_iter()
            .filter(|pn| self.keys_set.contains(pn) && key.is_adjacent(pn))
            .collect();

        u_list.append(&mut neighbors_children_adj);
        u_list.append(&mut neighbors_adj);
        u_list.append(&mut parent_neighbours_adj);
        u_list.push(*key);

        if !u_list.is_empty() {
            Some(MortonKeys {
                keys: u_list,
                index: 0,
            })
        } else {
            None
        }
    }

    fn get_w_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices> {
        let w_list = key
            .neighbors()
            .iter()
            .flat_map(|n| n.children())
            .filter(|nc| self.keys_set.contains(nc) && !key.is_adjacent(nc))
            .collect_vec();

        if !w_list.is_empty() {
            Some(MortonKeys {
                keys: w_list,
                index: 0,
            })
        } else {
            None
        }
    }

    fn get_x_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices> {
        let x_list = key
            .parent()
            .neighbors()
            .into_iter()
            .filter(|pn| self.keys_set.contains(pn) && !key.is_adjacent(pn))
            .collect_vec();

        if !x_list.is_empty() {
            Some(MortonKeys {
                keys: x_list,
                index: 0,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::prelude::*;
    use rand::SeedableRng;

    pub fn points_fixture(npoints: i32) -> Vec<Point> {
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

        let points = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                global_idx: i,
                base_key: MortonKey::default(),
                encoded_key: MortonKey::default(),
                data: Vec::new(),
            })
            .collect_vec();
        points
    }

    #[test]
    pub fn test_uniform_tree() {
        let npoints = 10000;
        let points = points_fixture(npoints);
        let point_data = vec![vec![1.]; npoints as usize];
        let depth = 3;
        let n_crit = 150;
        let tree = SingleNodeTree::new(&points, &point_data, false, Some(n_crit), Some(depth));

        // Test that the tree really is uniform
        let levels: Vec<u64> = tree.get_leaves().iter().map(|node| node.level()).collect();
        let first = levels[0];
        assert!(levels.iter().all(|key| *key == first));

        // Test that max level constraint is satisfied
        assert!(first == depth);
    }

    #[test]
    pub fn test_adaptive_tree() {
        let npoints = 10000;
        let points = points_fixture(npoints);
        let point_data = vec![vec![1.]; npoints as usize];

        let adaptive = true;
        let n_crit = 150;
        let tree = SingleNodeTree::new(&points, &point_data, adaptive, Some(n_crit), None);

        // Test that tree is not uniform
        let levels: Vec<u64> = tree.get_leaves().iter().map(|node| node.level()).collect();
        let first = levels[0];
        assert_eq!(false, levels.iter().all(|level| *level == first));

        // Test that adjacent leaves are 2:1 balanced
        for node in tree.leaves.iter() {
            let adjacent_levels: Vec<u64> = tree
                .leaves
                .iter()
                .cloned()
                .filter(|n| node.is_adjacent(&n))
                .map(|n| n.level())
                .collect();
            for l in adjacent_levels.iter() {
                assert!(l.abs_diff(node.level()) <= 1);
            }
        }
    }

    pub fn test_no_overlaps_helper(nodes: &[MortonKey]) {
        let key_set: HashSet<MortonKey> = nodes.iter().cloned().collect();

        for node in key_set.iter() {
            let ancestors = node.ancestors();
            let int: Vec<&MortonKey> = key_set.intersection(&ancestors).collect();
            assert!(int.len() == 1);
        }
    }

    #[test]
    pub fn test_no_overlaps() {
        let npoints = 10000;
        let points = points_fixture(npoints);
        let point_data = vec![vec![1.]; npoints as usize];
        let uniform = SingleNodeTree::new(&points, &point_data, false, Some(150), Some(4));
        let adaptive = SingleNodeTree::new(&points, &point_data, true, Some(150), None);
        test_no_overlaps_helper(uniform.get_leaves());
        test_no_overlaps_helper(adaptive.get_leaves());
    }

    #[test]
    pub fn test_assign_nodes_to_points() {
        // Generate points in a single octant of the domain
        let npoints = 10;
        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0..0.5);
        let mut points: Vec<[PointType; 3]> = Vec::new();

        for _ in 0..npoints {
            points.push([
                between.sample(&mut range),
                between.sample(&mut range),
                between.sample(&mut range),
            ])
        }

        let domain = Domain {
            origin: [0.0, 0.0, 0.0],
            diameter: [1.0, 1.0, 1.0],
        };
        let depth = 1;

        let mut points: Points = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let key = MortonKey::from_point(p, &domain, depth);
                Point {
                    coordinate: *p,
                    encoded_key: key,
                    base_key: key,
                    global_idx: i,
                    data: Vec::new(),
                }
            })
            .collect();

        let keys = MortonKeys {
            keys: ROOT.children(),
            index: 0,
        };

        SingleNodeTree::assign_nodes_to_points(&keys, &mut points);

        let leaves_to_points = points
            .points
            .iter()
            .enumerate()
            .fold(
                (HashMap::new(), 0, points.points[0].clone()),
                |(mut leaves_to_points, curr_idx, curr), (i, point)| {
                    if point.encoded_key != curr.encoded_key {
                        leaves_to_points.insert(curr.encoded_key, (curr_idx, i + 1));

                        (leaves_to_points, i + 1, point.clone())
                    } else {
                        (leaves_to_points, curr_idx, curr)
                    }
                },
            )
            .0;

        // Test that a single octant contains all the points
        for (_, (l, r)) in leaves_to_points.iter() {
            if (r - l) > 0 {
                assert!((r - l) == npoints);
            }
        }
    }

    #[test]
    pub fn test_split_blocks() {
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1.0, 1.0, 1.0],
        };
        let depth = 5;
        let mut points = Points {
            points: points_fixture(10000),
            index: 0,
        };

        let n_crit = 15;

        // Test case where blocks span the entire domain
        let blocktree = MortonKeys {
            keys: vec![ROOT],
            index: 0,
        };

        SingleNodeTree::split_blocks(&mut points, blocktree, n_crit);
        let split_blocktree = MortonKeys {
            keys: points.points.iter().map(|p| p.encoded_key).collect_vec(),
            index: 0,
        };

        test_no_overlaps_helper(&split_blocktree);

        // Test case where the blocktree only partially covers the area
        let mut children = ROOT.children();
        children.sort();

        let a = children[0];
        let b = children[6];

        let mut seeds = MortonKeys {
            keys: vec![a, b],
            index: 0,
        };

        let blocktree = SingleNodeTree::complete_blocktree(&mut seeds);

        SingleNodeTree::split_blocks(&mut points, blocktree, 25);
        let split_blocktree = MortonKeys {
            keys: points.points.iter().map(|p| p.encoded_key).collect_vec(),
            index: 0,
        };
        test_no_overlaps_helper(&split_blocktree);
    }

    #[test]
    fn test_complete_blocktree() {
        let a = ROOT.first_child();
        let b = ROOT.children().last().unwrap().clone();

        let mut seeds = MortonKeys {
            keys: vec![a, b],
            index: 0,
        };

        let mut blocktree = SingleNodeTree::complete_blocktree(&mut seeds);

        blocktree.sort();

        let mut children = ROOT.children();
        children.sort();
        // Test that the blocktree is completed
        assert_eq!(blocktree.len(), 8);

        for (a, b) in children.iter().zip(blocktree.iter()) {
            assert_eq!(a, b)
        }
    }
}
