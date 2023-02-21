use itertools::Itertools;
use std::{
    collections::{HashMap, HashSet},
    vec,
};

use solvers_traits::{
    fmm::FmmTree,
    tree::Tree,
};

use crate::{
    constants::{DEEPEST_LEVEL, LEVEL_SIZE, NCRIT, ROOT},
    implementations::impl_morton::{complete_region, encode_anchor},
    types::{
        domain::Domain,
        morton::{MortonKey, MortonKeys},
        node::{LeafNode, LeafNodes, Node, NodeData, Nodes},
        point::{Point, PointType, Points},
        single_node::SingleNodeTree,
    },
};

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

pub fn group_points_by_encoded_leaves(points: &mut Points) -> LeafNodes {
    points.sort();
    let mut nodes = Vec::new();
    let mut curr = LeafNode {
        key: points[0].encoded_key,
        points: vec![points[0].clone()],
    };
    for point in points.iter().skip(1) {
        if point.encoded_key == curr.key {
            curr.points.push(point.clone())
        } else {
            nodes.push(curr);
            curr = LeafNode {
                key: point.encoded_key,
                points: vec![point.clone()],
            }
        }
    }

    nodes
}

/// Split tree coarse blocks by counting how many particles they contain.
pub fn split_blocks(points: &mut Points, mut blocktree: MortonKeys, n_crit: usize) {
    let split_blocktree;

    loop {
        let mut new_blocktree = MortonKeys::new();

        // Map between blocks and the leaves they contain
        assign_nodes_to_points(&blocktree, points);

        let blocks = group_points_by_encoded_leaves(points);

        // Generate a new blocktree with a block's children if they violate the n_crit parameter
        let mut check = 0;
        for block in blocks.iter() {
            if block.points.len() > n_crit {
                let mut children = block.key.children();
                new_blocktree.append(&mut children)
            } else {
                new_blocktree.push(block.key);
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
    assign_nodes_to_points(&split_blocktree, points);
}

/// Create a mapping between octree nodes and the points they contain, assumed to overlap.
pub fn assign_nodes_to_points(nodes: &MortonKeys, points: &mut Points) {
    let nodes: HashSet<MortonKey> = nodes.iter().cloned().collect();

    for point in points.iter_mut() {
        // Ancestor could be the key itself
        if let Some(ancestor) = point
            .base_key
            .ancestors()
            .into_iter()
            .sorted()
            .rev()
            .find(|a| nodes.contains(a))
        {
            point.encoded_key = ancestor;
        }
    }
}

impl SingleNodeTree {
    /// Create a new single-node tree. If non-adaptive (uniform) trees are created, they are specified
    /// by a user defined maximum depth, if an adaptive tree is created it is specified by only by the
    /// user defined maximum leaf maximum occupancy n_crit.
    pub fn new(
        points: &[[PointType; 3]],
        point_data: &Vec<Vec<PointType>>,
        adaptive: bool,
        n_crit: Option<usize>,
        depth: Option<u64>,
    ) -> SingleNodeTree {
        let domain = Domain::from_local_points(points);

        let n_crit = n_crit.unwrap_or(NCRIT);
        let depth = depth.unwrap_or(DEEPEST_LEVEL);

        if adaptive {
            SingleNodeTree::adaptive_tree(points, point_data, &domain, n_crit)
        } else {
            SingleNodeTree::uniform_tree(points, point_data, &domain, depth)
        }
    }

    /// Constructor for uniform trees
    pub fn uniform_tree(
        points: &[[PointType; 3]],
        point_data: &Vec<Vec<PointType>>,
        &domain: &Domain,
        depth: u64,
    ) -> SingleNodeTree {
        // Encode points at deepest level, and map to specified depth
        let mut points: Points = points
            .iter()
            .zip(point_data.iter())
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

        assign_nodes_to_points(&leaves, &mut points);

        // Sort the points into Morton order, to perform a groupby,
        let leaves = group_points_by_encoded_leaves(&mut points);
        let mut keys = Vec::<Node>::new();

        for node in leaves.iter() {
            let ancestors = node
                .key
                .ancestors()
                .into_iter()
                .map(|k| Node {
                    key: k.clone(),
                    data: NodeData::default(),
                })
                .collect_vec();
            keys.extend(ancestors);
        }

        let depth = depth as usize;
        let keys_set: HashSet<MortonKey> = keys.iter().map(|k| k.key).collect();
        let mut key_to_index: HashMap<MortonKey, usize> = HashMap::new();
        let mut leaf_to_index: HashMap<MortonKey, usize> = HashMap::new();

        for (i, key) in keys.iter().enumerate() {
            key_to_index.insert(key.key, i);
        }

        for (i, leaf) in leaves.iter().enumerate() {
            leaf_to_index.insert(leaf.key, i);
        }

        SingleNodeTree {
            depth,
            points,
            domain,
            leaves,
            keys,
            keys_set,
            key_to_index,
            leaf_to_index,
        }
    }

    /// Constructor for adaptive trees
    pub fn adaptive_tree(
        points: &[[PointType; 3]],
        point_data: &Vec<Vec<PointType>>,
        &domain: &Domain,
        n_crit: usize,
    ) -> SingleNodeTree {
        // Encode points at deepest level
        let mut points: Points = points
            .iter()
            .zip(point_data)
            .enumerate()
            .map(|(i, (p, d))| {
                let key = MortonKey::from_point(p, &domain, DEEPEST_LEVEL);
                Point {
                    coordinate: *p,
                    base_key: key,
                    encoded_key: key,
                    global_idx: i,
                    data: d.clone(),
                }
            })
            .collect();

        // Complete the region spanned by the points
        let mut complete = MortonKeys {
            keys: points.iter().map(|p| p.encoded_key).collect_vec(),
            index: 0,
        };

        complete.linearize();
        complete.complete();

        // Find seeds (coarsest node(s))
        let mut seeds = find_seeds(&complete);

        // The tree's domain is defined by the finest first/last descendants
        let blocktree = SingleNodeTree::complete_blocktree(&mut seeds);

        // Split the blocks based on the n_crit constraint
        split_blocks(&mut points, blocktree, n_crit);

        // TODO: Check if this is OK to do, when points don't occupy the whole grid, pretty sure it's NOT.
        let mut balanced = MortonKeys {
            keys: points.iter().map(|p| p.encoded_key).collect_vec(),
            index: 0,
        };

        // Balance and linearize
        balanced.sort();
        balanced.balance();
        balanced.linearize();

        // Form leaf nodes
        let leaves = group_points_by_encoded_leaves(&mut points);

        let depth = points.iter().map(|p| p.encoded_key.level()).max().unwrap() as usize;
        let mut keys = Vec::new();

        for node in leaves.iter() {
            let ancestors = node
                .key
                .ancestors()
                .into_iter()
                .map(|k| Node {
                    key: k.clone(),
                    data: NodeData::default(),
                })
                .collect_vec();
            keys.extend(ancestors);
        }

        let keys_set: HashSet<MortonKey> = keys.iter().map(|k| k.key).collect();

        // Impose order on final keys, and create index pointer.
        let mut key_to_index: HashMap<MortonKey, usize> = HashMap::new();
        let mut leaf_to_index: HashMap<MortonKey, usize> = HashMap::new();

        for (i, key) in keys.iter().enumerate() {
            key_to_index.insert(key.key, i);
        }

        for (i, leaf) in leaves.iter().enumerate() {
            leaf_to_index.insert(leaf.key, i);
        }

        SingleNodeTree {
            depth,
            points,
            domain,
            leaves,
            keys,
            keys_set,
            key_to_index,
            leaf_to_index,
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
}

impl <'a>Tree<'a> for SingleNodeTree {
    type Domain = Domain;
    type Point = Point;
    type Points = Points;
    type NodeIndex = Node;
    type NodeIndices = Nodes;
    type LeafNodeIndex = LeafNode;
    type LeafNodeIndices = LeafNodes;
    type RawNodeIndex = MortonKey;

    fn get_depth(&self) -> usize {
        self.depth
    }

    fn get_leaves(&self) -> &Self::LeafNodeIndices {
        &self.leaves
    }

    fn get_keys(&self) -> &Self::NodeIndices {
        &self.keys
    }

    fn get_keys_mut(&mut self) -> &mut Self::NodeIndices {
        &mut self.keys
    }

    // Get domain, gets global domain in multi-node setting
    fn get_domain(&self) -> &Domain {
        &self.domain
    }

    fn get_keys_set(&self) -> &HashSet<Self::RawNodeIndex> {
        &self.keys_set
    }

    fn key_to_index(&self, key: Self::RawNodeIndex) -> usize {
        self.key_to_index(key)
    }

    fn leaf_to_index(&self, key: Self::RawNodeIndex) -> usize {
        self.leaf_to_index(key)
    }
}

impl<'a> FmmTree<'a> for SingleNodeTree {
    type FmmNodeIndex = Node;
    type FmmNodeIndices = Vec<&'a Node>;
    type FmmLeafNodeIndex = LeafNode;
    type FmmLeafNodeIndices = Vec<&'a LeafNode>;
    type FmmRawNodeIndex = MortonKey;

    // Single node trees are already locally essential trees
    fn create_let(&mut self) {}

    fn get_interaction_list(
        &'a self,
        node_index: &Self::FmmRawNodeIndex,
    ) -> Option<Self::FmmNodeIndices> {
        if node_index.level() >= 2 {
            let v_list = node_index
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| self.keys_set.contains(pnc) && !node_index.is_adjacent(pnc))
                .collect_vec();

            if !v_list.is_empty() {
                let nodes: Vec<&Node> = v_list
                    .iter()
                    .map(|k| &self.keys[self.key_to_index[k]])
                    .collect();
                return Some(nodes);
            } else {
                return None;
            }
        }
        None
    }

    fn get_near_field(
        &'a self,
        node_index: &Self::FmmRawNodeIndex,
    ) -> Option<Self::FmmLeafNodeIndices> {
        let mut u_list = Vec::<MortonKey>::new();
        let neighbours = node_index.neighbors();

        // Child level
        let mut neighbors_children_adj: Vec<MortonKey> = neighbours
            .iter()
            .flat_map(|n| n.children())
            .filter(|nc| self.keys_set.contains(nc) && node_index.is_adjacent(nc))
            .collect();

        // Key level
        let mut neighbors_adj: Vec<MortonKey> = neighbours
            .iter()
            .filter(|n| self.keys_set.contains(n) && node_index.is_adjacent(n))
            .cloned()
            .collect();

        // Parent level
        let mut parent_neighbours_adj: Vec<MortonKey> = node_index
            .parent()
            .neighbors()
            .into_iter()
            .filter(|pn| self.keys_set.contains(pn) && node_index.is_adjacent(pn))
            .collect();

        u_list.append(&mut neighbors_children_adj);
        u_list.append(&mut neighbors_adj);
        u_list.append(&mut parent_neighbours_adj);
        u_list.push(node_index.clone());

        if !u_list.is_empty() {
            let nodes: Vec<&LeafNode> = u_list
                .iter()
                .map(|k| &self.leaves[self.leaf_to_index[k]])
                .collect();
            Some(nodes)
        } else {
            None
        }
    }

    fn get_w_list(
        &'a self,
        node_index: &Self::FmmRawNodeIndex,
    ) -> Option<Self::FmmNodeIndices> {
        // Child level
        let w_list = node_index
            .neighbors()
            .iter()
            .flat_map(|n| n.children())
            .filter(|nc| self.keys_set.contains(nc) && !node_index.is_adjacent(nc))
            .collect_vec();

        if !w_list.is_empty() {
            let nodes: Vec<&Node> = w_list
                .iter()
                .map(|k| &self.keys[self.key_to_index[k]])
                .collect();
            Some(nodes)
        } else {
            None
        }
    }

    fn get_x_list(
        &'a self,
        node_index: &Self::FmmRawNodeIndex,
    ) -> Option<Self::FmmLeafNodeIndices> {
        let x_list = node_index
            .parent()
            .neighbors()
            .into_iter()
            .filter(|pn| self.keys_set.contains(pn) && !node_index.is_adjacent(pn))
            .collect_vec();

        if !x_list.is_empty() {
            let nodes: Vec<&LeafNode> = x_list
                .iter()
                .map(|k| &self.leaves[self.leaf_to_index[k]])
                .collect();
            Some(nodes)
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
        let point_data = vec![vec![1.0]; 10000];
        let depth = 4;
        let n_crit = 15;
        let tree = SingleNodeTree::new(&points, &point_data, false, Some(n_crit), Some(depth));

        // Test that particle constraint is met at leaf level
        for node in tree.leaves.iter() {
            assert!(node.points.len() <= n_crit)
        }

        // Test that the tree really is uniform
        let levels: Vec<u64> = tree
            .get_leaves()
            .iter()
            .map(|node| node.key.level())
            .collect();
        let first = levels[0];
        assert!(levels.iter().all(|key| *key == first));

        // Test that max level constraint is satisfied
        assert!(first == depth);
    }

    #[test]
    pub fn test_adaptive_tree() {
        let points = points_fixture(1000);
        let point_data = vec![vec![1.0]; 1000];
        let adaptive = true;
        let n_crit = 15;
        let tree = SingleNodeTree::new(&points, &point_data, adaptive, Some(n_crit), None);

        // Test that particle constraint is met
        for node in tree.leaves.iter() {
            assert!(node.points.len() <= n_crit)
        }

        // Test that tree is not uniform
        let levels: Vec<u64> = tree
            .get_leaves()
            .iter()
            .map(|node| node.key.level())
            .collect();
        let first = levels[0];
        assert_eq!(false, levels.iter().all(|level| *level == first));

        // Test that adjacent leaves are 2:1 balanced
        for node in tree.leaves.iter() {
            let adjacent_levels: Vec<u64> = tree
                .leaves
                .iter()
                .cloned()
                .filter(|n| node.key.is_adjacent(&n.key))
                .map(|n| n.key.level())
                .collect();

            for l in adjacent_levels.iter() {
                assert!(l.abs_diff(node.key.level()) <= 1);
            }
        }
    }

    pub fn test_no_overlaps_helper(nodes: &LeafNodes) {
        let key_set: HashSet<MortonKey> = nodes.iter().map(|n| n.key).clone().collect();

        for node in key_set.iter() {
            let ancestors = node.ancestors();
            let int: Vec<&MortonKey> = key_set.intersection(&ancestors).collect();
            assert!(int.len() == 1);
        }
    }
    pub fn test_no_overlaps_helper_morton(keys: &MortonKeys) {
        let key_set: HashSet<MortonKey> = keys.clone().collect();

        for node in key_set.iter() {
            let ancestors = node.ancestors();
            let int: Vec<&MortonKey> = key_set.intersection(&ancestors).collect();
            assert!(int.len() == 1);
        }
    }
    #[test]
    pub fn test_no_overlaps() {
        let points = points_fixture(10000);
        let point_data = vec![vec![1.0]; 10000];
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
                    data: vec![1., 0.],
                }
            })
            .collect();

        let keys = MortonKeys {
            keys: ROOT.children(),
            index: 0,
        };

        assign_nodes_to_points(&keys, &mut points);

        let nodes = group_points_by_encoded_leaves(&mut points);

        // Test that a single octant contains all the points
        for node in nodes.iter() {
            if node.points.len() > 0 {
                assert!(node.points.len() == npoints);
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
        let mut points: Vec<Point> = points_fixture(10000)
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let key = MortonKey::from_point(&p, &domain, depth);
                Point {
                    coordinate: p,
                    global_idx: i,
                    base_key: key,
                    encoded_key: key,
                    data: vec![1., 0.],
                }
            })
            // .cloned()
            .collect();

        let n_crit = 15;

        // Test case where blocks span the entire domain
        let blocktree = MortonKeys {
            keys: vec![ROOT],
            index: 0,
        };

        split_blocks(&mut points, blocktree, n_crit);
        let split_blocktree = MortonKeys {
            keys: points.iter().map(|p| p.encoded_key).collect_vec(),
            index: 0,
        };

        test_no_overlaps_helper_morton(&split_blocktree);

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

        split_blocks(&mut points, blocktree, 25);
        let split_blocktree = MortonKeys {
            keys: points.iter().map(|p| p.encoded_key).collect_vec(),
            index: 0,
        };
        test_no_overlaps_helper_morton(&split_blocktree);
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
