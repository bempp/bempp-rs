use itertools::Itertools;
use std::collections::{HashMap, HashSet};

use mpi::{
    topology::{Rank, UserCommunicator},
    traits::*,
};

use hyksort::hyksort;
use solvers_traits::tree::Tree;

use crate::{
    constants::{DEEPEST_LEVEL, K, LEVEL_SIZE, NCRIT, ROOT},
    implementations::{
        impl_morton::{complete_region, encode_anchor, point_to_anchor},
        impl_single_node::{
            assign_nodes_to_points, assign_points_to_nodes, find_seeds, split_blocks,
        },
    },
    types::{
        domain::Domain,
        morton::{KeyType, MortonKey, MortonKeys},
        multi_node::MultiNodeTree,
        point::{Point, PointType, Points},
    },
};

impl MultiNodeTree {
    /// Create a new MultiNodeTree from a set of distributed points which define a domain.
    pub fn new(
        world: &UserCommunicator,
        k: Option<i32>,
        points: &[[PointType; 3]],
        adaptive: bool,
        n_crit: Option<usize>,
        depth: Option<u64>,
    ) -> MultiNodeTree {
        let domain = Domain::from_global_points(points, world);

        let n_crit = n_crit.unwrap_or(NCRIT);
        let depth = depth.unwrap_or(DEEPEST_LEVEL);
        let k = k.unwrap_or(K);

        if adaptive {
            MultiNodeTree::adaptive_tree(world, k, points, &domain, n_crit)
        } else {
            MultiNodeTree::uniform_tree(world, k, points, &domain, depth)
        }
    }

    /// Specialization for uniform trees
    pub fn uniform_tree(
        world: &UserCommunicator,
        k: i32,
        points: &[[PointType; 3]],
        domain: &Domain,
        depth: u64,
    ) -> MultiNodeTree {
        // Encode points at specified depth
        let mut points = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                key: MortonKey::from_point(p, &domain, depth),
                global_idx: i,
            })
            .collect();

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();
        hyksort(&mut points, k, comm);

        // 2.ii Find leaf keys on each processor
        let min = points.iter().min().unwrap().key;
        let max = points.iter().max().unwrap().key;

        let diameter = 1 << (DEEPEST_LEVEL - depth as u64);

        let mut leaves = MortonKeys {
            keys: (0..LEVEL_SIZE)
                .step_by(diameter)
                .flat_map(|i| (0..LEVEL_SIZE).step_by(diameter).map(move |j| (i, j)))
                .flat_map(|(i, j)| (0..LEVEL_SIZE).step_by(diameter).map(move |k| [i, j, k]))
                .map(|anchor| {
                    let morton = encode_anchor(&anchor, depth as u64);
                    MortonKey { anchor, morton }
                })
                .filter(|k| (k >= &min) && (k <= &max))
                .collect(),
            index: 0,
        };

        // 3. Create bi-directional maps between keys and points
        let points_to_leaves= assign_points_to_nodes(&points, &leaves);
        let leaves_to_points = assign_nodes_to_points(&leaves, &points);
        
        // Only retain keys that contain points
        leaves = MortonKeys {
            keys: leaves_to_points.keys().cloned().collect(),
            index: 0,
        };

        let leaves_set: HashSet<MortonKey> = leaves.iter().cloned().collect();

        let min = leaves.iter().min().unwrap();
        let max = leaves.iter().max().unwrap();
        let range = [world.rank() as KeyType, min.morton, max.morton];
        
        MultiNodeTree {
            world: world.duplicate(),
            adaptive: false,
            points,
            leaves,
            leaves_set,
            domain: *domain,
            points_to_leaves,
            leaves_to_points,
            range
        }
    }

    /// Specialization for adaptive tree.
    pub fn adaptive_tree(
        world: &UserCommunicator,
        k: i32,
        points: &[[PointType; 3]],
        domain: &Domain,
        n_crit: usize,
    ) -> MultiNodeTree {
        // 1. Encode Points to Leaf Morton Keys, add a global index related to the processor
        let mut points: Points = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                global_idx: i,
                key: MortonKey::from_point(p, domain, DEEPEST_LEVEL),
            })
            .collect();

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();

        hyksort(&mut points, k, comm);

        // 2.ii Find unique leaf keys on each processor
        let mut local = MortonKeys {
            keys: points.iter().map(|p| p.key).collect(),
            index: 0,
        };

        // 3. Linearise received keys (remove overlaps if they exist).
        local.linearize();

        // 4. Complete region spanned by node.
        local.complete();

        // 5.i Find seeds and compute the coarse blocktree
        let mut seeds = find_seeds(&local);

        let blocktree = MultiNodeTree::complete_blocktree(world, &mut seeds);

        // 5.ii any data below the min seed sent to partner process
        let points = MultiNodeTree::transfer_points_to_blocktree(world, &points, &blocktree);

        // 6. Split blocks based on ncrit constraint
        let mut locally_balanced = split_blocks(&points, blocktree, n_crit);

        locally_balanced.sort();

        // 7. Create a minimal balanced octree for local octants spanning their domain and linearize
        locally_balanced.balance();
        locally_balanced.linearize();

        // 8. Find new maps between points and locally balanced tree
        let points_to_locally_balanced = assign_points_to_nodes(&points, &locally_balanced);
        let mut points: Points = points
            .iter()
            .map(|p| Point {
                coordinate: p.coordinate,
                global_idx: p.global_idx,
                key: *points_to_locally_balanced.get(p).unwrap(),
            })
            .collect();

        // 9. Perform another distributed sort and remove overlaps locally
        let comm = world.duplicate();

        hyksort(&mut points, k, comm);

        let mut globally_balanced = MortonKeys {
            keys: points.iter().map(|p| p.key).collect(),
            index: 0,
        };
        globally_balanced.linearize();

        // 10. Find final bidirectional maps to non-overlapping tree
        let points_to_globally_balanced = assign_points_to_nodes(&points, &globally_balanced);
        let globally_balanced_to_points = assign_nodes_to_points(&globally_balanced, &points);
        let points: Points = points
            .iter()
            .map(|p| Point {
                coordinate: p.coordinate,
                global_idx: p.global_idx,
                key: *points_to_globally_balanced.get(p).unwrap(),
            })
            .collect();
        
        let leaves_set: HashSet<MortonKey> = globally_balanced.iter().cloned().collect();

        let min = globally_balanced.iter().min().unwrap();
        let max = globally_balanced.iter().max().unwrap();
        let range = [world.rank() as KeyType, min.morton, max.morton];

        MultiNodeTree {
            world: world.duplicate(),
            adaptive: true,
            points,
            leaves: globally_balanced,
            leaves_set,
            domain: *domain,
            points_to_leaves: points_to_globally_balanced,
            leaves_to_points: globally_balanced_to_points,
            range
        }
    }
    
    /// Complete a distributed block tree from the seed octants, algorithm 4 in [1] (parallel).
    fn complete_blocktree(world: &UserCommunicator, seeds: &mut MortonKeys) -> MortonKeys {
        let rank = world.rank();
        let size = world.size();

        // Define the tree's global domain with the finest first/last descendants
        if rank == 0 {
            let ffc_root = ROOT.finest_first_child();
            let min = seeds.iter().min().unwrap();
            let fa = ffc_root.finest_ancestor(min);
            let first_child = fa.children().into_iter().min().unwrap();
            seeds.push(first_child);
            seeds.sort();
        }

        if rank == (size - 1) {
            let flc_root = ROOT.finest_last_child();
            let max = seeds.iter().max().unwrap();
            let fa = flc_root.finest_ancestor(max);
            let last_child = fa.children().into_iter().max().unwrap();
            seeds.push(last_child);
        }

        let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
        let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

        let previous_process = world.process_at_rank(previous_rank);
        let next_process = world.process_at_rank(next_rank);

        // Send required data to partner process.
        if rank > 0 {
            let min = *seeds.iter().min().unwrap();
            previous_process.send(&min);
        }

        let mut boundary = MortonKey::default();

        if rank < (size - 1) {
            next_process.receive_into(&mut boundary);
            seeds.push(boundary);
        }

        // Complete region between seeds at each process
        let mut complete = MortonKeys::new();

        for i in 0..(seeds.iter().len() - 1) {
            let a = seeds[i];
            let b = seeds[i + 1];

            let mut tmp: Vec<MortonKey> = complete_region(&a, &b);
            complete.keys.push(a);
            complete.keys.append(&mut tmp);
        }

        if rank == (size - 1) {
            complete.keys.push(seeds.last().unwrap());
        }

        complete.sort();
        complete
    }

    // Transfer points to correct processor based on the coarse distributed blocktree.
    fn transfer_points_to_blocktree(
        world: &UserCommunicator,
        points: &[Point],
        blocktree: &[MortonKey],
    ) -> Points {
        let rank = world.rank();
        let size = world.size();

        let mut received_points = Points::new();

        let min = blocktree.iter().min().unwrap();

        let prev_rank = if rank > 0 { rank - 1 } else { size - 1 };
        let next_rank = if rank + 1 < size { rank + 1 } else { 0 };

        if rank > 0 {
            let msg: Points = points.iter().filter(|&p| p.key < *min).cloned().collect();

            let msg_size: Rank = msg.len() as Rank;
            world.process_at_rank(prev_rank).send(&msg_size);
            world.process_at_rank(prev_rank).send(&msg[..]);
        }

        if rank < (size - 1) {
            let mut bufsize = 0;
            world.process_at_rank(next_rank).receive_into(&mut bufsize);
            let mut buffer = vec![Point::default(); bufsize as usize];
            world
                .process_at_rank(next_rank)
                .receive_into(&mut buffer[..]);
            received_points.append(&mut buffer);
        }

        // Filter out local points that's been sent to partner
        received_points = points.iter().filter(|&p| p.key >= *min).cloned().collect();

        received_points.sort();
        received_points
    }
}

impl Tree for MultiNodeTree {
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
        &self.leaves
    }

    // Get all points, gets local keys in multi-node setting
    fn get_points(&self) -> &Points {
        &self.points
    }

    // Get domain, gets global domain in multi-node setting
    fn get_domain(&self) -> &Domain {
        &self.domain
    }

    // Get points associated with a tree node key
    fn map_point_to_key(&self, point: &Point) -> Option<&MortonKey> {
        self.points_to_leaves.get(point)
    }

    // Get points associated with a tree node key
    fn map_key_to_points(&self, key: &MortonKey) -> Option<&Points> {
        self.leaves_to_points.get(key)
    }
}
