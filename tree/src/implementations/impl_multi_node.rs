use std::collections::{HashMap, HashSet};

use mpi::{
    topology::{Rank, UserCommunicator},
    traits::*,
};

use hyksort::hyksort;
use solvers_traits::tree::Tree;

use crate::{
    constants::{DEEPEST_LEVEL, K, NCRIT, ROOT},
    implementations::{
        impl_morton::{complete_region, encode_anchor, point_to_anchor},
        impl_single_node::{assign_nodes_to_points, assign_points_to_nodes},
    },
    types::{
        domain::Domain,
        morton::{MortonKey, MortonKeys},
        multi_node::MultiNodeTree,
        point::{Point, PointType, Points},
    },
};

impl MultiNodeTree {
    /// Create a new MultiNodeTree from a set of distributed points which define a domain.
    pub fn new(
        points: &[[PointType; 3]],
        adaptive: bool,
        n_crit: Option<usize>,
        depth: Option<u64>,
        world: &UserCommunicator,
    ) -> MultiNodeTree {
        let domain = Domain::from_global_points(points, world);

        let n_crit = n_crit.unwrap_or(NCRIT);
        let depth = depth.unwrap_or(DEEPEST_LEVEL);

        if adaptive {
            let (keys, points, points_to_keys, keys_to_points) =
                MultiNodeTree::adaptive_tree(world, points, &domain, n_crit);

            MultiNodeTree {
                adaptive,
                points,
                keys,
                domain,
                points_to_keys,
                keys_to_points,
            }
        } else {
            let (keys, points, points_to_keys, keys_to_points) =
                MultiNodeTree::uniform_tree(world, points, &domain, &depth);

            MultiNodeTree {
                adaptive,
                points,
                keys,
                domain,
                points_to_keys,
                keys_to_points,
            }
        }
    }

    /// Complete a distributed block tree from the seed octants, algorithm 4 in [1] (parallel).
    fn complete_blocktree(
        seeds: &mut MortonKeys,
        &rank: &Rank,
        &size: &Rank,
        world: &UserCommunicator,
    ) -> MortonKeys {
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

    /// Split tree nodes (blocks) by counting how many particles they contain.
    fn split_blocks(points: &Points, mut blocktree: MortonKeys, n_crit: usize) -> MortonKeys {
        let split_blocktree;
        let mut blocks_to_points;

        loop {
            let mut new_blocktree = MortonKeys::new();

            // Map between blocks and the leaves they contain
            blocks_to_points = assign_nodes_to_points(&blocktree, points);

            // Generate a new blocktree with a block's children if they violate the NCRIT constraint
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

            if check == blocks_to_points.len() {
                split_blocktree = new_blocktree;
                break;
            } else {
                blocktree = new_blocktree;
            }
        }

        split_blocktree
    }

    /// Find the seeds, defined as coarsest leaf/leaves, at each processor [1].
    fn find_seeds(leaves: &MortonKeys) -> MortonKeys {
        let min: MortonKey = *leaves.iter().min().unwrap();
        let max: MortonKey = *leaves.iter().max().unwrap();

        // Complete the region between the least and greatest leaves.
        let mut complete = complete_region(&min, &max);
        complete.push(min);
        complete.push(max);

        // Find seeds by filtering for leaves at coarsest level
        let coarsest_level = complete.iter().map(|k| k.level()).min().unwrap();
        let mut seeds: MortonKeys = MortonKeys {
            keys: complete
                .into_iter()
                .filter(|k| k.level() == coarsest_level)
                .collect(),
            index: 0,
        };

        seeds.sort();
        seeds
    }

    // Transfer points based on the coarse distributed blocktree.
    fn transfer_points_to_blocktree(
        world: &UserCommunicator,
        points: &[Point],
        seeds: &[MortonKey],
        &rank: &Rank,
        &size: &Rank,
    ) -> Points {
        let mut received_points: Points = Vec::new();

        let min_seed = if rank == 0 {
            points.iter().min().unwrap().key
        } else {
            *seeds.iter().min().unwrap()
        };

        let prev_rank = if rank > 0 { rank - 1 } else { size - 1 };
        let next_rank = if rank + 1 < size { rank + 1 } else { 0 };

        if rank > 0 {
            let msg: Points = points
                .iter()
                .filter(|&p| p.key < min_seed)
                .cloned()
                .collect();

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
        let mut points: Points = points
            .iter()
            .filter(|&p| p.key >= min_seed)
            .cloned()
            .collect();

        received_points.append(&mut points);
        received_points.sort();

        received_points
    }

    /// Specialization for uniform trees
    pub fn uniform_tree(
        world: &UserCommunicator,
        points: &[[PointType; 3]],
        domain: &Domain,
        depth: &u64,
    ) -> (
        MortonKeys,
        Points,
        HashMap<Point, MortonKey>,
        HashMap<MortonKey, Points>,
    ) {
        let mut points: Points = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let anchor = point_to_anchor(p, *depth, &domain).unwrap();
                let key = MortonKey {
                    morton: encode_anchor(&anchor, *depth),
                    anchor,
                };

                Point {
                    coordinate: *p,
                    global_idx: i,
                    key,
                }
            })
            .collect();

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();
        hyksort(&mut points, K, comm);

        // 2.ii Find unique leaf keys on each processor
        let mut keys = MortonKeys {
            keys: points.iter().map(|p| p.key).collect(),
            index: 0,
        };
        keys.linearize();

        // 3. Create bi-directional maps between keys and points
        let points_to_keys = assign_points_to_nodes(&points, &keys);
        let keys_to_points = assign_nodes_to_points(&keys, &points);

        // keys.sort();
        (keys, points, points_to_keys, keys_to_points)
    }

    /// Specialization for adaptive tree.
    pub fn adaptive_tree(
        world: &UserCommunicator,
        points: &[[PointType; 3]],
        domain: &Domain,
        n_crit: usize,
    ) -> (
        MortonKeys,
        Points,
        HashMap<Point, MortonKey>,
        HashMap<MortonKey, Points>,
    ) {
        let rank = world.rank();
        let size = world.size();

        // 1. Encode Points to Leaf Morton Keys, add a global index related to the processor
        let mut points: Points = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                global_idx: i,
                key: MortonKey::from_point(p, domain),
            })
            .collect();

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();

        hyksort(&mut points, K, comm);

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
        let mut seeds = MultiNodeTree::find_seeds(&local);

        let block_tree = MultiNodeTree::complete_blocktree(&mut seeds, &rank, &size, world);

        // 5.ii any data below the min seed sent to partner process
        let points =
            MultiNodeTree::transfer_points_to_blocktree(world, &points, &seeds, &rank, &size);

        // 6. Split blocks based on ncrit constraint
        let mut locally_balanced = MultiNodeTree::split_blocks(&points, block_tree, n_crit);

        // 7. Create a minimal balanced octree for local octants spanning their domain and linearize
        locally_balanced.balance();

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

        hyksort(&mut points, K, comm);

        let mut globally_balanced = MortonKeys {
            keys: points.iter().map(|p| p.key).collect(),
            index: 0,
        };
        globally_balanced.linearize();

        // 10. Find final bidirectional maps to non-overlapping tree
        let points_to_globally_balanced = assign_points_to_nodes(&points, &globally_balanced);
        let globally_balanced_to_points = assign_nodes_to_points(&globally_balanced, &points);
        let mut points: Points = points
            .iter()
            .map(|p| Point {
                coordinate: p.coordinate,
                global_idx: p.global_idx,
                key: *points_to_globally_balanced.get(p).unwrap(),
            })
            .collect();

        (
            globally_balanced,
            points,
            points_to_globally_balanced,
            globally_balanced_to_points,
        )
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

    // Get points associated with a tree node key
    fn map_point_to_key(&self, point: &Point) -> Option<&MortonKey> {
        self.points_to_keys.get(point)
    }

    // Get points associated with a tree node key
    fn map_key_to_points(&self, key: &MortonKey) -> Option<&Points> {
        self.keys_to_points.get(key)
    }
}
