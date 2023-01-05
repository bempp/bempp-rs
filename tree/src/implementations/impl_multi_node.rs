use itertools::Itertools;
use std::collections::{HashMap, HashSet};

use mpi::{
    collective::SystemOperation,
    request::{WaitGuard},
    topology::{Rank, UserCommunicator},
    traits::*,
};

use hyksort::hyksort;
use solvers_traits::tree::{LocallyEssentialTree, Tree};

use crate::{
    constants::{DEEPEST_LEVEL, K, LEVEL_SIZE, NCRIT, ROOT},
    implementations::{
        impl_morton::{complete_region, encode_anchor, point_to_anchor},
        impl_single_node::{
            assign_nodes_to_points, assign_points_to_nodes, find_seeds, split_blocks,
        },
        mpi_helpers::all_to_allv_sparse
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
        let points_to_leaves = assign_points_to_nodes(&points, &leaves);
        let leaves_to_points = assign_nodes_to_points(&leaves, &points);

        // Only retain keys that contain points
        leaves = MortonKeys {
            keys: leaves_to_points.keys().cloned().collect(),
            index: 0,
        };

        let leaves_set: HashSet<MortonKey> = leaves.iter().cloned().collect();

        let mut keys_set: HashSet<MortonKey> = HashSet::new();
        for key in leaves.iter() {
            let ancestors = key.ancestors();
            keys_set.extend(&ancestors);
        }

        let min = leaves.iter().min().unwrap();
        let max = leaves.iter().max().unwrap();
        let range = [world.rank() as KeyType, min.morton, max.morton];

        MultiNodeTree {
            world: world.duplicate(),
            adaptive: false,
            points,
            keys_set,
            leaves,
            leaves_set,
            domain: *domain,
            points_to_leaves,
            leaves_to_points,
            range,
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
        let mut keys_set: HashSet<MortonKey> = HashSet::new();
        for key in globally_balanced.iter() {
            let ancestors = key.ancestors();
            keys_set.extend(&ancestors);
        }

        let min = globally_balanced.iter().min().unwrap();
        let max = globally_balanced.iter().max().unwrap();
        let range = [world.rank() as KeyType, min.morton, max.morton];

        MultiNodeTree {
            world: world.duplicate(),
            adaptive: true,
            points,
            keys_set,
            leaves: globally_balanced,
            leaves_set,
            domain: *domain,
            points_to_leaves: points_to_globally_balanced,
            leaves_to_points: globally_balanced_to_points,
            range,
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
    type NodeIndicesSet = HashSet<MortonKey>;

    // Get adaptivity information
    fn get_adaptive(&self) -> bool {
        self.adaptive
    }

    // Get all keys, gets local keys in multi-node setting
    fn get_keys(&self) -> &MortonKeys {
        &self.leaves
    }

    fn get_keys_set(&self) -> &HashSet<MortonKey> {
        &self.leaves_set
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

impl LocallyEssentialTree for MultiNodeTree {
    type RawTree = MultiNodeTree;
    type NodeIndex = MortonKey;
    type NodeIndices = MortonKeys;

    fn create_let(&mut self) {
        // Communicate ranges globally using AllGather
        let rank = self.world.rank();
        let size = self.world.size();

        let mut ranges = vec![0 as KeyType; (size as usize) * 3];

        self.world.all_gather_into(&self.range, &mut ranges);

        let mut let_vec: Vec<MortonKey> = self.keys_set.iter().cloned().collect();

        // Calculate users and contributors for each leaf
        let mut users: Vec<Vec<KeyType>> = Vec::new();
        for key in let_vec.iter() {
            // Calculate the contributor processors for this octant
            let mut user_tmp: Vec<KeyType> = Vec::new();

            for chunk in ranges.chunks_exact(3) {
                let rank = chunk[0];
                let min = MortonKey::from_morton(chunk[1]);
                let max = MortonKey::from_morton(chunk[2]);

                // Check if ranges overlap of the neighbors of the key's parent, if so
                // add to user list
                if key.level() > 1 {
                    let mut colleagues_parent: Vec<MortonKey> = key.parent().neighbors();
                    let mut cp_min = colleagues_parent.iter().min();
                    let mut cp_max = colleagues_parent.iter().max();

                    match (cp_min, cp_max) {
                        (Some(cp_min), Some(cp_max)) => {
                            if (cp_min >= &min) && (cp_max <= &max)
                                || (cp_min <= &min) && (cp_max >= &min) && (cp_max <= &max)
                                || (cp_min >= &min) && (cp_min <= &max) && (cp_max >= &max)
                                || (cp_min <= &min) && (cp_max >= &max)
                            {
                                user_tmp.push(rank)
                            }
                        }
                        _ => (),
                    }
                } else if key.level() <= 1 {
                    user_tmp.push(rank)
                }
            }
            users.push(user_tmp);
        }

        let let_set: HashSet<MortonKey> = let_vec.iter().cloned().collect();

        // Send required nodes to user of each node
        for partner_rank in (0..self.world.size() as u64) {
            if partner_rank != self.world.rank().try_into().unwrap() {
                let key_packet: Vec<MortonKey> = let_vec
                    .iter()
                    .zip(users.iter())
                    .filter(|(_, user)| user.contains(&partner_rank))
                    .map(|(k, _)| *k)
                    .collect();

                let leaf_packet: Vec<MortonKey> =
                    let_set.intersection(&self.leaves_set).cloned().collect();

                let points_packet: Vec<Point> = self
                    .points
                    .iter()
                    .filter(|p| leaf_packet.contains(&p.key))
                    .cloned()
                    .collect();

                let partner_process = self.world.process_at_rank(partner_rank as i32);

                //////////////////
                let key_packet_size = key_packet.len() as u64;
                let mut received_key_packet_size: u64 = 0;

                mpi::point_to_point::send_receive_into(
                    &key_packet_size,
                    &partner_process,
                    &mut received_key_packet_size,
                    &partner_process,
                );

                let mut received_key_packet =
                    vec![MortonKey::default(); received_key_packet_size as usize];

                mpi::point_to_point::send_receive_into(
                    &key_packet[..],
                    &partner_process,
                    &mut received_key_packet,
                    &partner_process,
                );

                mpi::point_to_point::send_receive_into(
                    &key_packet_size,
                    &partner_process,
                    &mut received_key_packet_size,
                    &partner_process,
                );

                //////////////////
                let leaf_packet_size = leaf_packet.len() as u64;
                let mut received_leaf_packet_size: u64 = 0;

                mpi::point_to_point::send_receive_into(
                    &leaf_packet_size,
                    &partner_process,
                    &mut received_leaf_packet_size,
                    &partner_process,
                );

                let mut received_leaf_packet =
                    vec![MortonKey::default(); received_leaf_packet_size as usize];

                mpi::point_to_point::send_receive_into(
                    &leaf_packet[..],
                    &partner_process,
                    &mut received_leaf_packet,
                    &partner_process,
                );
                //////////////////
                let points_packet_size = points_packet.len() as u64;
                let mut received_points_packet_size: u64 = 0;

                mpi::point_to_point::send_receive_into(
                    &points_packet_size,
                    &partner_process,
                    &mut received_points_packet_size,
                    &partner_process,
                );

                let mut received_points_packet =
                    vec![Point::default(); received_points_packet_size as usize];

                mpi::point_to_point::send_receive_into(
                    &points_packet[..],
                    &partner_process,
                    &mut received_points_packet,
                    &partner_process,
                );

                // Insert into local tree
                self.keys_set.extend(&received_leaf_packet);
                self.leaves.extend(&received_leaf_packet);
                self.leaves_set = self.leaves.iter().cloned().collect();
                self.points.extend(&received_points_packet);
                self.points_to_leaves = assign_points_to_nodes(&self.points, &self.leaves);
                self.leaves_to_points = assign_nodes_to_points(&self.leaves, &self.points);
            }
        }
    }

    fn load_balance_let(&mut self) {
        // Repartition based on size of interaction lists for each leaf
        // Use Algorithm 1 in Sundar et. al (2008)

        let size = self.world.size();
        let rank = self.world.rank();

        let weights: Vec<i32> = self
            .leaves
            .iter()
            .map(|l| {
                (self.get_near_field(l).len() + self.get_x_list(l).len() + self.get_w_list(l).len())
                    as i32
            })
            .collect();

        let mut cum_weights: Vec<i32> = weights
            .iter()
            .scan(0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        let mut total_weight = 0i32;
        let local_weight = cum_weights.last().unwrap().clone();
        self.world
            .scan_into(&local_weight, &mut total_weight, &SystemOperation::sum());

        let mut prev_local_weight = 0i32;

        if rank < size - 1 {
            let next_rank = rank + 1;
            let next_process = self.world.process_at_rank(next_rank);
            next_process.send(&total_weight);
        }

        if rank > 0 {
            let prev_rank = rank - 1;
            let prev_process = self.world.process_at_rank(prev_rank);
            prev_process.receive_into(&mut prev_local_weight);
        }

        cum_weights = cum_weights.iter().map(|x| x + prev_local_weight).collect();

        if rank == size - 1 {
            total_weight = cum_weights.last().unwrap().clone();
        }
        self.world
            .process_at_rank(size - 1)
            .broadcast_into(&mut total_weight);

        let mean_weight: f32 = total_weight as f32 / self.world.size() as f32;

        let k = total_weight % size;

        let mut packets: Vec<Vec<MortonKey>> = Vec::new();
        let mut removed_indices: HashSet<usize> = HashSet::new();
        let mut packet_destinations = vec![0i32; size as usize];

        for p in 1..=size {
            let mut packet: Vec<MortonKey> = Vec::new();
            if p <= k {
                for (i, &cw) in cum_weights.iter().enumerate() {
                    if ((p - 1) as f32) * (mean_weight + 1.0) <= (cw as f32)
                        && (cw as f32) < (p as f32) * (mean_weight + 1.0)
                    {
                        packet.push(self.leaves[i]);
                        removed_indices.insert(i as usize);
                    }
                }
            } else {
                for (i, &cw) in cum_weights.iter().enumerate() {
                    if ((p - 1) as f32) * (mean_weight) + (k as f32) <= (cw as f32)
                        && (cw as f32) < ((p) as f32) * (mean_weight) + (k as f32)
                    {
                        packet.push(self.leaves[i]);
                        removed_indices.insert(i as usize);
                    }
                }
            }

            if rank != (p - 1) && packet.len() > 0 {
                packet_destinations[(p - 1) as usize] = 1;
                packets.push(packet);
            }
        }

        // Communicate number of packets being received by each process globally
        let mut to_receive = vec![0i32; size as usize];
        self.world.all_reduce_into(
            &packet_destinations,
            &mut to_receive,
            SystemOperation::sum(),
        );

        let recv_count = to_receive[rank as usize];
        
        packet_destinations = packet_destinations
            .into_iter()
            .enumerate()
            .filter(|(i, x)| x > &0)
            .map(|(i, _)| i as i32)
            .collect();

        // Remove all data being sent
        self.leaves_set = self
            .leaves_set
            .difference(&packets.iter().flatten().cloned().collect())
            .cloned()
            .collect();

        self.leaves = self.leaves_set.iter().cloned().collect();

        self.keys_set = self
            .keys_set
            .difference(&packets.iter().flatten().cloned().collect())
            .cloned()
            .collect();

        let received_leaves = all_to_allv_sparse(&self.world, packets, packet_destinations, recv_count);
        
        // Insert into local leaves and keys
        self.leaves.extend(&received_leaves);
        self.leaves_set = self.leaves.iter().cloned().collect();
        self.keys_set.extend(&received_leaves);

        // self.points.extend(&received_points_packet);
        // self.points_to_leaves = assign_points_to_nodes(&self.points, &self.leaves);
        // self.leaves_to_points = assign_nodes_to_points(&self.leaves, &self.points);
    }

    // Calculate near field interaction list of  keys.
    fn get_near_field(&self, leaf: &MortonKey) -> MortonKeys {
        let mut result = Vec::<MortonKey>::new();
        let neighbours = leaf.neighbors();

        // Child level
        let mut neighbors_children_adj: Vec<MortonKey> = neighbours
            .iter()
            .flat_map(|n| n.children())
            .filter(|nc| self.keys_set.contains(nc) && leaf.is_adjacent(nc))
            .collect();

        // Key level
        let mut neighbors_adj: Vec<MortonKey> = neighbours
            .iter()
            .filter(|n| self.keys_set.contains(n) && leaf.is_adjacent(n))
            .cloned()
            .collect();

        // Parent level
        let mut neighbors_parents_adj: Vec<MortonKey> = neighbours
            .iter()
            .map(|n| n.parent())
            .filter(|np| self.leaves_set.contains(np) && leaf.is_adjacent(np))
            .collect();

        result.append(&mut neighbors_children_adj);
        result.append(&mut neighbors_adj);
        result.append(&mut neighbors_parents_adj);

        MortonKeys {
            keys: result,
            index: 0,
        }
    }

    // Calculate compressible far field interactions of leaf & other keys.
    fn get_interaction_list(&self, key: &MortonKey) -> Option<MortonKeys> {
        if key.level() >= 2 {
            return Some(MortonKeys {
                keys: key
                    .parent()
                    .neighbors()
                    .iter()
                    .flat_map(|pn| pn.children())
                    .filter(|pnc| self.keys_set.contains(pnc) && key.is_adjacent(pnc))
                    .collect_vec(),
                index: 0,
            });
        }
        {
            None
        }
    }

    // Calculate M2P interactions of leaf key.
    fn get_w_list(&self, leaf: &MortonKey) -> MortonKeys {
        // Child level
        MortonKeys {
            keys: leaf
                .neighbors()
                .iter()
                .flat_map(|n| n.children())
                .filter(|nc| self.keys_set.contains(nc) && !leaf.is_adjacent(nc))
                .collect_vec(),
            index: 0,
        }
    }

    // Calculate P2L interactions of leaf key.
    fn get_x_list(&self, leaf: &MortonKey) -> MortonKeys {
        MortonKeys {
            keys: leaf
                .parent()
                .neighbors()
                .into_iter()
                .filter(|pn| self.keys_set.contains(pn) && !leaf.is_adjacent(pn))
                .collect_vec(),
            index: 0,
        }
    }
}
