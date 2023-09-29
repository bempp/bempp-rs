use itertools::Itertools;
use std::collections::{HashMap, HashSet};

use mpi::{topology::UserCommunicator, traits::*, Rank};

use hyksort::hyksort;

use bempp_traits::tree::Tree;

use crate::{
    constants::{DEEPEST_LEVEL, NCRIT, ROOT},
    implementations::impl_morton::{complete_region, encode_anchor},
    types::{
        domain::Domain,
        morton::{KeyType, MortonKey, MortonKeys},
        multi_node::MultiNodeTree,
        point::{Point, PointType, Points},
        single_node::SingleNodeTree,
    },
};

impl MultiNodeTree {
    /// Constructor for uniform trees
    ///
    /// # Arguments
    /// * `world` -
    pub fn uniform_tree(
        world: &UserCommunicator,
        k: i32,
        points: &[PointType],
        domain: &Domain,
        depth: u64,
        global_idxs: &[usize],
    ) -> MultiNodeTree {
        // Encode points at deepest level, and map to specified depth.
        let dim = 3;
        let npoints = points.len() / dim;

        let mut tmp = Points::default();
        for i in 0..npoints {
            let point = [points[i], points[i + npoints], points[i + 2 * npoints]];
            let base_key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::from_point(&point, &domain, depth);
            tmp.points.push(Point {
                coordinate: point,
                base_key,
                encoded_key,
                global_idx: global_idxs[i],
            })
        }
        let mut points = tmp;

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();
        hyksort(&mut points.points, k, comm);

        // 2.ii Find leaf keys on each processor
        let min = points.points.iter().min().unwrap().encoded_key;
        let max = points.points.iter().max().unwrap().encoded_key;

        let diameter = 1 << (DEEPEST_LEVEL - depth as u64);

        // Find leaves within ths processor's range
        let leaves = MortonKeys {
            keys: (min.anchor[0]..max.anchor[0])
                .step_by(diameter)
                .flat_map(|i| {
                    (min.anchor[1]..max.anchor[1])
                        .step_by(diameter)
                        .map(move |j| (i, j))
                })
                .flat_map(|(i, j)| {
                    (min.anchor[2]..max.anchor[2])
                        .step_by(diameter)
                        .map(move |k| [i, j, k])
                })
                .map(|anchor| {
                    let morton = encode_anchor(&anchor, depth);
                    MortonKey { anchor, morton }
                })
                .collect(),
            index: 0,
        };

        // 3. Assign keys to points
        let unmapped = SingleNodeTree::assign_nodes_to_points(&leaves, &mut points);

        // Group points by leaves
        points.sort();

        let mut leaves_to_points = HashMap::new();
        let mut curr = points.points[0].clone();
        let mut curr_idx = 0;

        for (i, point) in points.points.iter().enumerate() {
            if point.encoded_key != curr.encoded_key {
                leaves_to_points.insert(curr.encoded_key, (curr_idx, i));
                curr_idx = i;
                curr = point.clone();
            }
        }
        leaves_to_points.insert(curr.encoded_key, (curr_idx, points.points.len()));

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

        let min = leaves.iter().min().unwrap();
        let max = leaves.iter().max().unwrap();
        let range = [world.rank() as KeyType, min.morton, max.morton];

        // Group by level to perform efficient lookup of nodes
        keys.sort_by_key(|a| a.level());

        let mut levels_to_keys = HashMap::new();
        let mut curr = keys[0];
        let mut curr_idx = 0;
        for (i, key) in keys.iter().enumerate() {
            if key.level() != curr.level() {
                levels_to_keys.insert(curr.level(), (curr_idx, i));
                curr_idx = i;
                curr = *key;
            }
        }
        levels_to_keys.insert(curr.level(), (curr_idx, keys.len()));

        // Return tree in sorted order
        for l in 0..=depth {
            let &(l, r) = levels_to_keys.get(&l).unwrap();
            let subset = &mut keys[l..r];
            subset.sort();
        }

        MultiNodeTree {
            world: world.duplicate(),
            depth,
            domain: *domain,
            points,
            leaves,
            keys,
            leaves_to_points,
            levels_to_keys,
            leaves_set,
            keys_set,
            range,
        }
    }

    /// Constructor for adaptive tree.
    pub fn adaptive_tree(
        world: &UserCommunicator,
        k: i32,
        points: &[PointType],
        domain: &Domain,
        n_crit: u64,
        global_idxs: &[usize],
    ) -> MultiNodeTree {
        // 1. Encode Points to Leaf Morton Keys, add a global index related to the processor
        let dim = 3;
        let npoints = points.len() / dim;

        let mut tmp = Points::default();
        for i in 0..npoints {
            let point = [points[i], points[i + npoints], points[i + 2 * npoints]];
            let key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);
            tmp.points.push(Point {
                coordinate: point,
                base_key: key,
                encoded_key: key,
                global_idx: global_idxs[i],
            })
        }
        let mut points = tmp;

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();

        hyksort(&mut points.points, k, comm);

        // 2.ii Find unique leaf keys on each processor
        let mut local = MortonKeys {
            keys: points.points.iter().map(|p| p.encoded_key).collect(),
            index: 0,
        };

        // 3. Linearise received keys (remove overlaps if they exist).
        local.linearize();

        // 4. Complete region spanned by node.
        local.complete();

        // 5.i Find seeds and compute the coarse blocktree
        let mut seeds = SingleNodeTree::find_seeds(&local);

        let blocktree = MultiNodeTree::complete_blocktree(world, &mut seeds);

        // 5.ii any data below the min seed sent to partner process
        let mut points =
            MultiNodeTree::transfer_points_to_blocktree(world, &points.points[..], &blocktree);

        // 6. Split blocks based on ncrit constraint
        let mut locally_balanced =
            SingleNodeTree::split_blocks(&mut points, blocktree, n_crit as usize);

        // 7. Create a minimal balanced octree for local octants spanning their domain and linearize
        locally_balanced.sort();
        locally_balanced.balance();
        locally_balanced.linearize();

        // // 8. Find new maps between points and locally balanced tree
        let _unmapped = SingleNodeTree::assign_nodes_to_points(&locally_balanced, &mut points);

        // 9. Perform another distributed sort and remove overlaps locally
        let comm = world.duplicate();

        hyksort(&mut points.points, k, comm);

        let mut globally_balanced = MortonKeys {
            keys: points.points.iter().map(|p| p.encoded_key).collect(),
            index: 0,
        };
        globally_balanced.linearize();

        // Group points by leaves
        points.sort();

        let mut leaves_to_points = HashMap::new();
        let mut curr = points.points[0].clone();
        let mut curr_idx = 0;

        for (i, point) in points.points.iter().enumerate() {
            if point.encoded_key != curr.encoded_key {
                leaves_to_points.insert(curr.encoded_key, (curr_idx, i));
                curr_idx = i;
                curr = point.clone();
            }
        }
        leaves_to_points.insert(curr.encoded_key, (curr_idx, points.points.len()));

        // 10. Find final maps to non-overlapping tree
        let unmapped = SingleNodeTree::assign_nodes_to_points(&globally_balanced, &mut points);

        // Add unmapped leaves
        let globally_balanced = MortonKeys {
            keys: leaves_to_points
                .keys()
                .cloned()
                .chain(unmapped.iter().copied())
                .collect_vec(),
            index: 0,
        };

        // Find all keys in tree
        let tmp: HashSet<MortonKey> = globally_balanced
            .iter()
            .flat_map(|leaf| leaf.ancestors().into_iter())
            .collect();

        let mut keys = MortonKeys {
            keys: tmp.into_iter().collect_vec(),
            index: 0,
        };

        let leaves_set: HashSet<MortonKey> = globally_balanced.iter().cloned().collect();
        let mut keys_set: HashSet<MortonKey> = HashSet::new();

        // Create maps between points and globally balanced tree
        let mut leaves_to_points = HashMap::new();
        let mut curr = points.points[0].clone();
        let mut curr_idx = 0;

        for (i, point) in points.points.iter().enumerate() {
            if point.encoded_key != curr.encoded_key {
                leaves_to_points.insert(curr.encoded_key, (curr_idx, i));
                curr_idx = i;
                curr = point.clone();
            }
        }
        leaves_to_points.insert(curr.encoded_key, (curr_idx, points.points.len()));

        for key in globally_balanced.iter() {
            let ancestors = key.ancestors();
            keys_set.extend(&ancestors);
        }

        let min = globally_balanced.iter().min().unwrap();
        let max = globally_balanced.iter().max().unwrap();
        let range = [world.rank() as KeyType, min.morton, max.morton];

        // Group by level to perform efficient lookup of nodes
        keys.sort_by_key(|a| a.level());

        let mut levels_to_keys = HashMap::new();
        let mut curr = keys[0];
        let mut curr_idx = 0;
        for (i, key) in keys.iter().enumerate() {
            if key.level() != curr.level() {
                levels_to_keys.insert(curr.level(), (curr_idx, i));
                curr_idx = i;
                curr = *key;
            }
        }
        levels_to_keys.insert(curr.level(), (curr_idx, keys.len()));

        // Find depth
        let depth = points
            .points
            .iter()
            .map(|p| p.encoded_key.level())
            .max()
            .unwrap();

        // Return tree in sorted order
        for l in 0..=depth {
            let &(l, r) = levels_to_keys.get(&l).unwrap();
            let subset = &mut keys[l..r];
            subset.sort();
        }

        MultiNodeTree {
            world: world.duplicate(),
            depth,
            domain: *domain,
            points,
            leaves: globally_balanced,
            keys,
            leaves_to_points,
            levels_to_keys,
            leaves_set,
            keys_set,
            range,
        }
    }

    /// Create a new multi-node tree. If non-adaptive (uniform) trees are created, they are specified
    /// by a user defined maximum depth, if an adaptive tree is created it is specified by only by the
    /// user defined maximum leaf maximum occupancy n_crit.
    pub fn new(
        world: &UserCommunicator,
        points: &[PointType],
        adaptive: bool,
        n_crit: Option<u64>,
        depth: Option<u64>,
        k: i32,
        global_idxs: &[usize],
    ) -> MultiNodeTree {
        // TODO: Come back and reconcile a runtime point dimension detector

        let domain = Domain::from_global_points(points, &world);

        let n_crit = n_crit.unwrap_or(NCRIT);
        let depth = depth.unwrap_or(DEEPEST_LEVEL);

        if adaptive {
            MultiNodeTree::adaptive_tree(world, k, points, &domain, n_crit, global_idxs)
        } else {
            MultiNodeTree::uniform_tree(world, k, points, &domain, depth, &global_idxs)
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
            // Check for overlap
            if first_child < *min {
                seeds.push(first_child)
            }
            seeds.sort();
        }

        if rank == (size - 1) {
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

        let mut received_points = Vec::new();

        let min = blocktree.iter().min().unwrap();

        let prev_rank = if rank > 0 { rank - 1 } else { size - 1 };
        let next_rank = if rank + 1 < size { rank + 1 } else { 0 };

        if rank > 0 {
            let msg: Vec<_> = points
                .iter()
                .filter(|&p| p.encoded_key < *min)
                .cloned()
                .collect_vec();

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
        received_points = points
            .iter()
            .filter(|&p| p.encoded_key >= *min)
            .cloned()
            .collect();

        received_points.sort();

        Points {
            points: received_points,
            index: 0,
        }
    }
}

impl Tree for MultiNodeTree {
    type Domain = Domain;
    type NodeIndex = MortonKey;
    type NodeIndexSlice<'a> = &'a [MortonKey];
    type NodeIndices = MortonKeys;
    type Point = Point;
    type PointSlice<'a> = &'a [Point];
    type PointData = f64;
    type PointDataSlice<'a> = &'a [f64];
    type GlobalIndex = usize;
    type GlobalIndexSlice<'a> = &'a [usize];

    fn get_depth(&self) -> u64 {
        self.depth
    }

    fn get_domain(&self) -> &'_ Self::Domain {
        &self.domain
    }

    fn get_keys(&self, level: u64) -> Option<Self::NodeIndexSlice<'_>> {
        if let Some(&(l, r)) = self.levels_to_keys.get(&level) {
            Some(&self.keys[l..r])
        } else {
            None
        }
    }

    fn get_all_keys(&self) -> Option<Self::NodeIndexSlice<'_>> {
        Some(&self.keys)
    }

    fn get_all_keys_set(&self) -> &'_ HashSet<Self::NodeIndex> {
        &self.keys_set
    }

    fn get_all_leaves_set(&self) -> &'_ HashSet<Self::NodeIndex> {
        &self.leaves_set
    }

    fn get_leaves(&self) -> Option<Self::NodeIndexSlice<'_>> {
        Some(&self.leaves)
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

// // Helper function for creating locally essential trees.
// fn let_helper(tree: &mut MultiNodeTree) {
//     // Communicate ranges globally using AllGather
//     let rank = tree.world.rank();
//     let size = tree.world.size();

//     let mut ranges = vec![0 as KeyType; (size as usize) * 3];

//     tree.world.all_gather_into(&tree.range, &mut ranges);

//     let let_vec: Vec<MortonKey> = tree.keys_set.iter().cloned().collect();

//     // Calculate users for each key in LET
//     let mut users: Vec<Vec<Rank>> = Vec::new();
//     let mut key_packet_destinations = vec![0 as Rank; size as usize];
//     let mut leaf_packet_destinations = vec![0 as Rank; size as usize];

//     for key in let_vec.iter() {
//         let mut user_tmp: Vec<Rank> = Vec::new();

//         // Loop over all ranges, each key may be used by multiple processes
//         for chunk in ranges.chunks_exact(3) {
//             let rank = chunk[0] as Rank;
//             let min = MortonKey::from_morton(chunk[1]);
//             let max = MortonKey::from_morton(chunk[2]);

//             // Check if ranges overlap with the neighbors of the key's parent
//             if rank != tree.world.rank() {
//                 if key.level() > 1 {
//                     let colleagues_parent: Vec<MortonKey> = key.parent().neighbors();
//                     let (cp_min, cp_max) = (
//                         colleagues_parent.iter().min(),
//                         colleagues_parent.iter().max(),
//                     );

//                     match (cp_min, cp_max) {
//                         (Some(cp_min), Some(cp_max)) => {
//                             if (cp_min >= &min) && (cp_max <= &max)
//                                 || (cp_min <= &min) && (cp_max >= &min) && (cp_max <= &max)
//                                 || (cp_min >= &min) && (cp_min <= &max) && (cp_max >= &max)
//                                 || (cp_min <= &min) && (cp_max >= &max)
//                             {
//                                 user_tmp.push(rank);

//                                 // Mark ranks as users of keys/leaves from this process
//                                 if key_packet_destinations[rank as usize] == 0 {
//                                     key_packet_destinations[rank as usize] = 1
//                                 }

//                                 if leaf_packet_destinations[rank as usize] == 0
//                                     && tree.leaves_set.contains(key)
//                                 {
//                                     leaf_packet_destinations[rank as usize] = 1;
//                                 }
//                             }
//                         }
//                         _ => (),
//                     }
//                 }
//                 // If the key as at level one its parent is the root node, so by definition
//                 // it will always overlap with a range
//                 else if key.level() <= 1 {
//                     user_tmp.push(rank);
//                     if key_packet_destinations[rank as usize] == 0 {
//                         key_packet_destinations[rank as usize] = 1
//                     }

//                     if leaf_packet_destinations[rank as usize] == 0 && tree.leaves_set.contains(key)
//                     {
//                         leaf_packet_destinations[rank as usize] = 1;
//                     }
//                 }
//             }
//         }
//         users.push(user_tmp);
//     }

//     // Communicate number of packets being received by each process globally
//     let mut keys_to_receive = vec![0i32; size as usize];
//     tree.world.all_reduce_into(
//         &key_packet_destinations,
//         &mut keys_to_receive,
//         SystemOperation::sum(),
//     );

//     let mut leaves_to_receive = vec![0i32; size as usize];
//     tree.world.all_reduce_into(
//         &leaf_packet_destinations,
//         &mut leaves_to_receive,
//         SystemOperation::sum(),
//     );

//     // Calculate the number of receives that this process is expecting
//     let recv_count_keys = keys_to_receive[rank as usize];
//     let recv_count_leaves = leaves_to_receive[rank as usize];

//     // Filter for packet destinations
//     key_packet_destinations = key_packet_destinations
//         .into_iter()
//         .enumerate()
//         .filter(|(_, x)| x > &0)
//         .map(|(i, _)| i as Rank)
//         .collect();

//     leaf_packet_destinations = leaf_packet_destinations
//         .into_iter()
//         .enumerate()
//         .filter(|(_, x)| x > &0)
//         .map(|(i, _)| i as Rank)
//         .collect();

//     let mut key_packets: Vec<Vec<MortonKey>> = Vec::new();
//     let mut leaf_packets: Vec<Vec<MortonKey>> = Vec::new();
//     let mut point_packets: Vec<Vec<Point>> = Vec::new();

//     let mut key_packet_destinations_filt: Vec<Rank> = Vec::new();
//     let mut leaf_packet_destinations_filt: Vec<Rank> = Vec::new();

//     // Form packets for each send
//     for &rank in key_packet_destinations.iter() {
//         let key_packet: Vec<MortonKey> = let_vec
//             .iter()
//             .zip(users.iter())
//             .filter(|(_, user)| user.contains(&rank))
//             .map(|(k, _)| *k)
//             .collect();
//         let key_packet_set: HashSet<MortonKey> = key_packet.iter().cloned().collect();

//         if key_packet.len() > 0 {
//             key_packets.push(key_packet);
//             key_packet_destinations_filt.push(rank);
//         }

//         if leaf_packet_destinations.contains(&rank) {
//             let leaf_packet: Vec<MortonKey> = key_packet_set
//                 .intersection(&tree.leaves_set)
//                 .cloned()
//                 .collect();

//             let point_packet: Vec<Point> = leaf_packet
//                 .iter()
//                 .map(|leaf| tree.leaves_to_points.get(leaf).unwrap())
//                 .cloned()
//                 .flatten()
//                 .collect();

//             if leaf_packet.len() > 0 {
//                 leaf_packets.push(leaf_packet);
//                 point_packets.push(point_packet);
//                 leaf_packet_destinations_filt.push(rank);
//             }
//         }
//     }

//     // Communicate keys, leaves and points
//     let received_leaves = all_to_allv_sparse(
//         &tree.world,
//         &leaf_packets,
//         &leaf_packet_destinations_filt,
//         &recv_count_leaves,
//     );

//     let received_points = all_to_allv_sparse(
//         &tree.world,
//         &point_packets,
//         &leaf_packet_destinations_filt,
//         &recv_count_leaves,
//     );

//     let received_keys = all_to_allv_sparse(
//         &tree.world,
//         &key_packets,
//         &key_packet_destinations_filt,
//         &recv_count_keys,
//     );

//     // Insert received data into local tree
//     tree.keys_set.extend(&received_keys);
//     tree.leaves_set.extend(&received_leaves);
//     tree.leaves = tree.leaves_set.iter().cloned().collect();
//     tree.points.extend(&received_points);
//     tree.points_to_leaves = assign_points_to_nodes(&tree.points, &tree.leaves);
//     tree.leaves_to_points = assign_nodes_to_points(&tree.leaves, &tree.points);
// }

// // Repartition based on size of interaction lists for each leaf Use Algorithm 1
// // in Sundar et. al (2008)
// fn load_balance_let(tree: &mut MultiNodeTree) {
//     let size = tree.world.size();
//     let rank = tree.world.rank();

//     let weights: Vec<Count> = tree
//         .leaves
//         .iter()
//         .map(|l| {
//             let u = tree.get_near_field(l).unwrap_or(MortonKeys::default());
//             let w = tree.get_w_list(l).unwrap_or(MortonKeys::default());
//             let x = tree.get_x_list(l).unwrap_or(MortonKeys::default());
//             (u.len() + w.len() + x.len()) as i32
//         })
//         .collect();

//     let mut cum_weights: Vec<Count> = weights
//         .iter()
//         .scan(0, |acc, &x| {
//             *acc += x;
//             Some(*acc)
//         })
//         .collect();

//     let mut total_weight = 0 as Count;
//     let local_weight = cum_weights.last().unwrap().clone();
//     tree.world
//         .scan_into(&local_weight, &mut total_weight, &SystemOperation::sum());

//     let mut prev_local_weight = 0 as Count;

//     if rank < size - 1 {
//         let next_rank = rank + 1;
//         let next_process = tree.world.process_at_rank(next_rank);
//         next_process.send(&total_weight);
//     }

//     if rank > 0 {
//         let prev_rank = rank - 1;
//         let prev_process = tree.world.process_at_rank(prev_rank);
//         prev_process.receive_into(&mut prev_local_weight);
//     }

//     cum_weights = cum_weights.iter().map(|x| x + prev_local_weight).collect();

//     if rank == size - 1 {
//         total_weight = cum_weights.last().unwrap().clone();
//     }

//     tree.world
//         .process_at_rank(size - 1)
//         .broadcast_into(&mut total_weight);

//     let mean_weight: f32 = total_weight as f32 / tree.world.size() as f32;

//     let k = total_weight % size;

//     let mut leaves_packets: Vec<Vec<MortonKey>> = Vec::new();
//     let mut points_packets: Vec<Vec<Point>> = Vec::new();
//     let mut packet_destinations = vec![0 as Rank; size as usize];

//     for p in 1..=size {
//         let mut leaves_packet: Vec<MortonKey> = Vec::new();
//         let mut points_packet: Vec<Point> = Vec::new();

//         if p <= k {
//             for (i, &cw) in cum_weights.iter().enumerate() {
//                 if ((p - 1) as f32) * (mean_weight + 1.0) <= (cw as f32)
//                     && (cw as f32) < (p as f32) * (mean_weight + 1.0)
//                 {
//                     leaves_packet.push(tree.leaves[i]);
//                     points_packet.extend(tree.leaves_to_points.get(&tree.leaves[i]).unwrap())
//                 }
//             }
//         } else {
//             for (i, &cw) in cum_weights.iter().enumerate() {
//                 if ((p - 1) as f32) * (mean_weight) + (k as f32) <= (cw as f32)
//                     && (cw as f32) < ((p) as f32) * (mean_weight) + (k as f32)
//                 {
//                     leaves_packet.push(tree.leaves[i]);
//                     points_packet.extend(tree.leaves_to_points.get(&tree.leaves[i]).unwrap())
//                 }
//             }
//         }

//         if rank != (p - 1) && leaves_packet.len() > 0 {
//             packet_destinations[(p - 1) as usize] = 1;
//             leaves_packets.push(leaves_packet);
//             points_packets.push(points_packet);
//         }
//     }

//     // Communicate number of packets being received by each process globally
//     let mut to_receive = vec![0i32; size as usize];
//     tree.world.all_reduce_into(
//         &packet_destinations,
//         &mut to_receive,
//         SystemOperation::sum(),
//     );

//     let recv_count = to_receive[rank as usize];

//     // Filter for packet destinations
//     packet_destinations = packet_destinations
//         .into_iter()
//         .enumerate()
//         .filter(|(_, x)| x > &0)
//         .map(|(i, _)| i as Rank)
//         .collect();

//     // Remove all data being sent from local tree
//     tree.leaves_set = tree
//         .leaves_set
//         .difference(&leaves_packets.iter().flatten().cloned().collect())
//         .cloned()
//         .collect();

//     tree.leaves = tree.leaves_set.iter().cloned().collect();

//     tree.keys_set = tree
//         .keys_set
//         .difference(&leaves_packets.iter().flatten().cloned().collect())
//         .cloned()
//         .collect();

//     // This line works as the leaves being sent have already been removed
//     tree.points = tree
//         .leaves
//         .iter()
//         .map(|leaf| tree.leaves_to_points.get(leaf).unwrap())
//         .cloned()
//         .flatten()
//         .collect();

//     let received_leaves = all_to_allv_sparse(
//         &tree.world,
//         &leaves_packets,
//         &packet_destinations,
//         &recv_count,
//     );

//     let received_points = all_to_allv_sparse(
//         &tree.world,
//         &points_packets,
//         &packet_destinations,
//         &recv_count,
//     );

//     // Insert received data into local tree
//     tree.keys_set.extend(&received_leaves);
//     tree.leaves_set.extend(&received_leaves);
//     tree.leaves = tree.leaves_set.iter().cloned().collect();
//     tree.points.extend(&received_points);
//     tree.points_to_leaves = assign_points_to_nodes(&tree.points, &tree.leaves);
//     tree.leaves_to_points = assign_nodes_to_points(&tree.leaves, &tree.points);
// }
