//! Implementation of interaction lists for FMMs (single and multi node)
use itertools::Itertools;
use rlst_common::types::Scalar;

use bempp_traits::{field::SourceToTargetData, fmm::InteractionLists, kernel::Kernel, tree::Tree};
use bempp_tree::types::morton::{MortonKey, MortonKeys};
use num::Float;

// impl<T, U, V, W> InteractionLists for KiFmm<T, U, V, W>
// where
//     T: Tree<NodeIndex = MortonKey, NodeIndices = MortonKeys>,
//     U: Kernel<T = W>,
//     V: SourceToTargetData<U>,
//     W: Scalar + Float + Default,
// {
//     type Tree = T;

//     fn get_u_list(
//         &self,
//         key: &<Self::Tree as Tree>::NodeIndex,
//     ) -> Option<<Self::Tree as Tree>::NodeIndices> {
//         let mut u_list = Vec::<MortonKey>::new();
//         let neighbours = key.neighbors();

//         // Child level
//         let mut neighbors_children_adj: Vec<MortonKey> = neighbours
//             .iter()
//             .flat_map(|n| n.children())
//             .filter(|nc| {
//                 self.tree.get_all_leaves_set().unwrap().contains(nc) && key.is_adjacent(nc)
//             })
//             .collect();

//         // Key level
//         let mut neighbors_adj: Vec<MortonKey> = neighbours
//             .iter()
//             .filter(|n| self.tree.get_all_leaves_set().unwrap().contains(n) && key.is_adjacent(n))
//             .cloned()
//             .collect();

//         // Parent level
//         let mut parent_neighbours_adj: Vec<MortonKey> = key
//             .parent()
//             .neighbors()
//             .into_iter()
//             .filter(|pn| {
//                 self.tree.get_all_leaves_set().unwrap().contains(pn) && key.is_adjacent(pn)
//             })
//             .collect();

//         u_list.append(&mut neighbors_children_adj);
//         u_list.append(&mut neighbors_adj);
//         u_list.append(&mut parent_neighbours_adj);
//         u_list.push(*key);

//         if !u_list.is_empty() {
//             Some(MortonKeys {
//                 keys: u_list,
//                 index: 0,
//             })
//         } else {
//             None
//         }
//     }

//     fn get_v_list(
//         &self,
//         key: &<Self::Tree as Tree>::NodeIndex,
//     ) -> Option<<Self::Tree as Tree>::NodeIndices> {
//         if key.level() >= 2 {
//             let v_list = key
//                 .parent()
//                 .neighbors()
//                 .iter()
//                 .flat_map(|pn| pn.children())
//                 .filter(|pnc| {
//                     self.tree.get_all_keys_set().unwrap().contains(pnc) && !key.is_adjacent(pnc)
//                 })
//                 .collect_vec();

//             if !v_list.is_empty() {
//                 return Some(MortonKeys {
//                     keys: v_list,
//                     index: 0,
//                 });
//             } else {
//                 return None;
//             }
//         }
//         None
//     }

//     fn get_w_list(
//         &self,
//         leaf: &<Self::Tree as Tree>::NodeIndex,
//     ) -> Option<<Self::Tree as Tree>::NodeIndices> {
//         // Child level
//         let w_list = leaf
//             .neighbors()
//             .iter()
//             .flat_map(|n| n.children())
//             .filter(|nc| {
//                 self.tree.get_all_keys_set().unwrap().contains(nc) && !leaf.is_adjacent(nc)
//             })
//             .collect_vec();

//         if !w_list.is_empty() {
//             Some(MortonKeys {
//                 keys: w_list,
//                 index: 0,
//             })
//         } else {
//             None
//         }
//     }

//     fn get_x_list(
//         &self,
//         key: &<Self::Tree as Tree>::NodeIndex,
//     ) -> Option<<Self::Tree as Tree>::NodeIndices> {
//         let x_list = key
//             .parent()
//             .neighbors()
//             .into_iter()
//             .filter(|pn| self.tree.get_all_keys_set().unwrap().contains(pn) && !key.is_adjacent(pn))
//             .collect_vec();

//         if !x_list.is_empty() {
//             Some(MortonKeys {
//                 keys: x_list,
//                 index: 0,
//             })
//         } else {
//             None
//         }
//     }
// }

// impl<T, U, V, W> InteractionLists for KiFmmMatrix<T, U, V, W>
// where
//     T: Tree<NodeIndex = MortonKey, NodeIndices = MortonKeys>,
//     U: Kernel<T = W>,
//     V: SourceToTargetData<U>,
//     W: Scalar + Float + Default,
// {
//     type Tree = T;

//     fn get_u_list(
//         &self,
//         key: &<Self::Tree as Tree>::NodeIndex,
//     ) -> Option<<Self::Tree as Tree>::NodeIndices> {
//         let mut u_list = Vec::<MortonKey>::new();
//         let neighbours = key.neighbors();

//         // Child level
//         let mut neighbors_children_adj: Vec<MortonKey> = neighbours
//             .iter()
//             .flat_map(|n| n.children())
//             .filter(|nc| {
//                 self.tree.get_all_leaves_set().unwrap().contains(nc) && key.is_adjacent(nc)
//             })
//             .collect();

//         // Key level
//         let mut neighbors_adj: Vec<MortonKey> = neighbours
//             .iter()
//             .filter(|n| self.tree.get_all_leaves_set().unwrap().contains(n) && key.is_adjacent(n))
//             .cloned()
//             .collect();

//         // Parent level
//         let mut parent_neighbours_adj: Vec<MortonKey> = key
//             .parent()
//             .neighbors()
//             .into_iter()
//             .filter(|pn| {
//                 self.tree.get_all_leaves_set().unwrap().contains(pn) && key.is_adjacent(pn)
//             })
//             .collect();

//         u_list.append(&mut neighbors_children_adj);
//         u_list.append(&mut neighbors_adj);
//         u_list.append(&mut parent_neighbours_adj);
//         u_list.push(*key);

//         if !u_list.is_empty() {
//             Some(MortonKeys {
//                 keys: u_list,
//                 index: 0,
//             })
//         } else {
//             None
//         }
//     }

//     fn get_v_list(
//         &self,
//         key: &<Self::Tree as Tree>::NodeIndex,
//     ) -> Option<<Self::Tree as Tree>::NodeIndices> {
//         if key.level() >= 2 {
//             let v_list = key
//                 .parent()
//                 .neighbors()
//                 .iter()
//                 .flat_map(|pn| pn.children())
//                 .filter(|pnc| {
//                     self.tree.get_all_keys_set().unwrap().contains(pnc) && !key.is_adjacent(pnc)
//                 })
//                 .collect_vec();

//             if !v_list.is_empty() {
//                 return Some(MortonKeys {
//                     keys: v_list,
//                     index: 0,
//                 });
//             } else {
//                 return None;
//             }
//         }
//         None
//     }

//     fn get_w_list(
//         &self,
//         leaf: &<Self::Tree as Tree>::NodeIndex,
//     ) -> Option<<Self::Tree as Tree>::NodeIndices> {
//         // Child level
//         let w_list = leaf
//             .neighbors()
//             .iter()
//             .flat_map(|n| n.children())
//             .filter(|nc| {
//                 self.tree.get_all_keys_set().unwrap().contains(nc) && !leaf.is_adjacent(nc)
//             })
//             .collect_vec();

//         if !w_list.is_empty() {
//             Some(MortonKeys {
//                 keys: w_list,
//                 index: 0,
//             })
//         } else {
//             None
//         }
//     }

//     fn get_x_list(
//         &self,
//         key: &<Self::Tree as Tree>::NodeIndex,
//     ) -> Option<<Self::Tree as Tree>::NodeIndices> {
//         let x_list = key
//             .parent()
//             .neighbors()
//             .into_iter()
//             .filter(|pn| self.tree.get_all_keys_set().unwrap().contains(pn) && !key.is_adjacent(pn))
//             .collect_vec();

//         if !x_list.is_empty() {
//             Some(MortonKeys {
//                 keys: x_list,
//                 index: 0,
//             })
//         } else {
//             None
//         }
//     }
// }
