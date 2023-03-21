use bempp_traits::tree::Tree;
use bempp_traits::{
    fmm::{Fmm, SourceDataTree, SourceTranslation, TargetDataTree, TargetTranslation},
    tree::AttachedDataTree,
};
use bempp_tree::types::domain::Domain;
use bempp_tree::types::morton::MortonKey;
use bempp_tree::types::point::Point;
use bempp_tree::types::single_node::SingleNodeTree;
use itertools::Itertools;
use std::{collections::HashMap, hash::Hash};

pub struct FmmDataTree {
    multipoles: HashMap<MortonKey, Vec<f64>>,
    locals: HashMap<MortonKey, Vec<f64>>,
    potentials: HashMap<MortonKey, Vec<f64>>,
    points: HashMap<MortonKey, Vec<Point>>,
}

pub struct KiFmmSingleNode {
    tree: SingleNodeTree,

    // TODO: Wrapped into ArcMutex
    source_data_tree: FmmDataTree,
    target_data_tree: FmmDataTree,
}

impl FmmDataTree {
    fn new<'a>(tree: &SingleNodeTree) -> Self {
        let mut multipoles = HashMap::new();
        let mut locals = HashMap::new();
        let mut potentials = HashMap::new();
        let mut points = HashMap::new();

        for level in (0..=tree.get_depth()).rev() {
            if let Some(keys) = tree.get_keys(level) {
                for key in keys.iter() {
                    multipoles.insert(key.clone(), Vec::new());
                    locals.insert(key.clone(), Vec::new());
                    potentials.insert(key.clone(), Vec::new());
                    if let Some(point_data) = tree.get_points(key) {
                        points.insert(key.clone(), point_data.iter().cloned().collect_vec());
                    }
                }
            }
        }

        Self {
            multipoles,
            locals,
            potentials,
            points,
        }
    }
}

impl SourceDataTree for FmmDataTree {
    type Tree = SingleNodeTree;
    type Coefficient = f64;
    type Coefficients<'a> = &'a [f64];

    fn get_multipole_expansion<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Coefficients<'a>> {
        if let Some(multipole) = self.multipoles.get(key) {
            Some(multipole.as_slice())
        } else {
            None
        }
    }

    fn set_multipole_expansion<'a>(
        &'a mut self,
        key: &<Self::Tree as Tree>::NodeIndex,
        data: &Self::Coefficients<'a>,
    ) {
        if let Some(multipole) = self.multipoles.get_mut(key) {
            if !multipole.is_empty() {
                for (curr, &new) in multipole.iter_mut().zip(data.iter()) {
                    *curr += new;
                }
            } else {
                *multipole = data.clone().to_vec();
            }
        }
    }

    fn get_points<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::PointSlice<'a>> {
        if let Some(points) = self.points.get(key) {
            Some(points.as_slice())
        } else {
            None
        }
    }
}

impl SourceTranslation for FmmDataTree {
    fn p2m(&self) {}

    fn m2m(&self) {}
}

impl TargetTranslation for FmmDataTree {
    fn m2l(&self) {}

    fn l2l(&self) {}

    fn l2p(&self) {}

    fn p2l(&self) {}

    fn p2p(&self) {}
}

impl TargetDataTree for FmmDataTree {
    type Coefficient = f64;
    type Coefficients<'a> = &'a [f64];
    type Potential = f64;
    type Potentials<'a> = &'a [f64];
    type Tree = SingleNodeTree;

    fn get_local_expansion<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Coefficients<'a>> {
        if let Some(local) = self.locals.get(key) {
            Some(local.as_slice())
        } else {
            None
        }
    }

    fn set_local_expansion<'a>(
        &'a mut self,
        key: &<Self::Tree as Tree>::NodeIndex,
        data: &Self::Coefficients<'a>,
    ) {
        if let Some(locals) = self.locals.get_mut(key) {
            if !locals.is_empty() {
                for (curr, &new) in locals.iter_mut().zip(data.iter()) {
                    *curr += new;
                }
            } else {
                *locals = data.clone().to_vec();
            }
        }
    }

    fn get_potentials<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Potentials<'a>> {
        if let Some(potentials) = self.potentials.get(key) {
            Some(potentials.as_slice())
        } else {
            None
        }
    }

    fn set_potentials<'a>(
        &'a mut self,
        key: &<Self::Tree as Tree>::NodeIndex,
        data: &Self::Potentials<'a>,
    ) {
        if let Some(potentials) = self.potentials.get_mut(key) {
            if !potentials.is_empty() {
                for (curr, &new) in potentials.iter_mut().zip(data.iter()) {
                    *curr += new;
                }
            } else {
                *potentials = data.clone().to_vec();
            }
        }
    }
}

// impl Fmm for KiFmmSingleNode {

//     type SourceDataTree = FmmDataTree;
//     type TargetDataTree = FmmDataTree;
//     type PartitionTree = SingleNodeTree;

//     fn new<'a>(
//             points: <Self::PartitionTree as Tree>::PointSlice<'a>,
//             adaptive: bool,
//             n_crit: Option<u64>,
//             depth: Option<u64>

//         ) -> Self {

//         let tree = SingleNodeTree::new(points, adaptive, n_crit, depth);
//         let source_data_tree = FmmDataTree::new(&tree);
//         let target_data_tree= FmmDataTree::new(&tree);

//         KiFmmSingleNode { tree , source_data_tree, target_data_tree }
//     }

//     fn upward_pass(&self) {
//         // P2M over leaves
//         let leaves = self.tree.get_leaves();
//         self.source_data_tree.p2m(leaves);

//         // M2M over each key in a given level
//         for level in (1..=self.tree.get_depth()).rev() {
//             if let Some(keys) = self.tree.get_keys(level) {
//                 self.source_data_tree.m2m(keys);
//             }
//         }
//     }

//     fn downward_pass(&self) {

//         // Iterate down the tree (M2L/L2L)
//         for level in 2..=self.tree.get_depth() {

//             if let Some(keys) = self.tree.get_keys(level) {
//                 for key in keys.iter() {
//                     if let Some(v_list) = self.tree.get_interaction_list(key) {
//                         self.target_data_tree.m2l(v_list);
//                     }
//                 }

//                 self.target_data_tree.l2l(keys);
//             }
//         }

//         // Leaf level computations
//         let leaves = self.tree.leaves();
//         for leaf in leaves.iter() {
//             if let Some(x_list) = self.tree.get_x_list(leaf) {
//                 self.target_data_tree.p2l(leaf, &x_list);
//             }
//         }

//         // W list
//         self.target_data_tree.m2p(&leaves);

//         // U list
//         self.target_data_tree.p2p(&leaves);

//         // Translate local expansions to points in each node
//         self.target_data_tree.l2p(&leaves);

//     }

//     fn run(&self) {
//         self.upward_pass();
//         self.downward_pass();
//     }
// }
