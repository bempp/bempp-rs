//! FMM traits

use crate::tree::Tree;

pub trait Translation {
    type NodeIndex;

    // Particle to Multipole
    fn p2m(&mut self, node: &Self::NodeIndex);

    // Multipole to Multipole
    fn m2m(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // Multipole to Local
    fn m2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // Local to Local
    fn l2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // Multipole to Particle
    fn m2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // Local to Particle
    fn l2p(&mut self, node: &Self::NodeIndex);

    // Particle to Local
    fn p2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // Particle to Particle
    fn p2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);
}

/// FmmTree take care of ghost nodes on other processors, and have access to all
/// the information they need to build the interaction lists for a tree, as well as
/// perform an FMM loop.
pub trait FmmTree<'a>: Tree {
    // Container for data at tree nodes, must implement the FmmData trait
    // type FmmNodeDataType: FmmData;
    // type NodeDataContainer;

    type LeafNodeIndex: FmmLeafNodeData;
    type LeafNodeIndices: IntoIterator;
    type NodeIndex: FmmNodeData;
    type NodeIndices: IntoIterator;

    // Create a locally essential tree (LET) that handles all ghost octant communication.
    fn create_let(&mut self);

    // Query local data for interaction lists for a given node.
    fn get_near_field(&'a self, node_index: &<Self as FmmTree<'a>>::NodeIndex) -> Option<<Self as FmmTree>::NodeIndices>;
    fn get_x_list(&'a self, node_index: &<Self as FmmTree<'a>>::NodeIndex) -> Option<<Self as FmmTree>::NodeIndices>;
    fn get_w_list(&'a self, node_index: &<Self as FmmTree<'a>>::NodeIndex) -> Option<<Self as FmmTree>::NodeIndices>;
    fn get_interaction_list(&'a self, node_index: &<Self as FmmTree<'a>>::NodeIndex) -> Option<<Self as FmmTree>::NodeIndices>;

}

/// FmmData containers extend a data container with specialised methods for FMM data,
/// specifically to handle the multipole and local expansion coefficients.
pub trait FmmNodeData {
    type CoefficientData;
    type CoefficientView;

    fn set_order(&mut self, order: usize);
    fn set_multipole_expansion(&mut self, data: &Self::CoefficientData);
    fn get_multipole_expansion(&self) -> Self::CoefficientView;
    fn set_local_expansion(&mut self, data: &Self::CoefficientData);
    fn get_local_expansion(&self) -> Self::CoefficientView;
}

pub trait FmmLeafNodeData {
    type ParticleData;
    type ParticleView;
    fn get_leaf_data(&self) -> Self::ParticleView;
    fn set_leaf_data(&self, data: &Self::ParticleData);
}

pub trait Fmm {
    // FMM core loop
    fn upward_pass(&mut self);
    fn downward_pass(&mut self);
    fn run(&mut self);
}

// Special interface for NodeIndices in the KIFMM
pub trait KiFmmNode {
    type Surface;
    type Domain;

    fn compute_surface(&self, order: usize, alpha: f64, domain: &Self::Domain) -> Self::Surface;
}
