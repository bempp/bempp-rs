//! FMM traits

use crate::tree::Tree;

pub trait Translation {
    type NodeIndex;

    // Particle to Multipole
    fn p2m(&self, node: &Self::NodeIndex);

    // Multipole to Multipole
    fn m2m(
        &self,
        in_node: &Self::NodeIndex,
        out_node: &Self::NodeIndex,
    );

    // Multipole to Local
    fn m2l(
        &self,
        in_node: &Self::NodeIndex,
        out_node: &Self::NodeIndex,
    );

    // Local to Local
    fn l2l(
        &self,
        in_node: &Self::NodeIndex,
        out_node: &Self::NodeIndex,
    );

    // Multipole to Particle
    fn m2p(
        &self,
        in_node: &Self::NodeIndex,
        out_node: &Self::NodeIndex,
    );

    // Local to Particle
    fn l2p(
        &self,
        in_node: &Self::NodeIndex,
        out_node: &Self::NodeIndex,
    );
}

/// FmmTree take care of ghost nodes on other processors, and have access to all
/// the information they need to build the interaction lists for a tree, as well as
/// perform an FMM loop.
pub trait FmmTree: Tree {

    // Container for data at tree nodes, must implement the FmmData trait
    type NodeData: FmmData;
    type NodeDataContainer;

    // Type of particles in a given leaf
    // type ParticleData;

    // Create a locally essential tree (LET) that handles all ghost octant communication.
    fn create_let(&mut self);

    // Query local data for interaction lists for a given node.
    fn get_near_field(&self, node_index: &Self::NodeIndex) -> Option<Self::NodeIndices>;
    fn get_interaction_list(&self, node_index: &Self::NodeIndex) -> Option<Self::NodeIndices>;
    fn get_x_list(&self, node_index: &Self::NodeIndex) -> Option<Self::NodeIndices>;
    fn get_w_list(&self, node_index: &Self::NodeIndex) -> Option<Self::NodeIndices>;

    // Getters/setters for expansion coefficients.
    fn set_multipole_expansion(&mut self, node_index: &Self::NodeIndex, data: &Self::NodeDataContainer);
    fn get_multipole_expansion(&self, node_index: &Self::NodeIndex) -> Option<Self::NodeDataContainer>;
    fn set_local_expansion(&mut self, node_index: &Self::NodeIndex, data: &Self::NodeDataContainer);
    fn get_local_expansion(&self, node_index: &Self::NodeIndex) -> Option<Self::NodeDataContainer>;

    // Getters for particle data
    // fn get_particles(&self, node_index: &Self::NodeIndex) -> Option<Self::ParticleData>;

    // FMM core loop
    fn upward_pass(&mut self);
    fn downward_pass(&mut self);
    fn run(&mut self, expansion_order: usize);
}

/// FmmData containers extend a data container with specialised methods for FMM data,
/// specifically to handle the multipole and local expansion coefficients.
pub trait FmmData {
    type CoefficientDataType;

    fn set_expansion_order(&mut self, order: usize);
    fn get_expansion_order(&self) -> usize;
    fn set_multipole_expansion(&mut self, data: &Self::CoefficientDataType);
    fn get_multipole_expansion(&self) -> Self::CoefficientDataType;
    fn set_local_expansion(&mut self, data: &Self::CoefficientDataType);
    fn get_local_expansion(&self) -> Self::CoefficientDataType;
}


// Special interface for NodeIndices in the KIFMM
// TODO: Implement for MortonKey
trait KiFmmNode {

    type NodeIndex;

    fn upward_equivalent_surface(&self);

    fn upward_check_surface(&self);

    fn downward_equivalent_surface(&self);

    fn downward_check_surface(&self);
}