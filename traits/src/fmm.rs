//! FMM traits

use crate::tree::Tree;

pub trait Translation {
    type NodeIndex;
    type LeafNodeIndex;

    // Particle to Multipole
    fn p2m(&mut self, node: &Self::LeafNodeIndex);

    // // Multipole to Multipole
    // fn m2m(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // // Multipole to Local
    // fn m2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // // Local to Local
    // fn l2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // // Multipole to Particle
    // fn m2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // // Local to Particle
    // fn l2p(&mut self, node: &Self::NodeIndex);

    // // Particle to Local
    // fn p2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // // Particle to Particle
    // fn p2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);
}

/// FmmTree take care of ghost nodes on other processors, and have access to all
/// the information they need to build the interaction lists for a tree, as well as
/// perform an FMM loop.
pub trait FmmTree<'a>: Tree<'a>
{
    // Container for data at tree nodes, must implement the FmmData trait
    type FmmLeafNodeIndex: FmmLeafNodeData<'a>;
    type FmmLeafNodeIndices: IntoIterator;
    type FmmNodeIndex: FmmNodeData<'a>;
    type FmmNodeIndices: IntoIterator;
    type FmmRawNodeIndex;

    // Create a locally essential tree (LET) that handles all ghost octant communication.
    fn create_let(&mut self);

    // Query local data for interaction lists for a given node.
    fn get_near_field(
        &'a self,
        node_index: &Self::FmmRawNodeIndex,
    ) -> Option<Self::FmmLeafNodeIndices>;

    fn get_x_list(
        &'a self,
        node_index: &Self::FmmRawNodeIndex,
    ) -> Option<Self::FmmLeafNodeIndices>;
    
    fn get_w_list(
        &'a self,
        node_index: &Self::FmmRawNodeIndex,
    ) -> Option<Self::FmmNodeIndices>;
    
    fn get_interaction_list(
        &'a self,
        node_index: &Self::FmmRawNodeIndex,
    ) -> Option<Self::FmmNodeIndices>;
}

/// FmmData containers extend a data container with specialised methods for FMM data,
/// specifically to handle the multipole and local expansion coefficients.
pub trait FmmNodeData<'a> {
    type CoefficientData;
    type CoefficientView;

    fn set_order(&mut self, order: usize);
    fn set_multipole_expansion(&mut self, data: &Self::CoefficientData);
    fn get_multipole_expansion(&'a self) -> Self::CoefficientView;
    fn set_local_expansion(&mut self, data: &Self::CoefficientData);
    fn get_local_expansion(&'a self) -> Self::CoefficientView;
}

pub trait FmmLeafNodeData<'a> {
    type Points;
    type PointData;
    type PointDataView;
    type PointIndices;
    fn get_points(&'a self) -> Self::Points;
    fn get_point_indices(&'a self) -> Self::PointIndices;
    fn get_point_data(&'a self, index: usize) -> Self::PointDataView;
    fn set_point_data(&mut self, index: usize, data: Self::PointData);
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
