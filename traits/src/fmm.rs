//! FMM traits

use crate::tree::Tree;

/// Translation provides an interface for field translations as required by the FMM.
pub trait Translation {
    // Key index
    type NodeIndex;

    // Particle to Multipole.
    fn p2m(&mut self, leaf: &Self::NodeIndex);

    // Multipole to Multipole.
    fn m2m(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // Multipole to Local.
    fn m2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // Local to Local.
    fn l2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // Multipole to Particle.
    fn m2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // Local to Particle.
    fn l2p(&mut self, leaf: &Self::NodeIndex);

    // Particle to Local.
    fn p2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    // Particle to Particle.
    fn p2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);
}

/// FmmTree take care of ghost nodes on other processors, and have access to all
/// the information they need to build the interaction lists for a tree, as well as
/// perform an FMM loop.
pub trait FmmTree<'a>: Tree<'a> {
    type FmmRawNodeIndex;
    type FmmRawNodeIndices: IntoIterator;

    // Create a locally essential tree (LET) that handles all ghost octant communication.
    fn create_let(&mut self);

    // Query local nodes for the near field (u list) for a given node.
    fn get_near_field(&self, node_index: &Self::FmmRawNodeIndex)
        -> Option<Self::FmmRawNodeIndices>;

    // Query local nodes for the x list of a given node.
    fn get_x_list(&self, node_index: &Self::FmmRawNodeIndex) -> Option<Self::FmmRawNodeIndices>;

    // Query local nodes for the w list of a given node.
    fn get_w_list(&self, node_index: &Self::FmmRawNodeIndex) -> Option<Self::FmmRawNodeIndices>;

    // Query local nodes for the interaction list (u list) for a given node.
    fn get_interaction_list(
        &self,
        node_index: &Self::FmmRawNodeIndex,
    ) -> Option<Self::FmmRawNodeIndices>;
}

/// FmmNodeData extends a data container with specialised methods for FMM data,
/// specifically to handle the multipole and local expansion coefficients.
pub trait FmmNodeData<'a> {
    type CoefficientData;
    type CoefficientView;

    fn set_order(&mut self, order: usize);
    fn set_multipole_expansion(&mut self, data: Self::CoefficientData, order: usize);
    fn get_multipole_expansion(&'a self) -> Self::CoefficientView;
    fn set_local_expansion(&mut self, data: Self::CoefficientData, order: usize);
    fn get_local_expansion(&'a self) -> Self::CoefficientView;
}

/// FmmLeafNodeData extends a data container with specialised methods for FMM leaf data,
/// specifically to handle points, potentials and charges.
pub trait FmmLeafNodeData<'a> {
    type Points;
    type PointData;
    type PointDataView;
    type PointIndices;
    fn get_points(&'a self) -> &'a Self::Points;
    fn get_points_mut(&mut self) -> &mut Self::Points;
    fn get_charge(&'a self, index: usize) -> Self::PointDataView;
    fn get_potential(&'a self, index: usize) -> Self::PointDataView;
    fn set_charge(&mut self, index: usize, data: Self::PointData);
    fn set_potential(&mut self, index: usize, data: Self::PointData);
}

/// Fmm describes an interface of a generic FMM.
pub trait Fmm<'a> {
    // FMM core loop
    fn upward_pass(&mut self);
    fn downward_pass(&mut self);
    fn run(&mut self);
}

// KiFmmNode describes methods specialised to the KIFMM.
pub trait KiFmmNode {
    type Surface;
    type Domain;

    fn compute_surface(&self, order: usize, alpha: f64, domain: &Self::Domain) -> Self::Surface;
}
