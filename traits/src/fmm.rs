//! FMM traits

use crate::tree::Tree;

/// Translation provides an interface for field translations as required by the FMM.
pub trait Translation {
    /// Key index to uniquely identify tree nodes.
    type NodeIndex;

    /// Particle to Multipole.
    fn p2m(&mut self, leaf: &Self::NodeIndex);

    /// Multipole to Multipole.
    fn m2m(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    /// Multipole to Local.
    fn m2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    /// Local to Local.
    fn l2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    /// Multipole to Particle.
    fn m2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    /// Local to Particle.
    fn l2p(&mut self, leaf: &Self::NodeIndex);

    /// Particle to Local.
    fn p2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);

    /// Particle to Particle.
    fn p2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex);
}

/// FmmTree take care of ghost nodes on other processors, and have access to all
/// the information they need to build the interaction lists for a tree, as well as
/// perform an FMM loop.
pub trait FmmTree<'a>: Tree<'a> {
    /// Key index to uniquely identify tree nodes, 'raw' reflects the fact that this
    /// should not refer to an index that contains associated node data.
    type FmmRawNodeIndex;

    /// Containers for FmmRawNodeIndices should implement the IntoIterator trait.
    type FmmRawNodeIndices: IntoIterator;

    /// Create a locally essential tree (LET) that handles all ghost octant communication.
    fn create_let(&mut self);

    /// Query local nodes for the near field (u list) for a given node.
    fn get_near_field(&self, node_index: &Self::FmmRawNodeIndex)
        -> Option<Self::FmmRawNodeIndices>;

    /// Query local nodes for the x list of a given node.
    fn get_x_list(&self, node_index: &Self::FmmRawNodeIndex) -> Option<Self::FmmRawNodeIndices>;

    /// Query local nodes for the w list of a given node.
    fn get_w_list(&self, node_index: &Self::FmmRawNodeIndex) -> Option<Self::FmmRawNodeIndices>;

    /// Query local nodes for the interaction list (u list) for a given node.
    fn get_interaction_list(
        &self,
        node_index: &Self::FmmRawNodeIndex,
    ) -> Option<Self::FmmRawNodeIndices>;
}

/// FmmNodeData extends a Node data container with specialised methods for FMM data,
/// specifically to handle the multipole and local expansion coefficients.
pub trait FmmNodeData<'a> {
    /// A container of coefficients related to multipole/local expansions.
    type CoefficientData;

    /// A view of coefficients related to multipole/local expansions.
    type CoefficientView;

    /// Set the expansion order at this node.
    fn set_order(&mut self, order: usize);

    /// Set the multipole expansion at this node.
    fn set_multipole_expansion(&mut self, data: Self::CoefficientData, order: usize);

    /// Get the multipole expansion at this node.
    fn get_multipole_expansion(&'a self) -> Self::CoefficientView;

    /// Set the local expansion at this node.
    fn set_local_expansion(&mut self, data: Self::CoefficientData, order: usize);

    /// Get the local expansion at this node.
    fn get_local_expansion(&'a self) -> Self::CoefficientView;
}

/// FmmLeafNodeData extends a data container with specialised methods for FMM leaf data,
/// specifically to handle points, potentials and charges.
pub trait FmmLeafNodeData<'a> {
    /// Containers for points.
    type Points;

    /// Containers for associated point data.
    type PointData;

    /// View for associated point data.
    type PointDataView;

    /// Get a reference to all the points in a LeafNode.
    fn get_points(&'a self) -> &'a Self::Points;

    /// Get a mutable reference to all the points in a Leaf Node.
    fn get_points_mut(&mut self) -> &mut Self::Points;

    /// Get the charge at a point with a given global index.
    fn get_charge(&'a self, index: usize) -> Self::PointDataView;

    /// Get the potential at a point with a given global index.
    fn get_potential(&'a self, index: usize) -> Self::PointDataView;

    /// Set the charge at a point with a given global index.
    fn set_charge(&mut self, index: usize, data: Self::PointData);

    /// Set the potential at a point with a given global index.
    fn set_potential(&mut self, index: usize, data: Self::PointData);
}

/// Fmm provides an interface for using different FMM implementations.
pub trait Fmm<'a> {
    /// Perform the upward pass for multipole expansions.
    fn upward_pass(&mut self);

    /// Perform the downward pass to find local expansions.
    fn downward_pass(&mut self);

    /// Perform the upward and downward passes together.
    fn run(&mut self);
}

/// KiFmmNode describes methods specialised to the KIFMM.
pub trait KiFmmNode {
    /// The type of surface used when calculating expansions.
    type Surface;

    /// The type of domain used to encapsulate the points in the FmmTree
    type Domain;

    /// Compute the check/equivalent surfaces to a specified order.
    fn compute_surface(&self, order: usize, alpha: f64, domain: &Self::Domain) -> Self::Surface;
}
