//! FMM traits

use crate::tree::Tree;

// TODO
// I don't quite understand how this makes sense. What should Translation be attached to?
// If it is attached to an FMM object (i.e. KiFMM) we have the problems with &mut self
// and also we cannot batch operations together as we always have an in-node and an out-node.
// My suggestion is to attach Translation to nodes of the tree. Then for example calling p2m on
// the node performs all particle-to-multipole operations for that node, etc.
// This also solves the problem with &mut as we can operate on all nodes simultaneously.

/// Translation provides an interface for field translations as required by the FMM.
pub trait Translation {
    /// Key index to uniquely identify tree nodes.
    type NodeIndex;

    // Only on source tree
    /// Particle to Multipole.
    fn p2m(&self, leaf: &Self::NodeIndex);

    // Only on source tree
    /// Multipole to Multipole.
    fn m2m(&self, node: &Self::NodeIndex);

    // Only on target tree
    /// Multipole to Local.
    fn m2l(&self, node: &Self::NodeIndex);

    // Need to correct the others...

    // Only on target tree
    /// Local to Local.
    fn l2l(&self, node: &Self::NodeIndex);

    // Only on target tree
    /// Multipole to Particle.
    fn m2p(&self, node: &Self::NodeIndex);

    // Only on target tree
    /// Local to Particle.
    fn l2p(&mut self, leaf: &Self::NodeIndex);

    // Only on target tree
    /// Particle to Local.
    fn p2l(&self, node: &Self::NodeIndex);

    // Only on target tree
    /// Particle to Particle.
    fn p2p(&self, node: &Self::NodeIndex);
}

/// FmmTree take care of ghost nodes on other processors, and have access to all
/// the information they need to build the interaction lists for a tree, as well as
/// perform an FMM loop.
pub trait Fmm {
    // TODO Not required
    /// Key index to uniquely identify tree nodes, 'raw' reflects the fact that this
    /// should not refer to an index that contains associated node data.
    type Tree: Tree;

    /// Create a locally essential tree (LET) that handles all ghost octant communication.
    fn create_locally_essential_tree(&mut self);

    /// Query local nodes for the near field (u list) for a given node.
    fn get_near_field(
        &self,
        node_index: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndexIterator>;

    // Do the rest like `get_near_field`

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

// This should be handled by the simple DataTree trait. We don't need all those details in the
// trait. How the data is manipulated is part of the implementation.

/// FmmNodeData extends a Node data container with specialised methods for FMM data,
/// specifically to handle the multipole and local expansion coefficients.
pub trait FmmNodeData {
    /// A container of coefficients related to multipole/local expansions.
    type CoefficientData;

    /// A view of coefficients related to multipole/local expansions.
    type CoefficientView<'a>
    where
        Self: 'a;

    /// Set the expansion order at this node.
    fn set_order(&mut self, order: usize);

    /// Set the multipole expansion at this node.
    fn set_multipole_expansion(&mut self, data: Self::CoefficientData, order: usize);

    /// Get the multipole expansion at this node.
    fn get_multipole_expansion<'a>(&'a self) -> Self::CoefficientView<'a>;

    /// Set the local expansion at this node.
    fn set_local_expansion(&mut self, data: Self::CoefficientData, order: usize);

    /// Get the local expansion at this node.
    fn get_local_expansion<'a>(&'a self) -> Self::CoefficientView<'a>;
}

// TODO: Again, all of this is very much only needed as part of an implementation.
// TODO: The trait does not need the lifetime.
// Take for example fn get_points(&'a self) -> &'a Self::Points;
// You are creating here a reference with &'a self that has the lifetime of the object as
// you have tied the lifetime of self to the lifetime of FmmLeafNodeData.
// You should rather use fn get_points<'a>(&'a self) -> &'a Self::Points; Hence,
// the lifetime becomes a generic parameter attached to the method and now simply means
// &'a Self::Points is bounded by the lifetime of &'a self.

// Fix similar to NodeData

/// FmmLeafNodeData extends a data container with specialised methods for FMM leaf data,
/// specifically to handle points, potentials and charges.
pub trait FmmLeafNodeData {
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
pub trait Fmm {
    /// Perform the upward pass for multipole expansions.
    fn upward_pass(&mut self);

    /// Perform the downward pass to find local expansions.
    fn downward_pass(&mut self);

    /// Perform the upward and downward passes together.
    fn run(&mut self);
}

// TODO: This again is an implementational detail and not needed for generic interface.
/// KiFmmNode describes methods specialised to the KIFMM.
pub trait KiFmmNode {
    /// The type of surface used when calculating expansions.
    type Surface;

    /// The type of domain used to encapsulate the points in the FmmTree
    type Domain;

    /// Compute the check/equivalent surfaces to a specified order.
    fn compute_surface(&self, order: usize, alpha: f64, domain: &Self::Domain) -> Self::Surface;
}
