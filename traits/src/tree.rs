/// Tree is the trait interface for distributed octrees implemented by Rusty Fast Solvers.
pub trait Tree {
    // The computational domain defined by the tree
    type Domain;

    // The type of points that define a tree
    type Point;

    // A container for multiple Points
    type Points;

    // Unique index for tree nodes
    type NodeIndex;

    // Container for multiple tree nodes
    type NodeIndices;

    // A set of NodeIndices
    type NodeIndicesSet;

    // Type of element in a node's data container
    type NodeDataType;

    // Get adaptivity information
    fn get_adaptive(&self) -> bool;

    // Get all leaves, gets local keys in multi-node setting
    fn get_leaves(&self) -> &Self::NodeIndices;

    // Get all keys as a set, gets local keys in a multi-node setting
    fn get_leaves_set(&self) -> &Self::NodeIndicesSet;
    
    // Get all keys as a set, gets local keys in a multi-node setting
    fn get_keys_set(&self) -> &Self::NodeIndicesSet;

    // Get all points, gets local keys in multi-node setting
    fn get_all_points(&self) -> &Self::Points;

    // Get domain, gets global domain in multi-node setting
    fn get_domain(&self) -> &Self::Domain;

    // Get tree leaf associated with a given point
    fn get_leaf(&self, point: &Self::Point) -> Option<&Self::NodeIndex>;

    // Get points associated with a tree leaf
    fn get_points(&self, leaf: &Self::NodeIndex) -> Option<&Self::Points>;

    // Set data associated with a given leaf node.
    fn set_data(&mut self, node_index: &Self::NodeIndex, data: Self::NodeDataType);

    // Get data associated with a given leaf node.
    fn get_data(&self, node_index: &Self::NodeIndex) -> Option<&Self::NodeDataType>;
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
