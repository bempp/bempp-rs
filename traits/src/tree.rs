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

    // Get depth of tree
    fn get_depth(&self) -> usize;

    // Get adaptivity information
    fn get_adaptive(&self) -> bool;

    // Get all leaves, gets local keys in multi-node setting
    fn get_leaves(&self) -> &Self::NodeIndices;

    // Get all keys as a set, gets local keys in a multi-node setting
    fn get_leaves_set(&self) -> &Self::NodeIndicesSet;

    // Get all keys as a set, gets local keys in a multi-node setting
    fn get_keys_set(&self) -> &Self::NodeIndicesSet;
    
    // Get all keys at a given level, gets matching local keys in a multi-node setting
    fn get_keys(&self, level: usize) -> Self::NodeIndices;

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
