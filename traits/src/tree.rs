pub trait Tree {
    type Domain;
    type Point;
    type Points;
    type NodeIndex;
    type NodeIndices;
    type NodeIndicesSet;

    // Get adaptivity information
    fn get_adaptive(&self) -> bool;

    // Get all keys, gets local keys in multi-node setting
    fn get_keys(&self) -> &Self::NodeIndices;

    // Get all keys as a set, gets local keys in a multi-node setting
    fn get_keys_set(&self) -> &Self::NodeIndicesSet;

    // Get all points, gets local keys in multi-node setting
    fn get_points(&self) -> &Self::Points;

    // Get domain, gets global domain in multi-node setting
    fn get_domain(&self) -> &Self::Domain;

    // Get tree node key associated with a given point
    fn map_point_to_key(&self, point: &Self::Point) -> Option<&Self::NodeIndex>;

    // Get points associated with a tree node key
    fn map_key_to_points(&self, key: &Self::NodeIndex) -> Option<&Self::Points>;
}

/// Locally Essential Trees take care of ghost nodes on other processors, and have access to all
/// the information they need to build the interaction lists for a tree.
pub trait LocallyEssentialTree {
    type RawTree: Tree;
    type NodeIndex;
    type NodeIndices;

    fn create_let(&mut self);
    fn get_near_field(&self, key: &Self::NodeIndex) -> Self::NodeIndices;
    fn get_interaction_list(&self, key: &Self::NodeIndex) -> Option<Self::NodeIndices>;
    fn get_x_list(&self, key: &Self::NodeIndex) -> Self::NodeIndices;
    fn get_w_list(&self, key: &Self::NodeIndex) -> Self::NodeIndices;
}
