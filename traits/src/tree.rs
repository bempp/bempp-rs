pub trait Tree {
    type Domain;
    type Point;
    type Points;
    type NodeIndex;
    type NodeIndices;

    // Get balancing information
    fn get_balanced(&self) -> bool;

    // Get all keys, gets local keys in multi-node setting
    fn get_keys(&self) -> &Self::NodeIndices;

    // Get all points, gets local keys in multi-node setting
    fn get_points(&self) -> &Self::Points;

    // Get domain, gets global domain in multi-node setting
    fn get_domain(&self) -> &Self::Domain;

    // Get tree node key associated with a given point
    fn map_point_to_key(&self, point: &Self::Point) -> Option<&Self::NodeIndex>;

    // Get points associated with a tree node key
    fn map_key_to_points(&self, key: &Self::NodeIndex) -> Option<&Self::Points>;
}
