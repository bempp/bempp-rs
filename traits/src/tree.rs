use crate::types::{Locality, Scalar};

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
    type NodeIndex;
    type NodeIndices;

    fn create_let(&mut self);
    fn locality(&self, node_index: &Self::NodeIndex) -> Locality;
    fn get_near_field(&self, node_index: &Self::NodeIndex) -> Option<Self::NodeIndices>;
    fn get_interaction_list(&self, node_index: &Self::NodeIndex) -> Option<Self::NodeIndices>;
    fn get_x_list(&self, node_index: &Self::NodeIndex) -> Option<Self::NodeIndices>;
    fn get_w_list(&self, node_index: &Self::NodeIndex) -> Option<Self::NodeIndices>;
}

// Definition of a tree node for the FMM
// A node contains
// - `Data`: Typically an array of numbers associated with a node.
// - `NodeData`: A type that describes geometric data of a node.
// - `NodeIndex`: An index associated with the node.
pub trait FmmNode {
    // The type of the coefficients associated with the node.
    type Item;

    // Type of the geometry definition.
    type Geometry;

    // Type that describes node indices.
    type NodeIndex;

    // Data view
    type View<'a>
    where
        Self: 'a;

    type ViewMut<'a>
    where
        Self: 'a;

    // Get the node geometry.
    fn node_geometry(&self) -> Self::Geometry;

    // Get a view onto the data.
    fn view<'a>(&'a self) -> Self::View<'a>;

    // Get a mutable view onto the data.
    fn view_mut<'a>(&'a mut self) -> Self::ViewMut<'a>;

    // Get the index of the node.
    fn node_index(&self) -> Self::NodeIndex;
}

pub trait FmmTree {
    type FmmNodeIndex: FmmNode;
    type FmmNodeIndices<'a>: std::iter::Iterator<Item = Self::FmmNodeIndex>
    where
        Self: 'a;

    fn upward_pass(&self);
    fn downward_pass(&self);
    fn run(&self);
}
