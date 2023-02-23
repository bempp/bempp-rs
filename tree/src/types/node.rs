/// Types for handling data on tree nodes.
use crate::types::{morton::MortonKey, point::Points};

/// A LeafNode contains Points which define the discretisation, and is indexed by a MortonKey.
#[derive(Debug, Clone, Default)]
pub struct LeafNode {
    pub key: MortonKey,
    pub points: Points,
}

pub type LeafNodes = Vec<LeafNode>;

/// A Node contains a generic data vector which a user can create views for specific to
/// their application.
#[derive(Debug, Clone, Default)]
pub struct Node {
    pub key: MortonKey,
    pub data: NodeData,
}

pub type Nodes = Vec<Node>;

/// NodeData provides a wrapper for raw data stored in each node.
#[derive(Debug, Clone, Default)]
pub struct NodeData {
    pub field_size: Vec<usize>,
    pub raw: Vec<f64>,
    pub displacement: Vec<usize>,
    pub init: bool,
}
