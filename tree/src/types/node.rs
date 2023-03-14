//! Types for handling data on tree nodes.
use crate::types::{morton::MortonKey, point::Points};

/// A LeafNode contains Points which define the discretisation, and is indexed by a MortonKey.
#[derive(Debug, Clone, Default)]
pub struct LeafNode {
    pub key: MortonKey,
    pub points: Points,
}

pub type LeafNodes = Vec<LeafNode>;

/// A Node specifies an encoded MortonKey, and associatec data with it via a NodeData struct.
#[derive(Debug, Clone, Default)]
pub struct Node {
    /// Encoded MortonKey associated with this Node.
    pub key: MortonKey,

    /// Data contained in this Node.
    pub data: NodeData,
}

pub type Nodes = Vec<Node>;

/// NodeData provides a wrapper for raw data stored in each node.
#[derive(Debug, Clone, Default)]
pub struct NodeData {
    /// The size of each field wrapped in a raw data container.
    pub field_size: Vec<usize>,

    /// The raw data container, currently only support f64 data.
    pub raw: Vec<f64>,

    /// The displacement of each field wrapped in the raw data container.
    pub displacement: Vec<usize>,

    /// Specifies whether the Node has been initialized, used when updating NodeData to avoid
    /// reallocations.
    pub init: bool,
}
