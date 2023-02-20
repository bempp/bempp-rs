use crate::types::{morton::MortonKey, point::Points};

#[derive(Debug, Clone)]
pub struct LeafNode {
    pub key: MortonKey,
    pub points: Points,
}

pub type LeafNodes = Vec<LeafNode>;

#[derive(Debug, Clone)]
pub struct Node {
    pub key: MortonKey,
    pub data: NodeData,
}

pub type Nodes = Vec<Node>;

#[derive(Debug, Clone, Default)]
pub struct NodeData {
    pub field_size: Vec<usize>,
    pub raw: Vec<f64>,
    pub displacement: Vec<usize>,
}
