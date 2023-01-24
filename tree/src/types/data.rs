use std::vec;

use solvers_traits::tree::FmmData;

use super::morton::MortonKey;

pub enum NodeType {
    Default,
    Fmm,
}

#[derive(Debug, Clone)]
pub struct NodeData {
    pub field_size: Vec<usize>,
    pub data: Vec<f64>,
    pub displacement: Vec<usize>,
}
