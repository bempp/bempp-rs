//! Tree data structures

use cauchy::Scalar;

pub enum Locality {
    Local,
    Ghost,
    Remote,
}

pub trait Node {
    type Item: Scalar;
    type NodeData;
    type NodeIndex;

    fn get_coefficients(&self) -> &[Self::Item];
    fn set_coefficients(&mut self) -> &mut [Self::Item];

    fn node_data(&self) -> Self::NodeData;

    fn node_index(&self) -> Self::NodeIndex;
}

pub trait Tree {
    type NodeIndex;

    fn locality(&self, node_index: Self::NodeIndex) -> Locality;

    fn near_field(&self, node_index: Self::NodeIndex) -> &[Self::NodeIndex];

    fn get_interaction_list(&self, node_index: Self::NodeIndex) -> &[Self::NodeIndex];
    fn get_x_list(&self, node_index: Self::NodeIndex) -> &[Self::NodeIndex];
    fn get_w_list(&self, node_index: Self::NodeIndex) -> &[Self::NodeIndex];

    fn level(&self, node_index: Self::NodeIndex) -> usize;

    fn ancestor(&self, node_index: Self::NodeIndex) -> Self::NodeIndex;

    fn descendents(&self, node_index: Self::NodeIndex) -> &[Self::NodeIndex];
}
