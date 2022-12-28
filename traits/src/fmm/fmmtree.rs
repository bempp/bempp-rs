
use crate::types::{Locality, Scalar};

// Definition of a tree node.
// A node contains
// - `Data`: Typically an array of numbers associated with a node.
// - `NodeData`: A type that describes geometric data of a node.
// - `NodeIndex`: An index associated with the node.
pub trait Node {
    // The type of the coefficients associated with the node.
    type Item: Scalar;
    // Type of the geometry definition.
    type Geometry;
    // Type that describes node indices.
    type NodeIndex;
    // Data view with a lifetime that depends on the
    // lifetime of the tree node.
    // type View<'a>: crate::general::IndexableView
    // where
    //     Self: 'a;

    // Get the node geometry.
    fn node_geometry(&self) -> Self::Geometry;

    // // Get a view onto the data.
    // fn view<'a>(&'a self) -> Self::View<'a>;

    // // Get a mutable view onto the data.
    // fn view_mut<'a>(&'a mut self) -> Self::View<'a>;

    // Get the index of the node.
    fn node_index(&self) -> Self::NodeIndex;
}

// Implementation of an FMM Tree.
// A FMM Tree is an octree with additional information
// about near-field, interaction list, etc.
pub trait FmmTree {
    type NodeIndex;
    type IndexIter<'a>: std::iter::Iterator<Item = Self::NodeIndex>
    where
        Self: 'a;

    fn locality(&self, node_index: Self::NodeIndex) -> Locality;

    // Get the near-field for local indices.
    // If the index is not local, `None` is returned.
    fn get_near_field<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>>;

    // Get the interaction list.
    fn get_interaction_list<'a>(
        &'a self,
        node_index: Self::NodeIndex,
    ) -> Option<Self::IndexIter<'a>>;

    // Get the x list.
    fn get_x_list<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>>;

    // Get the w list.
    fn get_w_list<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>>;

    // Get the level of the node.
    fn get_level(&self, node_index: Self::NodeIndex) -> Option<usize>;

    // Get the parent of the node.
    fn get_parent(&self, node_index: Self::NodeIndex) -> Option<Self::NodeIndex>;

    // Get the children of the node.
    fn get_children<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>>;
}