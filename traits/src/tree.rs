//! Traits
use std::collections::HashSet;

/// Tree is the trait interface for distributed octrees implemented by Rusty Fast Solvers.
/// This trait makes no assumptions about the downstream usage of a struct implementing Tree,
/// it simply provides methods for accessing tree nodes, and associated data, and is generically
/// defined for both single and multi-node settings.
pub trait Tree {
    // The computational domain defined by the tree.
    type Domain;

    // The type of points that define a tree.
    type Point;

    // Slice of points.
    type PointSlice<'a>: IntoIterator<Item = &'a Self::Point>
    where
        Self: 'a;

    // Data to be attached to each point.
    type PointData;

    type PointDataSlice<'a>: IntoIterator<Item = &'a Self::PointData>
    where
        Self: 'a;

    // A tree node.
    type NodeIndex;

    // Slice of nodes.
    type NodeIndexSlice<'a>: IntoIterator<Item = &'a Self::NodeIndex>
    where
        Self: 'a;

    // Copy of nodes
    type NodeIndices: IntoIterator<Item = Self::NodeIndex>;

    fn new<'a>(
        points: Self::PointSlice<'a>,
        point_data: Self::PointDataSlice<'a>,
        adaptive: bool,
        n_crit: Option<u64>,
        depth: Option<u64>,
    ) -> Self;

    // Get depth of tree.
    fn get_depth(&self) -> u64;

    // Get a reference to all leaves, gets local keys in multi-node setting.
    fn get_leaves<'a>(&'a self) -> Self::NodeIndexSlice<'a>;

    // Get a reference to keys at a given level, gets local keys in a multi-node setting.
    fn get_keys<'a>(&'a self, level: u64) -> Option<Self::NodeIndexSlice<'a>>;

    // Get a reference to all keys, gets local keys in a multi-node setting.
    fn get_all_keys<'a>(&'a self) -> Option<Self::NodeIndexSlice<'a>>;
    
    // Gets a reference to the points contained with a leaf node.
    fn get_points<'a>(&'a self, key: &Self::NodeIndex) -> Option<Self::PointSlice<'a>>;

    // Get domain defined by the points, gets global domain in multi-node setting.
    fn get_domain<'a>(&'a self) -> &'a Self::Domain;

    // Checks whether a a given node corresponds to a leaf
    fn is_leaf(&self, key: &Self::NodeIndex) -> bool;

    // Checks whether a a given node is contained in the tree
    fn is_node(&self, key: &Self::NodeIndex) -> bool;
}

pub trait AttachedDataTree {
    type Data<'a>
    where
        Self: 'a;
    type Tree: Tree;

    fn get_data<'a>(&'a self, key: &<Self::Tree as Tree>::NodeIndex) -> Option<&Self::Data<'a>>;

    fn get_data_mut<'a>(
        &'a mut self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<&'a mut Self::Data<'a>>;
}

pub trait FmmInteractionLists {
    type Tree: Tree;

    fn get_v_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;
    fn get_x_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;
    fn get_w_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;
    fn get_u_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;
}
