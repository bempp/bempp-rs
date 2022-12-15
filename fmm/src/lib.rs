//! Fast Solver FMM library

use solvers_traits::{
    fmm::{
        fmmtree::{FmmTree, Node},
        translation::Translation,
    },
    tree::Tree,
};

use solvers_tree::types::{
    domain::Domain,
    morton::{MortonKey, MortonKeys},
    point::{Point, Points},
};

pub enum Geometry {
    Square,
    Cube,
}

pub struct KiFmmNode3D {
    key: MortonKey,
}

pub struct KiFmmTree {
    raw_tree: Box<
        dyn Tree<
            Point = Point,
            Domain = Domain,
            Points = Points,
            NodeIndex = MortonKey,
            NodeIndices = MortonKeys,
        >,
    >,
 }

impl Node for KiFmmNode3D {
    type Item = f64;

    type Geometry = Geometry;

    type NodeIndex = MortonKey;

    // type View =

    fn node_geometry(&self) -> Self::Geometry {
        Geometry::Cube
    }

    fn node_index(&self) -> Self::NodeIndex {
        self.key
    }
}

impl FmmTree for KiFmmTree {
    type NodeIndex = MortonKey;
    type IndexIter<'a>: std::iter::Iterator<Item = Self::NodeIndex>
        where
            Self: 'a = MortonKeys;
    
    fn locality(&self, node_index: Self::NodeIndex) -> solvers_traits::types::Locality {
        
    }

    fn near_field<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>> {
        
    }

    fn interaction_list<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>> {
        
    }

    fn get_x_list<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>> {
        
    }

    fn get_w_list<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>> {
        
    }

    fn level(&self, node_index: Self::NodeIndex) -> Option<usize> {
        
    }

    fn ancestor(&self, node_index: Self::NodeIndex) -> Option<Self::NodeIndex> {
        
    }

    fn descendants<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>> {
        
    }

}
