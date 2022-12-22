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

#[derive(Clone, Copy)]
pub struct KiFmmNode3D {
    node: MortonKey,
}

pub struct KiFmmNodes3D {
    nodes: Vec<KiFmmNode3D>,
    index: usize,
}

impl KiFmmNodes3D {
    fn new() -> KiFmmNodes3D {
        KiFmmNodes3D {
            nodes: Vec::<KiFmmNode3D>::new(),
            index: 0,
        }
    }

    fn add(&mut self, elem: KiFmmNode3D) {
        self.nodes.push(elem);
    }
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
        self.node
    }
}

impl Iterator for KiFmmNodes3D {
    type Item = KiFmmNode3D;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.nodes.len() {
            return None;
        }

        self.index += 1;
        self.nodes.get(self.index).copied()
    }
}

impl FromIterator<KiFmmNode3D> for KiFmmNodes3D {
    fn from_iter<T: IntoIterator<Item = KiFmmNode3D>>(iter: T) -> Self {
        let mut res = KiFmmNodes3D::new();

        for item in iter {
            res.add(item)
        }
        res
    }
}

impl FmmTree for KiFmmTree {
    type NodeIndex = KiFmmNode3D;
    type IndexIter<'a> = KiFmmNodes3D;

    fn locality(&self, node_index: Self::NodeIndex) -> solvers_traits::types::Locality {
        match self.raw_tree.map_key_to_points(&node_index.node) {
            Some(_) => solvers_traits::types::Locality::Local,
            None => solvers_traits::types::Locality::Remote,
        }
    }

    // Composed of adjacent siblings and neighbours,
    fn get_near_field<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>> {
        
        self.raw_tree.near_field()
    }

    fn get_interaction_list<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>> {}

    fn get_x_list<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>> {}

    fn get_w_list<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>> {}

    fn get_level(&self, node_index: Self::NodeIndex) -> Option<usize> {
        Some(node_index.node.level() as usize)
    }

    fn get_parent(&self, node_index: Self::NodeIndex) -> Option<Self::NodeIndex> {
        Some(Self::NodeIndex {
            node: node_index.node.parent(),
        })
    }

    fn get_children<'a>(&'a self, node_index: Self::NodeIndex) -> Option<Self::IndexIter<'a>> {
        let children = node_index.node.children();
        let children = children
            .iter()
            .map(|&c| Self::NodeIndex { node: c })
            .collect();
        Some(children)
    }
}
