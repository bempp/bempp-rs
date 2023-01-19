// //! Fast Solver FMM library
use solvers_traits::tree::{FmmNode, FmmTree, LocallyEssentialTree};

use solvers_tree::types::{
    morton::{MortonKey, MortonKeys},
    multi_node::MultiNodeTree,
};

pub enum Geometry {
    Square,
    Cube,
}

#[derive(Clone)]
pub struct KiFmmNode3D {
    index: MortonKey,
    data: Vec<[f64; 5]>,
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

impl Iterator for KiFmmNodes3D {
    type Item = KiFmmNode3D;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.nodes.len() {
            return None;
        }

        self.index += 1;
        self.nodes.get(self.index).cloned()
    }
}

pub struct KiFmmTree {
    raw_tree: Box<dyn LocallyEssentialTree<NodeIndex = MortonKey, NodeIndices = MortonKeys>>,
}

impl FmmNode for KiFmmNode3D {
    type Item = f64;

    type Geometry = Geometry;

    type NodeIndex = MortonKey;

    type View<'a> = std::slice::Iter<'a, [f64; 5]>;

    type ViewMut<'a> = std::slice::IterMut<'a, [f64; 5]>;

    fn node_geometry(&self) -> Self::Geometry {
        Geometry::Cube
    }

    fn view<'a>(&'a self) -> Self::View<'a> {
        self.data.iter()
    }

    fn view_mut<'a>(&'a mut self) -> Self::ViewMut<'a> {
        self.data.iter_mut()
    }

    fn node_index(&self) -> Self::NodeIndex {
        self.index
    }
}

impl FmmTree for KiFmmTree {
    type FmmNodeIndex = KiFmmNode3D;
    type FmmNodeIndices<'a> = KiFmmNodes3D;

    fn upward_pass(&self) {}

    fn downward_pass(&self) {}

    fn run(&self) {}
}
