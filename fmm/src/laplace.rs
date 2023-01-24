// Laplace kernel

use std::collections::HashSet;

use solvers_traits::types::{Error, EvalType};
use solvers_traits::{fmm::Translation, kernel::Kernel, tree::FmmTree};
use solvers_tree::types::data::NodeData;

use solvers_tree::types::{
    domain::Domain,
    morton::{MortonKey, MortonKeys},
    point::{Point, Points},
};


// TODO: Create from FMM Factory pattern, specialised for Rust in some way
pub struct KiFmm {
    pub kernel: Box<dyn Kernel<Data = Vec<f64>>>,
    pub tree: Box<
        dyn FmmTree<
            NodeIndex = MortonKey,
            NodeIndices = MortonKeys,
            NodeData = NodeData,
            NodeDataType = f64,
            Domain = Domain,
            Point = Point,
            Points = Points,
            NodeIndicesSet = HashSet<MortonKey>,
            NodeDataContainer = Vec<f64>,
        >,
    >,
}

// pub struct ChebyshevFmm {
//     pub kernel: Box<dyn Kernel<Data = Vec<f64>>>,
//     pub tree: Box<
//         dyn FmmTree<
//             NodeIndex = MortonKey,
//             NodeIndices = MortonKeys,
//             NodeData = NodeData,
//             NodeDataType = f64,
//             Domain = Domain,
//             Point = Point,
//             Points = Points,
//             NodeIndicesSet = HashSet<MortonKey>,
//             NodeDataContainer = Vec<f64>,
//         >,
//     >,
// }

pub struct LaplaceKernel {
    pub dim: usize,
    pub is_singular: bool,
    pub value_dimension: usize,
}

impl Kernel for LaplaceKernel {
    type Data = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn is_singular(&self) -> bool {
        self.is_singular
    }

    fn value_dimension(&self) -> usize {
        self.value_dimension
    }

    fn evaluate(
        &self,
        sources: &[f64],
        charges: &[f64],
        targets: &[f64],
        eval_type: &solvers_traits::types::EvalType,
    ) -> solvers_traits::types::Result<Self::Data> {
        let mut result: Vec<f64> = Vec::new();

        match eval_type {
            EvalType::Value => {
                for (source, source_charge) in sources.iter().zip(charges) {
                    for (target, target_charge) in targets.iter().zip(charges) {
                        result.push(1. / (source - target).abs() * target_charge * source_charge);
                    }
                }
                solvers_traits::types::Result::Ok(result)
            }

            _ => solvers_traits::types::Result::Err(Error::Generic("foo".to_string())),
        }
    }
}

impl Translation for KiFmm {
    type NodeIndex = MortonKey;

    fn p2m(&self, node: &Self::NodeIndex) {
        // Calculate centre of node from anchor, use to centre expansion.
        let centre = node.anchor();
        // self.kernel.evaluate();
        
        for leaf in self.tree.get_leaves().iter() {
            let sources = self.tree.get_points(leaf);
            // let targets = node.outward_equivalent_surface();
            // let result = self.kernel.evaluate(sources, charges, targets, eval_type);
            // self.tree.set_data(leaf, result)
        }
    }

    fn m2m(&self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {}

    fn l2l(&self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {}

    fn m2l(&self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {}

    fn l2p(&self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {}

    fn m2p(&self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {}
}
