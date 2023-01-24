// Laplace kernel

use solvers_traits::{
    kernel::{Kernel}, fmm::Translation
};
use solvers_tree::types::data::NodeData;
use solvers_traits::types::{Error, EvalType};

use solvers_tree::types::morton::MortonKey;

pub struct KiFmmLaplace {

}

pub struct LaplaceKernel {
    pub dim: usize,
    pub is_singular: bool,
    pub value_dimension: usize
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
                            result.push(
                                1./(source-target).abs()*target_charge*source_charge
                            );
                        }
                    }
                    solvers_traits::types::Result::Ok(result)
                } 

                _ => solvers_traits::types::Result::Err(Error::Generic("foo".to_string()))
            }

    }

}


impl Translation for KiFmmLaplace {
    
    type Node = NodeData;

    fn p2m(node: &mut Self::Node) {
        
    }

    fn m2m(in_node: &Self::Node, out_node: &mut Self::Node) {
        
    }

    fn m2l(in_node: &Self::Node, out_node: &mut Self::Node) {
        
    }

    fn m2p(in_node: &Self::Node, out_node: &mut Self::Node) {
        
    }

    fn l2l(in_node: &Self::Node, out_node: &mut Self::Node) {
        
    }

    fn l2p(node: &mut Self::Node) {
        
    }

}




