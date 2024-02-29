use bempp_tree::types::single_node::SingleNodeTree;
use cauchy::Scalar;
use num::Float;

pub struct SingleNodeFmmTree<T: Float + Default + Scalar<Real = T>> {
    pub source_tree: SingleNodeTree<T>,
    pub target_tree: SingleNodeTree<T>,
}
