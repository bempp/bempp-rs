use bempp_traits::tree::{FmmTree, Tree};
use bempp_tree::types::{domain::Domain, morton::MortonKey, single_node::SingleNodeTreeNew};
use cauchy::Scalar;
use num::Float;

#[derive(Default)]
pub struct SingleNodeFmmTree<T: Float + Default + Scalar<Real = T>> {
    pub source_tree: SingleNodeTreeNew<T>,
    pub target_tree: SingleNodeTreeNew<T>,
    pub domain: Domain<T>,
}

impl<T> FmmTree for SingleNodeFmmTree<T>
where
    T: Float + Default + Scalar<Real = T> + Send + Sync,
{
    type Precision = T;
    type NodeIndex = MortonKey;
    type Tree = SingleNodeTreeNew<T>;

    fn get_source_tree(&self) -> &Self::Tree {
        &self.target_tree
    }

    fn get_target_tree(&self) -> &Self::Tree {
        &self.source_tree
    }

    fn get_domain(&self) -> &<Self::Tree as Tree>::Domain {
        &self.domain
    }
}

unsafe impl<T: Float + Default + Scalar<Real = T>> Send for SingleNodeFmmTree<T> {}
unsafe impl<T: Float + Default + Scalar<Real = T>> Sync for SingleNodeFmmTree<T> {}
