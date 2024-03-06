use bempp_traits::{
    field::SourceToTargetData,
    fmm::TargetTranslation,
    kernel::Kernel,
    tree::{FmmTree, Tree},
};
use bempp_tree::types::single_node::SingleNodeTreeNew;
use cauchy::Scalar;
use num::Float;

use crate::fmm::KiFmm;

impl<T, U, V, W> TargetTranslation for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTreeNew<W>>,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: Scalar<Real = W> + Default + Float,
{
    fn l2l(&self, level: u64) {}

    fn l2p(&self) {}

    fn m2p(&self) {}

    fn p2p(&self) {}
}
