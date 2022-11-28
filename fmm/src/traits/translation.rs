//! Traits for dealing with translation operators
use crate::traits::tree::Node;
use cauchy::Scalar;

pub trait Translation {
    type Item: Scalar;
    type Node: Node;

    // Particle to Multipole
    fn p2m(node: &mut Self::Node);

    // Multipole to Multipole
    fn m2m(in_node: &Self::Node, out_node: &mut Self::Node);

    // Multipole to Local
    fn m2l(in_node: &Self::Node, out_node: &mut Self::Node);

    // Local to Local
    fn l2l(in_node: &Self::Node, out_node: &mut Self::Node);

    // Multipole to Particle
    fn m2p(in_node: &Self::Node, out_node: &mut Self::Node);

    // Local to Particle
    fn l2p(node: &mut Self::Node);
}
