//! FMM traits
use crate::tree::FmmNode;

pub trait Translation {
    type Node: Node;

    // Particle to Multipole
    fn p2m(node: &mut Self::FmmNode);

    // Multipole to Multipole
    fn m2m(in_node: &Self::FmmNode, out_node: &mut Self::FmmNode);

    // Multipole to Local
    fn m2l(in_node: &Self::FmmNode, out_node: &mut Self::FmmNode);

    // Local to Local
    fn l2l(in_node: &Self::FmmNode, out_node: &mut Self::FmmNode);

    // Multipole to Particle
    fn m2p(in_node: &Self::FmmNode, out_node: &mut Self::FmmNode);

    // Local to Particle
    fn l2p(node: &mut Self::FmmNode);
}
