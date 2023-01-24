//! FMM traits

use crate::tree::FmmData;

pub trait Translation {
    type Node: FmmData;

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
