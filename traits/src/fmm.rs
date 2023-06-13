//! FMM traits
use crate::kernel::Kernel;
use crate::tree::Tree;

pub trait SourceTranslation {
    fn p2m(&self);

    fn m2m(&self, level: u64);
}

pub trait TargetTranslation {
    fn m2l_batched(&self, level: u64);

    // V list (far field) interactions
    fn m2l(&self, level: u64);

    // Translate local potential from parent to child.
    fn l2l(&self, level: u64);

    // W list interactions.
    fn m2p(&self);

    // X list interactions.
    fn p2l(&self);

    // Translate local expansion to points within a leaf.
    fn l2p(&self);

    // U list (near field) interactions.
    fn p2p(&self);
}

pub trait Fmm {
    type Kernel: Kernel;
    type Tree: Tree;

    // Expansion order.
    fn order(&self) -> usize;

    fn kernel(&self) -> &Self::Kernel;

    fn tree(&self) -> &Self::Tree;
}

pub trait FmmLoop {
    fn upward_pass(&self);

    fn downward_pass(&self);

    fn run(&self);
}

pub trait InteractionLists {
    type Tree: Tree;

    fn get_v_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;
    fn get_x_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;
    fn get_w_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;
    fn get_u_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;
}
