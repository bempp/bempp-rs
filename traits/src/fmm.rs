//! FMM traits
use crate::kernel::Kernel;
use crate::tree::{AttachedDataTree, Tree};

pub trait SourceTranslation {
    fn p2m(&self);

    fn m2m(&self, level: u64);
}

pub trait TargetTranslation {
    fn m2l(&self, level: u64);

    fn l2l(&self, level: u64);

    fn l2p(&self);

    fn m2p(&self);

    fn p2l(&self);

    fn p2p(&self);
}

pub trait Fmm {
    type Kernel: Kernel;
    type Tree: Tree;

    fn order(&self) -> usize;

    fn kernel(&self) -> &Self::Kernel;

    fn tree(&self) -> &Self::Tree;
}

pub trait FmmLoop {
    fn upward_pass(&self);

    fn downward_pass(&self);

    fn run(&self);
}
