//! FMM traits
use crate::kernel::Kernel;
use crate::tree::{AttachedDataTree, Tree};

pub trait SourceTranslation {
    fn p2m(&self);

    fn m2m(&self, level: u64);
}

pub trait TargetTranslation {
    fn m2l(&self);

    fn l2l(&self);

    fn l2p(&self);

    fn p2l(&self);

    fn p2p(&self);
}

pub trait SourceDataTree {
    type Tree: Tree;
    type Coefficient;
    type Coefficients<'a>: IntoIterator<Item = &'a Self::Coefficient>
    where
        Self: 'a;

    fn get_multipole_expansion<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Coefficients<'a>>;

    fn set_multipole_expansion<'a>(
        &'a mut self,
        key: &<Self::Tree as Tree>::NodeIndex,
        data: &Self::Coefficients<'a>,
    );

    fn get_points<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::PointSlice<'a>>;
}

pub trait TargetDataTree {
    type Tree: Tree;
    type Potential;
    type Coefficient;
    type Coefficients<'a>: IntoIterator<Item = &'a Self::Coefficient>
    where
        Self: 'a;
    type Potentials<'a>: IntoIterator<Item = &'a Self::Potential>
    where
        Self: 'a;

    fn get_local_expansion<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Coefficients<'a>>;

    fn set_local_expansion<'a>(
        &'a mut self,
        key: &<Self::Tree as Tree>::NodeIndex,
        data: &Self::Coefficients<'a>,
    );

    fn get_potentials<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Potentials<'a>>;

    fn set_potentials<'a>(
        &'a mut self,
        key: &<Self::Tree as Tree>::NodeIndex,
        data: &Self::Potentials<'a>,
    );
}

pub trait Fmm {
    type Kernel: Kernel;
    type Tree: Tree;

    fn order(&self) -> usize;

    fn kernel(&self) -> &Self::Kernel;

    fn tree(&self) -> &Self::Tree;
}

pub trait FmmAlgorithm {

    fn upward_pass(&self);

    // fn downward_pass(&self);

    fn run(&self);
}