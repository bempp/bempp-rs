//! FMM traits
use crate::tree::{AttachedDataTree, Tree};
use crate::kernel::Kernel;

pub trait SourceTranslation {

    type Fmm: Fmm;

    fn p2m(&mut self, fmm: &Self::Fmm);

    fn m2m(&mut self, fmm: &Self::Fmm);
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

    // fn get_multipole_expansion<'a>(
    //     &'a self,
    //     key: &<Self::Tree as Tree>::NodeIndex,
    // ) -> Option<Self::Coefficients<'a>>;

    // fn set_multipole_expansion<'a>(
    //     &'a mut self,
    //     key: &<Self::Tree as Tree>::NodeIndex,
    //     data: &Self::Coefficients<'a>,
    // );

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

    type Tree: Tree;

    fn order(&self) -> usize;

    fn new<'a>(
        order: usize,
        alpha_inner: f64,
        alpha_outer: f64,
        kernel: Box<dyn Kernel<PotentialData = Vec<f64>>>,
        points: <Self::Tree as Tree>::PointSlice<'a>,
        point_data: <Self::Tree as Tree>::PointDataSlice<'a>,
        adaptive: bool,
        n_crit: Option<u64>,
        depth: Option<u64>,
    ) -> Self;
}
