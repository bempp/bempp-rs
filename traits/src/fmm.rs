//! FMM traits
use crate::tree::{AttachedDataTree, Tree};

pub trait SourceTranslation {
    fn p2m(&self);

    fn m2m(&self);
}

pub trait TargetTranslation {
    fn m2l(&self);

    fn l2l(&self);

    fn l2p(&self);

    fn p2l(&self);

    fn p2p(&self);
}

pub trait SourceDataTree {
    type DataTree: AttachedDataTree;
    type Coefficient;
    type Charge;
    type Coefficients<'a>: IntoIterator<Item = &'a Self::Coefficient>
    where
        Self: 'a;
    type Charges<'a>: IntoIterator<Item = &'a Self::Charge>
    where
        Self: 'a;

    fn get_multipole_expansion<'a>(
        &'a self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Coefficients<'a>>;

    fn set_multipole_expansion<'a>(
        &'a self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
        coefficients: &Self::Coefficients<'a>,
    );

    fn get_points<'a>(
        &'a self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
    ) -> Option<<<Self::DataTree as AttachedDataTree>::Tree as Tree>::PointSlice<'a>>;

    fn set_charges<'a>(
        &'a self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
        charges: &Self::Charges<'a>,
    );

    fn get_charges<'a>(
        &'a self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Charges<'a>>;
}

pub trait TargetDataTree {
    type DataTree: AttachedDataTree;
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
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Coefficients<'a>>;

    fn set_local_expansion<'a>(
        &'a self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
        coefficients: &Self::Coefficients<'a>,
    );

    fn get_points<'a>(
        &'a self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
    ) -> Option<<<Self::DataTree as AttachedDataTree>::Tree as Tree>::PointSlice<'a>>;

    fn get_potentials<'a>(
        &'a self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Potentials<'a>>;

    fn set_potentials<'a>(
        &'a self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
        potentials: &Self::Potentials<'a>,
    );
}

pub trait Fmm {
    type Charges;
    type Potentials;
    type SourceDataTree: SourceDataTree;
    type TargetDataTree: TargetDataTree;
    type PartitionTree: Tree;

    fn init<'a>(
        &'a self,
        points: <Self::PartitionTree as Tree>::PointSlice<'a>,
        charges: <Self::SourceDataTree as SourceDataTree>::Charges<'a>,
    );

    fn upward_pass(&self);

    fn downard_pass(&self);

    fn run(&self);
}
