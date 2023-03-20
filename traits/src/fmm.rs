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

    fn get_multipole_expansion(
        &self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
    );

    fn set_multipole_expansion(
        &self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
    );

    fn get_particles(&self, key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex);

    fn get_charges(&self, key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex);
}

pub trait TargetDataTree {
    type DataTree: AttachedDataTree;

    fn get_local_expansion(
        &self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
    );

    fn set_local_expansion(
        &self,
        key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex,
    );

    fn get_particles(&self, key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex);

    fn get_potentials(&self, key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex);

    fn set_potentials(&self, key: &<<Self::DataTree as AttachedDataTree>::Tree as Tree>::NodeIndex);
}
