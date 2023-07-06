use std::{
    sync::{Arc, Mutex},
    collections::HashMap
};

use bempp_traits::{
    field::FieldTranslationData,
    kernel::Kernel,
    tree::Tree,
    fmm::{Fmm}
};
use bempp_tree::{
    types::{morton::MortonKey, point::Point}
};
use rlst::dense::{base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix};
use rlst::dense::{traits::*};
use rlst::{
    self,
};



#[derive(Clone, Debug, Default)]
pub struct Charge {
    /// Charge data
    pub data: f64,

    /// Global unique index.
    pub global_idx: usize,
}

/// Container of **Points**.
#[derive(Clone, Debug, Default)]
pub struct Charges {
    /// A vector of Charges
    pub charges: Vec<Charge>,

    /// index for implementing the Iterator trait.
    pub index: usize,
}

pub type Expansions =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

pub type Potentials =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

pub type C2EType =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

pub struct FmmData<T: Fmm> {
    pub fmm: Arc<T>,
    pub multipoles: HashMap<MortonKey, Arc<Mutex<Expansions>>>,
    pub locals: HashMap<MortonKey, Arc<Mutex<Expansions>>>,
    pub potentials: HashMap<MortonKey, Arc<Mutex<Potentials>>>,
    pub points: HashMap<MortonKey, Vec<Point>>,
    pub charges: HashMap<MortonKey, Arc<Vec<f64>>>,
}

pub struct KiFmm<T: Tree, U: Kernel, V: FieldTranslationData<U>> {
    pub order: usize,

    pub uc2e_inv: C2EType,

    pub dc2e_inv: C2EType,

    pub alpha_inner: f64,
    pub alpha_outer: f64,

    pub m2m: Vec<C2EType>,
    pub l2l: Vec<C2EType>,
    pub tree: T,
    pub kernel: U,
    pub m2l: V,
}