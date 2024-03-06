//! Multipole to Local field translations for uniform and adaptive Kernel Indepenent FMMs
use bempp_field::types::SvdFieldTranslationKiFmm;
use bempp_traits::tree::FmmTree;
use itertools::Itertools;
use num::Float;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use bempp_field::helpers::ncoeffs_kifmm;
use bempp_traits::{
    field::{SourceToTarget, SourceToTargetData},
    kernel::Kernel,
    tree::Tree,
    types::Scalar,
};
use bempp_tree::types::single_node::SingleNodeTreeNew;

use crate::fmm::KiFmm;
use crate::traits::FmmScalar;

use rlst_dense::{
    array::{empty_array, Array},
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_array_from_slice2, rlst_dynamic_array2,
    traits::{MatrixSvd, MultIntoResize, RawAccess, RawAccessMut},
};

/// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
impl<T, U, V> SourceToTarget for KiFmm<V, SvdFieldTranslationKiFmm<U, T>, T, U>
where
    T: Kernel<T = U> + std::marker::Send + std::marker::Sync + Default,
    U: FmmScalar,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree<Tree = SingleNodeTreeNew<U>>,
{
    fn m2l(&self, _level: u64) {}

    fn p2l(&self, _level: u64) {}
}
