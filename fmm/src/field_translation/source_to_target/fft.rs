//! Multipole to Local field translations for uniform and adaptive Kernel Indepenent FMMs
use bempp_field::types::FftFieldTranslationKiFmm;
use bempp_traits::tree::FmmTree;
use bempp_tree::types::single_node::SingleNodeTreeNew;
use itertools::Itertools;
use num::{Complex, Float};
use rayon::prelude::*;
use rlst_blis::interface::gemm::Gemm;
use rlst_dense::array::Array;
use rlst_dense::base_array::BaseArray;
use rlst_dense::data_container::VectorContainer;
use std::collections::HashSet;

use bempp_field::fft::Fft;

use bempp_field::helpers::ncoeffs_kifmm;
use bempp_traits::{
    field::{SourceToTarget, SourceToTargetData},
    fmm::InteractionLists,
    kernel::Kernel,
    tree::Tree,
    types::{EvalType, Scalar},
};
use bempp_tree::types::morton::MortonKey;

use crate::helpers::find_chunk_size;
use crate::{fmm::KiFmm, traits::FmmScalar};
use rlst_dense::{
    array::empty_array,
    rlst_dynamic_array2,
    traits::{MultIntoResize, RawAccess, RawAccessMut},
};

use rlst_dense::traits::{MatrixSvd, RandomAccessMut};

use crate::field_translation::hadamard::matmul8x8;

impl<T, U, V> SourceToTarget for KiFmm<V, FftFieldTranslationKiFmm<U, T>, T, U>
where
    T: Kernel<T = U> + std::marker::Send + std::marker::Sync + Default,
    U: FmmScalar,
    Complex<U>: Scalar,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree<Tree = SingleNodeTreeNew<U>>,
{
    fn m2l(&self, level: u64) {}

    fn p2l(&self, level: u64) {}
}
