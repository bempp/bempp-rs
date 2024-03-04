use std::{char, collections::HashMap, thread::LocalKey};

use bempp_field::{
    fft::Fft,
    types::{FftFieldTranslationKiFmm, FftFieldTranslationKiFmmNew, SvdFieldTranslationKiFmm},
};
use bempp_traits::{
    field::{SourceToTarget, SourceToTargetData, SourceToTargetHomogenousScaleInvariant},
    fmm::{Fmm, NewFmm, SourceTranslation, TargetTranslation},
    kernel::{self, HomogenousKernel, Kernel},
    tree::{FmmTree, MortonKeyInterface, Tree},
    types::EvalType,
};
use bempp_tree::{
    constants::ROOT,
    types::{
        domain::Domain,
        morton::MortonKey,
        single_node::{SingleNodeTree, SingleNodeTreeNew},
    },
};
use cauchy::Scalar;
use num::{traits::real::Real, Complex, Float};
use rlst_blis::interface::gemm::Gemm;
use rlst_dense::{
    array::{empty_array, Array},
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_dynamic_array2,
    traits::{MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, Shape},
};

use crate::{
    charge::{Charges, Coordinates},
    constants::{ALPHA_INNER, ALPHA_OUTER},
    field_translation::target,
    pinv::pinv,
    types::{C2EType, SendPtr, SendPtrMut},
};

#[derive(Default)]
pub struct SingleNodeFmmTree<T: Float + Default + Scalar<Real = T>> {
    pub source_tree: SingleNodeTreeNew<T>,
    pub target_tree: SingleNodeTreeNew<T>,
    pub domain: Domain<T>,
}

impl<T> FmmTree for SingleNodeFmmTree<T>
where
    T: Float + Default + Scalar<Real = T> + Send + Sync,
{
    type Precision = T;
    type Tree = SingleNodeTreeNew<T>;

    fn get_source_tree(&self) -> &Self::Tree {
        &self.target_tree
    }

    fn get_target_tree(&self) -> &Self::Tree {
        &self.source_tree
    }

    fn get_domain(&self) -> &<Self::Tree as Tree>::Domain {
        &self.domain
    }
}

unsafe impl<T: Float + Default + Scalar<Real = T>> Send for SingleNodeFmmTree<T> {}
unsafe impl<T: Float + Default + Scalar<Real = T>> Sync for SingleNodeFmmTree<T> {}
