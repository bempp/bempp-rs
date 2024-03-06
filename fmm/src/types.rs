//! Data structures FMM data and metadata.
use std::collections::HashMap;

use bempp_traits::{field::SourceToTargetData, kernel::Kernel, tree::Tree};
use bempp_tree::types::morton::MortonKey;
use num::{Complex, Float};
use rlst_common::types::Scalar;
use rlst_dense::{array::Array, base_array::BaseArray, data_container::VectorContainer};

/// Type alias for charge data
pub type Charge<T> = T;

/// Type alias for global index for identifying charge data with a point
pub type GlobalIdx = usize;

/// Type alias for mapping charge data to global indices.
pub type ChargeDict<T> = HashMap<GlobalIdx, Charge<T>>;

/// Type alias for approximation of FMM operator matrices.
pub type C2EType<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// A threadsafe mutable raw pointer
#[derive(Clone, Debug, Copy)]
pub struct SendPtrMut<T> {
    pub raw: *mut T,
}

unsafe impl<T> Sync for SendPtrMut<T> {}
unsafe impl<T> Send for SendPtrMut<Complex<T>> {}

impl<T> Default for SendPtrMut<T> {
    fn default() -> Self {
        SendPtrMut {
            raw: std::ptr::null_mut(),
        }
    }
}

/// A threadsafe raw pointer
#[derive(Clone, Debug, Copy)]
pub struct SendPtr<T> {
    pub raw: *const T,
}

unsafe impl<T> Sync for SendPtr<T> {}

impl<T> Default for SendPtr<T> {
    fn default() -> Self {
        SendPtr {
            raw: std::ptr::null(),
        }
    }
}
