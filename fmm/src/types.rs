//! Data structures FMM data and metadata.
use num::Complex;
use rlst_dense::{array::Array, base_array::BaseArray, data_container::VectorContainer};

pub type Charges<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

pub type Coordinates<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

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
