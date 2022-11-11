//! Container to make working with arrays easier.
//!
//! This module defines a type [DataContainer], which
//! is a simple C data structure that can be flexibly converted to
//! various Rust types.

use crate::{assert_dtype, get_itemsize, ConversionType, DTYPE, MUTABILITY, OWNERSHIP};
use libc::{c_void, size_t};

/// A data container for communication with a C ABI.
///
/// The basic task of this container is to provide a C ABI
/// compatible type to store arbitrary data arrays, and
/// to convert them back and forth into corresponding Rust types.
pub struct RustyDataContainer {
    /// The number of elements in the Array.
    nitems: size_t,
    /// The size in bytes of each element.
    itemsize: size_t,
    /// The capacity of the underlying array.
    /// This is only needed if the container is allocated
    /// from a Rust Vec.
    capacity: size_t,
    /// The type of the data.
    dtype: DTYPE,
    /// The ownership of the data. It is either
    /// [OWNERSHIP::Owner] or [OWNERSHIP::NotOwner].
    /// The underlying data can only be destroyed if
    /// [DataContainer] is owner.
    is_owner: OWNERSHIP,
    /// Mutability of the underlying data. It is either
    /// [MUTABILITY::Mutable] or [MUTABILITY::NotMutable].
    is_mutable: MUTABILITY,
    /// A pointer to the underlying data.
    data: *mut c_void,
}

impl RustyDataContainer {
    /// Create a new non-owning and non-mutable container from a given slice.
    pub fn from_slice<T: ConversionType>(slice: &[T]) -> Self {
        Self {
            nitems: slice.len(),
            capacity: slice.len(),
            itemsize: crate::get_size::<T>(),
            dtype: crate::get_dtype::<T>(),
            is_owner: OWNERSHIP::NotOwner,
            is_mutable: MUTABILITY::NotMutable,
            data: slice.as_ptr() as *mut c_void,
        }
    }
    /// Create a new non-owning but mutable container from a given slice.
    pub fn from_slice_mut<T: ConversionType>(slice: &mut [T]) -> Self {
        Self {
            nitems: slice.len(),
            capacity: slice.len(),
            itemsize: crate::get_size::<T>(),
            dtype: crate::get_dtype::<T>(),
            is_owner: OWNERSHIP::NotOwner,
            is_mutable: MUTABILITY::Mutable,
            data: slice.as_ptr() as *mut c_void,
        }
    }

    // To boxed pointer.
    pub fn to_box(self) -> Box<RustyDataContainer> {
        Box::new(self)
    }

    pub unsafe fn to_vec<T: ConversionType>(mut self) -> Vec<T> {
        assert_eq!(self.is_owner, OWNERSHIP::Owner);
        assert_dtype::<T>(self.dtype);
        // Have to remove ownership as the Vec takes ownership of the
        // contained data.
        self.is_owner = OWNERSHIP::NotOwner;
        Vec::<T>::from_raw_parts(self.data as *mut T, self.nitems, self.capacity)
    }

    /// Get a mutable reference to a RustyDataContainer from a ptr.
    /// Ensures that the destructor of the data container is not run.
    pub fn leak_mut(ptr: Option<Box<RustyDataContainer>>) -> &'static mut RustyDataContainer {
        let ptr_ref = Box::leak(ptr.unwrap());
        assert_eq!(ptr_ref.is_mutable, MUTABILITY::Mutable);
        ptr_ref
    }

    /// Get a reference to a RustyDataContainer from a ptr.
    /// Ensures that the destructor of the data container is not run.
    pub fn leak(ptr: Option<Box<RustyDataContainer>>) -> &'static RustyDataContainer {
        Box::leak(ptr.unwrap())
    }

    /// Create a new owning and mutable container from a vector.
    /// The vector is consumed by this method.
    pub fn from_vec<T: ConversionType>(vec: Vec<T>) -> Self {
        let nitems = vec.len();
        let capacity = vec.capacity();
        let data = vec.as_ptr() as *mut c_void;
        std::mem::forget(vec);
        Self {
            nitems,
            capacity,
            itemsize: crate::get_size::<T>(),
            dtype: crate::get_dtype::<T>(),
            is_owner: OWNERSHIP::Owner,
            is_mutable: MUTABILITY::Mutable,
            data,
        }
    }

    /// Get a representation of the data as slice.
    /// This method does not take ownership of the container associated with `ptr`.
    pub unsafe fn as_slice<T: ConversionType>(
        ptr: Option<Box<RustyDataContainer>>,
    ) -> &'static [T] {
        let container = RustyDataContainer::leak(ptr);
        assert_dtype::<T>(container.dtype);
        std::slice::from_raw_parts::<'static, T>(container.data as *const T, container.nitems)
    }

    /// Get a representation of the data as mutable slice.
    /// This method does not take ownership of the container associated with `ptr`.
    pub unsafe fn as_slice_mut<T: ConversionType>(
        ptr: Option<Box<RustyDataContainer>>,
    ) -> &'static mut [T] {
        let container = RustyDataContainer::leak_mut(ptr);
        assert_eq!(container.is_mutable, MUTABILITY::Mutable);
        std::slice::from_raw_parts_mut::<'static, T>(container.data as *mut T, container.nitems)
    }
}

impl Drop for RustyDataContainer {
    /// Destroy a data container. If the container owns the
    /// data the corresponding memory is also deallocated.
    fn drop(&mut self) {
        if let OWNERSHIP::Owner = self.is_owner {
            let len = self.nitems * self.itemsize;
            let cap = self.capacity * self.itemsize;
            let vec = unsafe { Vec::<u8>::from_raw_parts(self.data as *mut u8, len, cap) };
            drop(vec);
        }
    }
}

/// Destroy a data container.
#[no_mangle]
pub extern "C" fn rusty_data_container_destroy(_: Option<Box<RustyDataContainer>>) {}

/// Create a new f32 data container.
#[no_mangle]
pub extern "C" fn rusty_data_container_new_f32(nitems: size_t) -> Box<RustyDataContainer> {
    RustyDataContainer::from_vec(vec![0 as f32; nitems]).to_box()
}

/// Create a new f64 data container.
#[no_mangle]
pub extern "C" fn rusty_data_container_new_f64(nitems: size_t) -> Box<RustyDataContainer> {
    RustyDataContainer::from_vec(vec![0 as f64; nitems]).to_box()
}

/// Create a new u8 data container.
#[no_mangle]
pub extern "C" fn rusty_data_container_new_u8(nitems: size_t) -> Box<RustyDataContainer> {
    RustyDataContainer::from_vec(vec![0 as u8; nitems]).to_box()
}

/// Create a new u32 data container.
#[no_mangle]
pub extern "C" fn rusty_data_container_new_u32(nitems: size_t) -> Box<RustyDataContainer> {
    RustyDataContainer::from_vec(vec![0 as u32; nitems]).to_box()
}

/// Create a new u64 data container.
#[no_mangle]
pub extern "C" fn rusty_data_container_new_u64(nitems: size_t) -> Box<RustyDataContainer> {
    RustyDataContainer::from_vec(vec![0 as u64; nitems]).to_box()
}

/// Create a new i8 data container.
#[no_mangle]
pub extern "C" fn rusty_data_container_new_i8(nitems: size_t) -> Box<RustyDataContainer> {
    RustyDataContainer::from_vec(vec![0 as i8; nitems]).to_box()
}

/// Create a new i32 data container.
#[no_mangle]
pub extern "C" fn rusty_data_container_new_i32(nitems: size_t) -> Box<RustyDataContainer> {
    RustyDataContainer::from_vec(vec![0 as i32; nitems]).to_box()
}

/// Create a new i64 data container.
#[no_mangle]
pub extern "C" fn rusty_data_container_new_i64(nitems: size_t) -> Box<RustyDataContainer> {
    RustyDataContainer::from_vec(vec![0 as i64; nitems]).to_box()
}

/// Create a new usize data container.
#[no_mangle]
pub extern "C" fn rusty_data_container_new_usize(nitems: size_t) -> Box<RustyDataContainer> {
    RustyDataContainer::from_vec(vec![0 as usize; nitems]).to_box()
}

/// Get nitems
#[no_mangle]
pub extern "C" fn rusty_data_container_get_nitems(ptr: Option<Box<RustyDataContainer>>) -> size_t {
    RustyDataContainer::leak(ptr).nitems
}

/// Get itemsize
#[no_mangle]
pub extern "C" fn rusty_data_container_get_itemsize(
    ptr: Option<Box<RustyDataContainer>>,
) -> size_t {
    RustyDataContainer::leak(ptr).itemsize
}

/// Get dtype
#[no_mangle]
pub extern "C" fn rusty_data_container_get_dtype(ptr: Option<Box<RustyDataContainer>>) -> DTYPE {
    RustyDataContainer::leak(ptr).dtype
}

/// Get is_owner
#[no_mangle]
pub extern "C" fn rusty_data_container_get_is_owner(
    ptr: Option<Box<RustyDataContainer>>,
) -> OWNERSHIP {
    RustyDataContainer::leak(ptr).is_owner
}

/// Get is_mutable
#[no_mangle]
pub extern "C" fn rusty_data_container_get_is_mutable(
    ptr: Option<Box<RustyDataContainer>>,
) -> MUTABILITY {
    RustyDataContainer::leak(ptr).is_mutable
}

/// Get data
#[no_mangle]
pub extern "C" fn rusty_data_container_get_data(
    ptr: Option<Box<RustyDataContainer>>,
) -> *mut c_void {
    RustyDataContainer::leak(ptr).data
}

#[no_mangle]
pub extern "C" fn new_from_pointer(
    ptr: *mut c_void,
    nitems: size_t,
    dtype: DTYPE,
    is_mutable: MUTABILITY,
) -> Box<RustyDataContainer> {
    RustyDataContainer {
        nitems,
        capacity: nitems,
        itemsize: get_itemsize(dtype) as size_t,
        dtype,
        is_owner: OWNERSHIP::NotOwner,
        is_mutable,
        data: ptr,
    }
    .to_box()
}
