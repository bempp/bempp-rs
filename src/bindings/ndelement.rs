use c_api_tools::DType;

use ndelement::{
    bindings::{ciarlet as ciarlet_b, ciarlet::ElementFamilyT},
    types::Continuity,
};

#[no_mangle]
pub unsafe extern "C" fn element_family_t_free_bempp(ptr: *mut ElementFamilyT) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub extern "C" fn create_lagrange_family_bempp(
    degree: usize,
    continuity: Continuity,
    dtype: DType,
) -> *mut ElementFamilyT {
    ciarlet_b::create_lagrange_family(degree, continuity, dtype)
}

#[no_mangle]
pub extern "C" fn create_raviart_thomas_family_bempp(
    degree: usize,
    continuity: Continuity,
    dtype: DType,
) -> *mut ElementFamilyT {
    ciarlet_b::create_raviart_thomas_family(degree, continuity, dtype)
}

#[no_mangle]
pub extern "C" fn create_nedelec_family_bempp(
    degree: usize,
    continuity: Continuity,
    dtype: DType,
) -> *mut ElementFamilyT {
    ciarlet_b::create_nedelec_family(degree, continuity, dtype)
}
