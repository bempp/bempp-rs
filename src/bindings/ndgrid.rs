use c_api_tools::DType;
use ndelement::types::ReferenceCellType;
use ndgrid::bindings::{grid::single_element_grid::single_element_grid_borrowed_create, GridT};
use std::ffi::c_void;

#[no_mangle]
pub unsafe extern "C" fn grid_t_free_bempp(ptr: *mut GridT) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn single_element_grid_borrowed_create_bempp(
    tdim: usize,
    id_sizes: *const usize,
    id_pointers: *const *const usize,
    entity_types: *const ReferenceCellType,
    entity_counts: *const usize,
    downward_connectivity: *const *const *const usize,
    downward_connectivity_shape0: *const *const usize,
    upward_connectivity: *const *const *const *const usize,
    upward_connectivity_lens: *const *const *const usize,
    points: *const c_void,
    gdim: usize,
    npoints: usize,
    dtype: DType,
    cells: *const usize,
    points_per_cell: usize,
    ncells: usize,
    geometry_degree: usize,
) -> *mut GridT {
    single_element_grid_borrowed_create(
        tdim,
        id_sizes,
        id_pointers,
        entity_types,
        entity_counts,
        downward_connectivity,
        downward_connectivity_shape0,
        upward_connectivity,
        upward_connectivity_lens,
        points,
        gdim,
        npoints,
        dtype,
        cells,
        points_per_cell,
        ncells,
        geometry_degree,
    )
}
