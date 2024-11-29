//! Bindings for C

#![allow(missing_docs)]
#![allow(clippy::missing_safety_doc)]

use c_api_tools::cfuncs;

pub mod serial_function_space;

#[cfuncs(name = "space_t", create, free, unwrap)]
pub struct SpaceT;

pub mod ndelement {
    use c_api_tools::{concretise_types, DType};

    use rlst::prelude::{c32, c64};

    use ndelement::{
        bindings::{ciarlet as ciarlet_b, ciarlet::ElementFamilyT},
        ciarlet::{
            CiarletElement, LagrangeElementFamily, NedelecFirstKindElementFamily,
            RaviartThomasElementFamily,
        },
        traits::ElementFamily,
        types::{Continuity, ReferenceCellType},
    };

    use ndgrid::{traits::Grid, types::RealScalar, SingleElementGrid};
    use rlst::{MatrixInverse, RlstScalar};

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
}

pub mod ndgrid {
    use c_api_tools::{concretise_types, DType};
    use ndelement::{ciarlet::CiarletElement, types::ReferenceCellType};
    use ndgrid::{
        bindings::{grid::single_element_grid::single_element_grid_borrowed_create, GridT},
        traits::Grid,
        SingleElementGrid, SingleElementGridBorrowed,
    };
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
}

pub mod space {
    use super::{space_t_create, space_t_unwrap, SpaceT};
    use crate::function::{FunctionSpace, SerialFunctionSpace};
    use c_api_tools::concretise_types;
    use ndelement::ciarlet::CiarletElement;
    use ndgrid::{types::Ownership, SingleElementGrid, SingleElementGridBorrowed};
    use rlst::{c32, c64, RlstScalar};

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
        field(arg = 0, name = "space", wrapper = "SpaceT",
            replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
    )]
    pub fn space_local_space<S: FunctionSpace>(space: &'static S) -> *mut SpaceT {
        let wrapper = space_t_create();
        let inner = unsafe { space_t_unwrap(wrapper).unwrap() };
        *inner = Box::new(space.local_space());
        wrapper
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
        field(arg = 0, name = "space", wrapper = "SpaceT",
            replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
    )]
    pub fn space_is_serial<S: FunctionSpace>(space: &S) -> bool {
        space.is_serial()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
        field(arg = 0, name = "space", wrapper = "SpaceT",
            replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
    )]
    pub fn space_get_local_dof_numbers_size<S: FunctionSpace>(
        space: &S,
        entity_dim: usize,
        entity_number: usize,
    ) -> usize {
        space.get_local_dof_numbers(entity_dim, entity_number).len()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
        field(arg = 0, name = "space", wrapper = "SpaceT",
            replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
    )]
    pub fn space_get_local_dof_numbers<S: FunctionSpace>(
        space: &S,
        entity_dim: usize,
        entity_number: usize,
        dofs: *mut usize,
    ) {
        for (i, j) in space
            .get_local_dof_numbers(entity_dim, entity_number)
            .iter()
            .enumerate()
        {
            unsafe {
                *dofs.add(i) = *j;
            }
        }
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
        field(arg = 0, name = "space", wrapper = "SpaceT",
            replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
    )]
    pub fn space_local_size<S: FunctionSpace>(space: &S) -> usize {
        space.local_size()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
        field(arg = 0, name = "space", wrapper = "SpaceT",
            replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
    )]
    pub fn space_global_size<S: FunctionSpace>(space: &S) -> usize {
        space.global_size()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
        field(arg = 0, name = "space", wrapper = "SpaceT",
            replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
    )]
    pub fn space_has_cell_dofs<S: FunctionSpace>(space: &S, cell: usize) -> bool {
        space.cell_dofs(cell).is_some()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
        field(arg = 0, name = "space", wrapper = "SpaceT",
            replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
    )]
    pub fn space_cell_dofs_size<S: FunctionSpace>(space: &S, cell: usize) -> usize {
        space.cell_dofs(cell).unwrap().len()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
        field(arg = 0, name = "space", wrapper = "SpaceT",
            replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
    )]
    pub fn space_cell_dofs<S: FunctionSpace>(space: &S, cell: usize, dofs: *mut usize) {
        for (i, j) in space.cell_dofs(cell).unwrap().iter().enumerate() {
            unsafe {
                *dofs.add(i) = *j;
            }
        }
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
        field(arg = 0, name = "space", wrapper = "SpaceT",
            replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
    )]
    pub fn space_global_dof_index<S: FunctionSpace>(space: &S, local_dof_index: usize) -> usize {
        space.global_dof_index(local_dof_index)
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
        field(arg = 0, name = "space", wrapper = "SpaceT",
            replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
    )]
    pub fn space_is_owned<S: FunctionSpace>(space: &S, local_dof_index: usize) -> bool {
        space.ownership(local_dof_index) == Ownership::Owned
    }
}
