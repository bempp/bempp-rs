use crate::function::{FunctionSpace, SerialFunctionSpace};
use c_api_tools::{cfuncs, concretise_types, DType, DTypeIdentifier};
use ndelement::{
    bindings::ciarlet::ElementFamilyT,
    ciarlet::{
        CiarletElement, LagrangeElementFamily, NedelecFirstKindElementFamily,
        RaviartThomasElementFamily,
    },
    traits::ElementFamily,
    types::ReferenceCellType,
};
use ndgrid::{
    bindings::GridT, traits::Grid, types::Ownership, SingleElementGrid, SingleElementGridBorrowed,
};
use rlst::{c32, c64, MatrixInverse, RlstScalar};

#[cfuncs(name = "space_t", create, free, unwrap)]
pub struct SpaceT;

#[concretise_types(
    gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
    field(arg = 0, name = "grid", wrapper = "GridT",
        replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
    field(arg = 1, name = "e_family", wrapper = "ElementFamilyT",
        replace_with = ["LagrangeElementFamily<{{dtype}}>",
            "RaviartThomasElementFamily<{{dtype}}>",
            "NedelecFirstKindElementFamily<{{dtype}}>"]),
)]
pub fn function_space<
    T: RlstScalar + MatrixInverse,
    F: ElementFamily<T = T, CellType = ReferenceCellType, FiniteElement = CiarletElement<T>>,
    GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
>(
    grid: &'static GridImpl,
    element_family: &'static F,
) -> *const SpaceT {
    let wrapper = space_t_create();
    let inner = unsafe { space_t_unwrap(wrapper).unwrap() };
    *inner = Box::new(SerialFunctionSpace::new(grid, element_family));
    wrapper
}

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

#[concretise_types(
    gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
    gen_type(name = "grid", replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
                    "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
    field(arg = 0, name = "space", wrapper = "SpaceT",
        replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"])
)]
pub fn space_dtype<T: RlstScalar + DTypeIdentifier, S: FunctionSpace<T = T>>(_space: &S) -> DType {
    <T as DTypeIdentifier>::dtype()
}
