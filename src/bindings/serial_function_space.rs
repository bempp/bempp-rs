use c_api_tools::{cfuncs, concretise_types, DType, DTypeIdentifier};
use ndelement::{
    ciarlet::CiarletElement,
    traits::{ElementFamily, FiniteElement},
    types::ReferenceCellType,
};

use rlst::prelude::{c32, c64};

use ndelement::bindings::ciarlet::{CiarletElementT, ElementFamilyT};
use ndelement::ciarlet::{
    LagrangeElementFamily, NedelecFirstKindElementFamily, RaviartThomasElementFamily,
};

use ndgrid::bindings::grid::GridT;

use ndgrid::{traits::Grid, types::RealScalar, SingleElementGrid};
use rlst::{MatrixInverse, RlstScalar};

use crate::function::SerialFunctionSpace;

#[cfuncs(name = "serial_function_space_t", create, free, unwrap)]
pub struct SerialFunctionSpaceT;

#[concretise_types(
    gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
    field(arg = 0, name = "grid", wrapper = "GridT",
        replace_with = ["SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"]),
    field(arg = 1, name = "e_family", wrapper = "ElementFamilyT",
        replace_with = ["LagrangeElementFamily<{{dtype}}>",
            "RaviartThomasElementFamily<{{dtype}}>",
            "NedelecFirstKindElementFamily<{{dtype}}>"]),
)]
pub fn function_space<
    T: RlstScalar + MatrixInverse,
    GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
    F: ElementFamily<T = T, CellType = ReferenceCellType, FiniteElement = CiarletElement<T>>,
>(
    grid: &'static GridImpl,
    element_family: &'static F,
) -> *const SerialFunctionSpaceT
where
    T::Real: RealScalar + DTypeIdentifier,
{
    let wrapper = serial_function_space_t_create();
    let inner = unsafe { serial_function_space_t_unwrap(wrapper).unwrap() };
    *inner = Box::new(SerialFunctionSpace::new(grid, element_family));
    wrapper
}
