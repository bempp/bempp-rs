use c_api_tools::concretise_types;
use ndelement::{
    bindings::ciarlet::{CiarletElementT, ElementFamilyT},
    ciarlet::{
        CiarletElement, LagrangeElementFamily, NedelecFirstKindElementFamily,
        RaviartThomasElementFamily,
    },
    traits::{ElementFamily, FiniteElement},
    types::ReferenceCellType,
};

use ndgrid::{
    bindings::GridT, traits::Grid, types::RealScalar, SingleElementGrid, SingleElementGridBorrowed,
};
use rlst::{c32, c64, MatrixInverse, RlstScalar};

use super::{space_t_create, space_t_unwrap, SpaceT};
use crate::function::SerialFunctionSpace;

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
    GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
    F: ElementFamily<T = T, CellType = ReferenceCellType, FiniteElement = CiarletElement<T>>,
>(
    grid: &'static GridImpl,
    element_family: &'static F,
) -> *const SpaceT
where
    T::Real: RealScalar,
{
    let wrapper = space_t_create();
    let inner = unsafe { space_t_unwrap(wrapper).unwrap() };
    *inner = Box::new(SerialFunctionSpace::new(grid, element_family));
    wrapper
}
