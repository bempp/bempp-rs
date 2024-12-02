use super::{matrix_t_create, matrix_t_unwrap, space::SpaceT, MatrixT};
use crate::{boundary_assemblers::BoundaryAssemblerOptions, helmholtz, laplace};
use crate::{
    boundary_assemblers::{integrands::BoundaryIntegrand, BoundaryAssembler},
    function::{FunctionSpace, SerialFunctionSpace},
};
use c_api_tools::{cfuncs, concretise_types, DType, DTypeIdentifier};
use green_kernels::traits::Kernel;
use ndelement::{ciarlet::CiarletElement, types::ReferenceCellType};
use ndgrid::{SingleElementGrid, SingleElementGridBorrowed};
use rlst::{c32, c64, MatrixInverse, RlstScalar};
use std::ffi::c_void;
use std::slice::from_raw_parts_mut;

#[repr(u8)]
#[derive(Debug)]
pub enum OperatorType {
    SingleLayer,
    DoubleLayer,
    AdjointDoubleLayer,
    Hypersingular,
    ElectricField,
    MagneticField,
}

#[cfuncs(name = "boundary_assembler_t", create, free, unwrap)]
pub struct BoundaryAssemblerT;

//#[cfuncs(name = "boundary_assembler_options_t", create, free, unwrap)]
pub struct BoundaryAssemblerOptionsT {
    options: BoundaryAssemblerOptions,
}

#[no_mangle]
pub extern "C" fn boundary_assembler_options_new() -> *const BoundaryAssemblerOptionsT {
    let obj = BoundaryAssemblerOptionsT {
        options: BoundaryAssemblerOptions::default(),
    };
    Box::into_raw(Box::new(obj)) as _
}

#[no_mangle]
pub unsafe extern "C" fn boundary_assembler_options_t_free(ptr: *mut BoundaryAssemblerOptionsT) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn boundary_assembler_options_set_regular_quadrature_degree(
    options: *mut BoundaryAssemblerOptionsT,
    cell_type: ReferenceCellType,
    npoints: usize,
) {
    let options = &mut (*options).options;
    options.set_regular_quadrature_degree(cell_type, npoints);
}
#[no_mangle]
pub unsafe extern "C" fn boundary_assembler_options_get_regular_quadrature_degree(
    options: *mut BoundaryAssemblerOptionsT,
    cell_type: ReferenceCellType,
) -> usize {
    let options = &(*options).options;
    options.get_regular_quadrature_degree(cell_type).unwrap()
}
#[no_mangle]
pub unsafe extern "C" fn boundary_assembler_options_has_regular_quadrature_degree(
    options: *mut BoundaryAssemblerOptionsT,
    cell_type: ReferenceCellType,
) -> bool {
    let options = &(*options).options;
    options.get_regular_quadrature_degree(cell_type).is_some()
}

#[no_mangle]
pub unsafe extern "C" fn boundary_assembler_options_set_singular_quadrature_degree(
    options: *mut BoundaryAssemblerOptionsT,
    cell_type0: ReferenceCellType,
    cell_type1: ReferenceCellType,
    npoints: usize,
) {
    let options = &mut (*options).options;
    options.set_singular_quadrature_degree((cell_type0, cell_type1), npoints);
}
#[no_mangle]
pub unsafe extern "C" fn boundary_assembler_options_get_singular_quadrature_degree(
    options: *mut BoundaryAssemblerOptionsT,
    cell_type0: ReferenceCellType,
    cell_type1: ReferenceCellType,
) -> usize {
    let options = &(*options).options;
    options
        .get_singular_quadrature_degree((cell_type0, cell_type1))
        .unwrap()
}
#[no_mangle]
pub unsafe extern "C" fn boundary_assembler_options_has_singular_quadrature_degree(
    options: *mut BoundaryAssemblerOptionsT,
    cell_type0: ReferenceCellType,
    cell_type1: ReferenceCellType,
) -> bool {
    let options = &(*options).options;
    options
        .get_singular_quadrature_degree((cell_type0, cell_type1))
        .is_some()
}

#[no_mangle]
pub unsafe extern "C" fn boundary_assembler_options_set_batch_size(
    options: *mut BoundaryAssemblerOptionsT,
    batch_size: usize,
) {
    let options = &mut (*options).options;
    options.set_batch_size(batch_size);
}
#[no_mangle]
pub unsafe extern "C" fn boundary_assembler_options_get_batch_size(
    options: *mut BoundaryAssemblerOptionsT,
) -> usize {
    let options = &(*options).options;
    options.get_batch_size()
}

#[no_mangle]
pub unsafe extern "C" fn laplace_boundary_assembler_new(
    options: *const BoundaryAssemblerOptionsT,
    operator_type: OperatorType,
    dtype: DType,
) -> *const BoundaryAssemblerT {
    unsafe fn _new<T: RlstScalar<Real = T> + MatrixInverse>(
        options: *const BoundaryAssemblerOptionsT,
        operator_type: OperatorType,
    ) -> *const BoundaryAssemblerT {
        let options = &(*options).options;
        let wrapper = boundary_assembler_t_create();
        let inner = unsafe { boundary_assembler_t_unwrap(wrapper).unwrap() };
        match operator_type {
            OperatorType::SingleLayer => {
                *inner = Box::new(laplace::assembler::single_layer::<T>(options));
            }
            OperatorType::DoubleLayer => {
                *inner = Box::new(laplace::assembler::double_layer::<T>(options));
            }
            OperatorType::AdjointDoubleLayer => {
                *inner = Box::new(laplace::assembler::adjoint_double_layer::<T>(options));
            }
            OperatorType::Hypersingular => {
                *inner = Box::new(laplace::assembler::hypersingular::<T>(options));
            }
            _ => {
                panic!("Invalid operator for Laplace: {operator_type:?}");
            }
        }
        wrapper
    }
    match dtype {
        DType::F32 => _new::<f32>(options, operator_type),
        DType::F64 => _new::<f64>(options, operator_type),
        _ => {
            panic!("Invalid dtype for Laplace: {dtype:?}");
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn helmholtz_boundary_assembler_new(
    k: *const c_void,
    options: *const BoundaryAssemblerOptionsT,
    operator_type: OperatorType,
    dtype: DType,
) -> *const BoundaryAssemblerT {
    unsafe fn _new<T: RlstScalar<Complex = T> + MatrixInverse>(
        k: *const c_void,
        options: *const BoundaryAssemblerOptionsT,
        operator_type: OperatorType,
    ) -> *const BoundaryAssemblerT {
        let k = *(k as *const T::Real);
        let options = &(*options).options;
        let wrapper = boundary_assembler_t_create();
        let inner = unsafe { boundary_assembler_t_unwrap(wrapper).unwrap() };
        match operator_type {
            OperatorType::SingleLayer => {
                *inner = Box::new(helmholtz::assembler::single_layer::<T>(k, options));
            }
            OperatorType::DoubleLayer => {
                *inner = Box::new(helmholtz::assembler::double_layer::<T>(k, options));
            }
            OperatorType::AdjointDoubleLayer => {
                *inner = Box::new(helmholtz::assembler::adjoint_double_layer::<T>(k, options));
            }
            OperatorType::Hypersingular => {
                *inner = Box::new(helmholtz::assembler::hypersingular::<T>(k, options));
            }
            _ => {
                panic!("Invalid operator for Helmholtz: {operator_type:?}");
            }
        }
        wrapper
    }
    match dtype {
        DType::C32 => _new::<c32>(k, options, operator_type),
        DType::C64 => _new::<c64>(k, options, operator_type),
        _ => {
            panic!("Invalid dtype for Helmholtz: {dtype:?}");
        }
    }
}

#[concretise_types(
    gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
    field(arg = 0, name = "boundary_assembler", wrapper = "BoundaryAssemblerT", replace_with = [
        "laplace::assembler::SingleLayer3dAssembler<<{{dtype}} as RlstScalar>::Real>",
        "laplace::assembler::DoubleLayer3dAssembler<<{{dtype}} as RlstScalar>::Real>",
        "laplace::assembler::AdjointDoubleLayer3dAssembler<<{{dtype}} as RlstScalar>::Real>",
        "laplace::assembler::Hypersingular3dAssembler<<{{dtype}} as RlstScalar>::Real>",
        "helmholtz::assembler::SingleLayer3dAssembler<<{{dtype}} as RlstScalar>::Complex>",
        "helmholtz::assembler::DoubleLayer3dAssembler<<{{dtype}} as RlstScalar>::Complex>",
        "helmholtz::assembler::AdjointDoubleLayer3dAssembler<<{{dtype}} as RlstScalar>::Complex>",
        "helmholtz::assembler::Hypersingular3dAssembler<<{{dtype}} as RlstScalar>::Complex>",
    ])
)]
pub fn boundary_assembler_dtype<
    T: RlstScalar + MatrixInverse + DTypeIdentifier,
    Integrand: BoundaryIntegrand<T = T>,
    K: Kernel<T = T>,
>(
    _assembler: &BoundaryAssembler<T, Integrand, K>,
) -> DType {
    <T as DTypeIdentifier>::dtype()
}

#[concretise_types(
    gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
    gen_type(name = "grid", replace_with = [
        "SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"
    ]),
    gen_type(name = "space", replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"]),
    field(arg = 0, name = "boundary_assembler", wrapper = "BoundaryAssemblerT", replace_with = [
        "laplace::assembler::SingleLayer3dAssembler<{{dtype}}>",
        "laplace::assembler::DoubleLayer3dAssembler<{{dtype}}>",
        "laplace::assembler::AdjointDoubleLayer3dAssembler<{{dtype}}>",
        "laplace::assembler::Hypersingular3dAssembler<{{dtype}}>",
        // TODO: Helmholtz
//        "helmholtz::assembler::SingleLayer3dAssembler<{{dtype}}>",
//        "helmholtz::assembler::DoubleLayer3dAssembler<{{dtype}}>",
//        "helmholtz::assembler::AdjointDoubleLayer3dAssembler<{{dtype}}>",
//        "helmholtz::assembler::Hypersingular3dAssembler<{{dtype}}>",
    ]),
    field(arg = 1, name = "trial_space", wrapper = "SpaceT", replace_with = ["{{space}}"]),
    field(arg = 2, name = "test_space", wrapper = "SpaceT", replace_with = ["{{space}}"]),
)]
pub fn boundary_assembler_assemble_singular<
    T: RlstScalar + MatrixInverse,
    Integrand: BoundaryIntegrand<T = T>,
    K: Kernel<T = T>,
    Space: FunctionSpace<T = T>,
>(
    assembler: &BoundaryAssembler<T, Integrand, K>,
    trial_space: &Space,
    test_space: &Space,
) -> *mut MatrixT {
    let wrapper = matrix_t_create();
    let inner = unsafe { matrix_t_unwrap(wrapper).unwrap() };
    *inner = Box::new(assembler.assemble_singular(trial_space, test_space));
    wrapper
}

#[concretise_types(
    gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
    gen_type(name = "grid", replace_with = [
        "SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"
    ]),
    gen_type(name = "space", replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"]),
    field(arg = 0, name = "boundary_assembler", wrapper = "BoundaryAssemblerT", replace_with = [
        "laplace::assembler::SingleLayer3dAssembler<{{dtype}}>",
        "laplace::assembler::DoubleLayer3dAssembler<{{dtype}}>",
        "laplace::assembler::AdjointDoubleLayer3dAssembler<{{dtype}}>",
        "laplace::assembler::Hypersingular3dAssembler<{{dtype}}>",
        // TODO: Helmholtz
//        "helmholtz::assembler::SingleLayer3dAssembler<{{dtype}}>",
//        "helmholtz::assembler::DoubleLayer3dAssembler<{{dtype}}>",
//        "helmholtz::assembler::AdjointDoubleLayer3dAssembler<{{dtype}}>",
//        "helmholtz::assembler::Hypersingular3dAssembler<{{dtype}}>",
    ]),
    field(arg = 1, name = "trial_space", wrapper = "SpaceT", replace_with = ["{{space}}"]),
    field(arg = 2, name = "test_space", wrapper = "SpaceT", replace_with = ["{{space}}"]),
)]
pub fn boundary_assembler_assemble<
    T: RlstScalar + MatrixInverse,
    Integrand: BoundaryIntegrand<T = T>,
    K: Kernel<T = T>,
    Space: FunctionSpace<T = T>,
>(
    assembler: &BoundaryAssembler<T, Integrand, K>,
    trial_space: &Space,
    test_space: &Space,
) -> *mut MatrixT {
    let wrapper = matrix_t_create();
    let inner = unsafe { matrix_t_unwrap(wrapper).unwrap() };
    *inner = Box::new(assembler.assemble(trial_space, test_space));
    wrapper
}

#[concretise_types(
    gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
    gen_type(name = "grid", replace_with = [
        "SingleElementGrid<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>",
        "SingleElementGridBorrowed<<{{dtype}} as RlstScalar>::Real, CiarletElement<<{{dtype}} as RlstScalar>::Real>>"
    ]),
    gen_type(name = "space", replace_with = ["SerialFunctionSpace<'_, {{dtype}}, {{grid}}>"]),
    field(arg = 0, name = "boundary_assembler", wrapper = "BoundaryAssemblerT", replace_with = [
        "laplace::assembler::SingleLayer3dAssembler<{{dtype}}>",
        "laplace::assembler::DoubleLayer3dAssembler<{{dtype}}>",
        "laplace::assembler::AdjointDoubleLayer3dAssembler<{{dtype}}>",
        "laplace::assembler::Hypersingular3dAssembler<{{dtype}}>",
        // TODO: Helmholtz
//        "helmholtz::assembler::SingleLayer3dAssembler<{{dtype}}>",
//        "helmholtz::assembler::DoubleLayer3dAssembler<{{dtype}}>",
//        "helmholtz::assembler::AdjointDoubleLayer3dAssembler<{{dtype}}>",
//        "helmholtz::assembler::Hypersingular3dAssembler<{{dtype}}>",
    ]),
    field(arg = 1, name = "trial_space", wrapper = "SpaceT", replace_with = ["{{space}}"]),
    field(arg = 2, name = "test_space", wrapper = "SpaceT", replace_with = ["{{space}}"]),
)]
pub unsafe fn boundary_assembler_assemble_into_memory<
    T: RlstScalar + MatrixInverse,
    Integrand: BoundaryIntegrand<T = T>,
    K: Kernel<T = T>,
    Space: FunctionSpace<T = T>,
>(
    assembler: &BoundaryAssembler<T, Integrand, K>,
    trial_space: &Space,
    test_space: &Space,
    data: *mut c_void,
) {
    let data = data as *mut T;

    assembler.assemble_into_memory(
        trial_space,
        test_space,
        from_raw_parts_mut(data, test_space.global_size() * trial_space.global_size()),
    );
}
