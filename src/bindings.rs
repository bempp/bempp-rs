//! Bindings for C

#![allow(missing_docs)]
#![allow(clippy::missing_safety_doc)]

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum DType {
    F32 = 0,
    F64 = 1,
    C32 = 2,
    C64 = 3,
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum RealDType {
    F32 = 0,
    F64 = 1,
}

mod function {
    use super::DType;
    use crate::{function::SerialFunctionSpace, traits::FunctionSpace};
    use ndelement::{
        bindings as ndelement_b, ciarlet, ciarlet::CiarletElement, traits::ElementFamily,
        types::ReferenceCellType,
    };
    use ndgrid::{bindings as ndgrid_b, traits::Grid, SingleElementGrid};
    use rlst::{c32, c64, MatrixInverse, RlstScalar};
    use std::ffi::c_void;

    #[derive(Debug, PartialEq, Clone, Copy)]
    #[repr(u8)]
    pub enum SpaceType {
        SerialFunctionSpace = 0,
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    #[repr(u8)]
    pub enum GridType {
        SerialSingleElementGrid = 0,
    }

    #[repr(C)]
    pub struct FunctionSpaceWrapper {
        pub space: *const c_void,
        pub dtype: DType,
        pub stype: SpaceType,
        pub gtype: GridType,
    }

    impl Drop for FunctionSpaceWrapper {
        fn drop(&mut self) {
            let Self {
                space,
                dtype,
                stype,
                gtype,
            } = self;
            match stype {
                SpaceType::SerialFunctionSpace => match gtype {
                    GridType::SerialSingleElementGrid => match dtype {
                        DType::F32 => drop(unsafe {
                            Box::from_raw(
                                *space
                                    as *mut SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                            )
                        }),
                        DType::F64 => drop(unsafe {
                            Box::from_raw(
                                *space
                                    as *mut SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                            )
                        }),
                        DType::C32 => drop(unsafe {
                            Box::from_raw(
                                *space
                                    as *mut SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                            )
                        }),
                        DType::C64 => drop(unsafe {
                            Box::from_raw(
                                *space
                                    as *mut SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                            )
                        }),
                    },
                },
            }
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn free_space(s: *mut FunctionSpaceWrapper) {
        assert!(!s.is_null());
        unsafe { drop(Box::from_raw(s)) }
    }

    pub(crate) unsafe fn extract_space<S: FunctionSpace>(
        space: *const FunctionSpaceWrapper,
    ) -> *const S {
        (*space).space as *const S
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_local_size(space: *mut FunctionSpaceWrapper) -> usize {
        match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => (*extract_space::<
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .local_size(),
                    DType::F64 => (*extract_space::<
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .local_size(),
                    DType::C32 => (*extract_space::<
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .local_size(),
                    DType::C64 => (*extract_space::<
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .local_size(),
                },
            },
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_global_size(space: *mut FunctionSpaceWrapper) -> usize {
        match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => (*extract_space::<
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .global_size(),
                    DType::F64 => (*extract_space::<
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .global_size(),
                    DType::C32 => (*extract_space::<
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .global_size(),
                    DType::C64 => (*extract_space::<
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .global_size(),
                },
            },
        }
    }

    pub unsafe extern "C" fn space_new_internal<
        T: RlstScalar + MatrixInverse,
        G: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
        E: ElementFamily<T = T, FiniteElement = CiarletElement<T>, CellType = ReferenceCellType>,
    >(
        g: *const ndgrid_b::grid::GridWrapper,
        f: *const ndelement_b::ciarlet::ElementFamilyWrapper,
    ) -> *const FunctionSpaceWrapper {
        Box::into_raw(Box::new(FunctionSpaceWrapper {
            space: Box::into_raw(Box::new(SerialFunctionSpace::new(
                &*((*g).grid as *const G),
                &*((*f).family as *const E),
            ))) as *const c_void,
            dtype: match (*f).dtype {
                ndelement_b::ciarlet::DType::F32 => DType::F32,
                ndelement_b::ciarlet::DType::F64 => DType::F64,
                ndelement_b::ciarlet::DType::C32 => DType::C32,
                ndelement_b::ciarlet::DType::C64 => DType::C64,
            },
            stype: SpaceType::SerialFunctionSpace,
            gtype: match (*g).gtype {
                ndgrid_b::grid::GridType::SerialSingleElementGrid => {
                    GridType::SerialSingleElementGrid
                }
            },
        }))
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_new(
        g: *const c_void,
        f: *const c_void,
    ) -> *const FunctionSpaceWrapper {
        let g = g as *const ndgrid_b::grid::GridWrapper;
        let f = f as *const ndelement_b::ciarlet::ElementFamilyWrapper;
        match (*g).gtype {
            ndgrid_b::grid::GridType::SerialSingleElementGrid => match (*f).etype {
                ndelement_b::ciarlet::ElementType::Lagrange => match (*g).dtype {
                    ndgrid_b::DType::F32 => match (*f).dtype {
                        ndelement_b::ciarlet::DType::F32 => space_new_internal::<
                            f32,
                            SingleElementGrid<f32, CiarletElement<f32>>,
                            ciarlet::LagrangeElementFamily<f32>,
                        >(g, f),
                        ndelement_b::ciarlet::DType::C32 => space_new_internal::<
                            c32,
                            SingleElementGrid<f32, CiarletElement<f32>>,
                            ciarlet::LagrangeElementFamily<c32>,
                        >(g, f),
                        _ => {
                            panic!("Incompatible data types.");
                        }
                    },
                    ndgrid_b::DType::F64 => match (*f).dtype {
                        ndelement_b::ciarlet::DType::F64 => space_new_internal::<
                            f64,
                            SingleElementGrid<f64, CiarletElement<f64>>,
                            ciarlet::LagrangeElementFamily<f64>,
                        >(g, f),
                        ndelement_b::ciarlet::DType::C64 => space_new_internal::<
                            c64,
                            SingleElementGrid<f64, CiarletElement<f64>>,
                            ciarlet::LagrangeElementFamily<c64>,
                        >(g, f),
                        _ => {
                            panic!("Incompatible data types.");
                        }
                    },
                },
                ndelement_b::ciarlet::ElementType::RaviartThomas => match (*g).dtype {
                    ndgrid_b::DType::F32 => match (*f).dtype {
                        ndelement_b::ciarlet::DType::F32 => space_new_internal::<
                            f32,
                            SingleElementGrid<f32, CiarletElement<f32>>,
                            ciarlet::RaviartThomasElementFamily<f32>,
                        >(g, f),
                        ndelement_b::ciarlet::DType::C32 => space_new_internal::<
                            c32,
                            SingleElementGrid<f32, CiarletElement<f32>>,
                            ciarlet::RaviartThomasElementFamily<c32>,
                        >(g, f),
                        _ => {
                            panic!("Incompatible data types.");
                        }
                    },
                    ndgrid_b::DType::F64 => match (*f).dtype {
                        ndelement_b::ciarlet::DType::F64 => space_new_internal::<
                            f64,
                            SingleElementGrid<f64, CiarletElement<f64>>,
                            ciarlet::RaviartThomasElementFamily<f64>,
                        >(g, f),
                        ndelement_b::ciarlet::DType::C64 => space_new_internal::<
                            c64,
                            SingleElementGrid<f64, CiarletElement<f64>>,
                            ciarlet::RaviartThomasElementFamily<c64>,
                        >(g, f),
                        _ => {
                            panic!("Incompatible data types.");
                        }
                    },
                },
            },
        }
    }
}
