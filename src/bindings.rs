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

impl DType {
    fn from(value: u8) -> Option<Self> {
        match value {
            0 => Some(DType::F32),
            1 => Some(DType::F64),
            2 => Some(DType::C32),
            3 => Some(DType::C64),
            _ => None,
        }
    }
}

pub mod function {
    use super::DType;
    use crate::{function::SerialFunctionSpace, traits::FunctionSpace};
    use ndelement::{
        bindings as ndelement_b, ciarlet, ciarlet::CiarletElement, traits::ElementFamily,
        types::ReferenceCellType,
    };
    use ndgrid::{bindings as ndgrid_b, traits::Grid, types::Ownership, SingleElementGrid};
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

    #[no_mangle]
    pub unsafe extern "C" fn space_is_serial(space: *mut FunctionSpaceWrapper) -> bool {
        match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => (*extract_space::<
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .is_serial(),
                    DType::F64 => (*extract_space::<
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .is_serial(),
                    DType::C32 => (*extract_space::<
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .is_serial(),
                    DType::C64 => (*extract_space::<
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .is_serial(),
                },
            },
        }
    }

    unsafe fn space_grid_internal<
        T: RlstScalar,
        G: Grid<T = T::Real>,
        S: FunctionSpace<T = T, Grid = G>,
    >(
        space: *mut FunctionSpaceWrapper,
    ) -> *const c_void {
        let grid = (*extract_space::<S>(space)).grid();
        let gtype = match (*space).gtype {
            GridType::SerialSingleElementGrid => ndgrid_b::grid::GridType::SerialSingleElementGrid,
        };
        let dtype = match (*space).dtype {
            DType::F32 => ndgrid_b::DType::F32,
            DType::F64 => ndgrid_b::DType::F64,
            DType::C32 => ndgrid_b::DType::F32,
            DType::C64 => ndgrid_b::DType::F64,
        };
        Box::into_raw(Box::new(ndgrid_b::grid::GridWrapper {
            grid: (grid as *const G) as *const c_void,
            gtype,
            dtype,
        })) as *const c_void
    }
    #[no_mangle]
    pub unsafe extern "C" fn space_grid(space: *mut FunctionSpaceWrapper) -> *const c_void {
        match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => space_grid_internal::<
                        f32,
                        SingleElementGrid<f32, CiarletElement<f32>>,
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space),
                    DType::F64 => space_grid_internal::<
                        f64,
                        SingleElementGrid<f64, CiarletElement<f64>>,
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space),
                    DType::C32 => space_grid_internal::<
                        c32,
                        SingleElementGrid<f32, CiarletElement<f32>>,
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space),
                    DType::C64 => space_grid_internal::<
                        c64,
                        SingleElementGrid<f64, CiarletElement<f64>>,
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space),
                },
            },
        }
    }

    unsafe fn space_element_internal<
        T: RlstScalar + MatrixInverse,
        G: Grid<T = T::Real>,
        S: FunctionSpace<T = T, Grid = G, FiniteElement = CiarletElement<T>>,
    >(
        space: *mut FunctionSpaceWrapper,
        entity_type: u8,
    ) -> *const c_void {
        let element =
            (*extract_space::<S>(space)).element(ReferenceCellType::from(entity_type).unwrap());
        let dtype = match (*space).dtype {
            DType::F32 => ndelement_b::ciarlet::DType::F32,
            DType::F64 => ndelement_b::ciarlet::DType::F64,
            DType::C32 => ndelement_b::ciarlet::DType::C32,
            DType::C64 => ndelement_b::ciarlet::DType::C64,
        };
        Box::into_raw(Box::new(ndelement_b::ciarlet::CiarletElementWrapper {
            element: (element as *const CiarletElement<T>) as *const c_void,
            dtype,
        })) as *const c_void
    }
    #[no_mangle]
    pub unsafe extern "C" fn space_element(
        space: *mut FunctionSpaceWrapper,
        entity_type: u8,
    ) -> *const c_void {
        match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => space_element_internal::<
                        f32,
                        SingleElementGrid<f32, CiarletElement<f32>>,
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space, entity_type),
                    DType::F64 => space_element_internal::<
                        f64,
                        SingleElementGrid<f64, CiarletElement<f64>>,
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space, entity_type),
                    DType::C32 => space_element_internal::<
                        c32,
                        SingleElementGrid<f32, CiarletElement<f32>>,
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space, entity_type),
                    DType::C64 => space_element_internal::<
                        c64,
                        SingleElementGrid<f64, CiarletElement<f64>>,
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space, entity_type),
                },
            },
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_get_local_dof_numbers_size(
        space: *mut FunctionSpaceWrapper,
        entity_dim: usize,
        entity_number: usize,
    ) -> usize {
        match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => (*extract_space::<
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .get_local_dof_numbers(entity_dim, entity_number),
                    DType::F64 => (*extract_space::<
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .get_local_dof_numbers(entity_dim, entity_number),
                    DType::C32 => (*extract_space::<
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .get_local_dof_numbers(entity_dim, entity_number),
                    DType::C64 => (*extract_space::<
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .get_local_dof_numbers(entity_dim, entity_number),
                },
            },
        }
        .len()
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_get_local_dof_numbers(
        space: *mut FunctionSpaceWrapper,
        entity_dim: usize,
        entity_number: usize,
        dofs: *mut usize,
    ) {
        for (i, dof) in match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => (*extract_space::<
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .get_local_dof_numbers(entity_dim, entity_number),
                    DType::F64 => (*extract_space::<
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .get_local_dof_numbers(entity_dim, entity_number),
                    DType::C32 => (*extract_space::<
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .get_local_dof_numbers(entity_dim, entity_number),
                    DType::C64 => (*extract_space::<
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .get_local_dof_numbers(entity_dim, entity_number),
                },
            },
        }
        .iter()
        .enumerate()
        {
            *dofs.add(i) = *dof;
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_has_cell_dofs(
        space: *mut FunctionSpaceWrapper,
        cell: usize,
    ) -> bool {
        match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => (*extract_space::<
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .cell_dofs(cell),
                    DType::F64 => (*extract_space::<
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .cell_dofs(cell),
                    DType::C32 => (*extract_space::<
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .cell_dofs(cell),
                    DType::C64 => (*extract_space::<
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .cell_dofs(cell),
                },
            },
        }
        .is_some()
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_cell_dofs_size(
        space: *mut FunctionSpaceWrapper,
        cell: usize,
    ) -> usize {
        match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => (*extract_space::<
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .cell_dofs(cell),
                    DType::F64 => (*extract_space::<
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .cell_dofs(cell),
                    DType::C32 => (*extract_space::<
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .cell_dofs(cell),
                    DType::C64 => (*extract_space::<
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .cell_dofs(cell),
                },
            },
        }
        .unwrap()
        .len()
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_cell_dofs(
        space: *mut FunctionSpaceWrapper,
        cell: usize,
        dofs: *mut usize,
    ) {
        for (i, dof) in match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => (*extract_space::<
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .cell_dofs(cell),
                    DType::F64 => (*extract_space::<
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .cell_dofs(cell),
                    DType::C32 => (*extract_space::<
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .cell_dofs(cell),
                    DType::C64 => (*extract_space::<
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .cell_dofs(cell),
                },
            },
        }
        .unwrap()
        .iter()
        .enumerate()
        {
            *dofs.add(i) = *dof;
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_global_dof_index(
        space: *mut FunctionSpaceWrapper,
        local_dof_index: usize,
    ) -> usize {
        match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => (*extract_space::<
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .global_dof_index(local_dof_index),
                    DType::F64 => (*extract_space::<
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .global_dof_index(local_dof_index),
                    DType::C32 => (*extract_space::<
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .global_dof_index(local_dof_index),
                    DType::C64 => (*extract_space::<
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .global_dof_index(local_dof_index),
                },
            },
        }
    }

    unsafe fn space_ownership(
        space: *mut FunctionSpaceWrapper,
        local_dof_index: usize,
    ) -> Ownership {
        match (*space).stype {
            SpaceType::SerialFunctionSpace => match (*space).gtype {
                GridType::SerialSingleElementGrid => match (*space).dtype {
                    DType::F32 => (*extract_space::<
                        SerialFunctionSpace<f32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .ownership(local_dof_index),
                    DType::F64 => (*extract_space::<
                        SerialFunctionSpace<f64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .ownership(local_dof_index),
                    DType::C32 => (*extract_space::<
                        SerialFunctionSpace<c32, SingleElementGrid<f32, CiarletElement<f32>>>,
                    >(space))
                    .ownership(local_dof_index),
                    DType::C64 => (*extract_space::<
                        SerialFunctionSpace<c64, SingleElementGrid<f64, CiarletElement<f64>>>,
                    >(space))
                    .ownership(local_dof_index),
                },
            },
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_is_owned(
        space: *mut FunctionSpaceWrapper,
        local_dof_index: usize,
    ) -> bool {
        space_ownership(space, local_dof_index) == Ownership::Owned
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_ownership_process(
        space: *mut FunctionSpaceWrapper,
        local_dof_index: usize,
    ) -> usize {
        if let Ownership::Ghost(process, _index) = space_ownership(space, local_dof_index) {
            process
        } else {
            panic!("Cannot get process of owned DOF");
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_ownership_index(
        space: *mut FunctionSpaceWrapper,
        local_dof_index: usize,
    ) -> usize {
        if let Ownership::Ghost(_process, index) = space_ownership(space, local_dof_index) {
            index
        } else {
            panic!("Cannot get process of owned DOF");
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn space_dtype(space: *const FunctionSpaceWrapper) -> u8 {
        (*space).dtype as u8
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

pub mod boundary_assembly {
    use super::function::{extract_space, FunctionSpaceWrapper, GridType, SpaceType};
    use super::DType;
    use crate::assembly::common::SparseMatrixData;
    use crate::{
        assembly::boundary::integrands::{
            AdjointDoubleLayerBoundaryIntegrand, BoundaryIntegrandScalarProduct,
            BoundaryIntegrandSum, DoubleLayerBoundaryIntegrand,
            HypersingularCurlCurlBoundaryIntegrand, HypersingularNormalNormalBoundaryIntegrand,
            SingleLayerBoundaryIntegrand,
        },
        assembly::boundary::BoundaryAssembler,
        assembly::kernels::KernelEvaluator,
        function::SerialFunctionSpace,
        traits::{
            BoundaryAssembly, BoundaryIntegrand, FunctionSpace,
            KernelEvaluator as KernelEvaluatorTrait,
        },
    };
    use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
    use ndelement::{ciarlet::CiarletElement, types::ReferenceCellType};
    use ndgrid::SingleElementGrid;
    use rlst::{c32, c64, rlst_array_from_slice_mut2, MatrixInverse, RlstScalar};
    use std::ffi::c_void;
    use std::slice::from_raw_parts_mut;

    type LaplaceHypersingularBoundaryIntegrand<T> = HypersingularCurlCurlBoundaryIntegrand<T>;
    type HelmholtzHypersingularBoundaryIntegrand<T> = BoundaryIntegrandSum<
        T,
        HypersingularCurlCurlBoundaryIntegrand<T>,
        BoundaryIntegrandScalarProduct<T, HypersingularNormalNormalBoundaryIntegrand<T>>,
    >;

    #[derive(Debug, PartialEq, Clone, Copy)]
    #[repr(u8)]
    pub enum BoundaryOperator {
        SingleLayer = 0,
        DoubleLayer = 1,
        AdjointDoubleLayer = 2,
        Hypersingular = 3,
        ElectricField = 4,
        MagneticField = 5,
    }

    impl BoundaryOperator {
        fn from(value: u8) -> Option<Self> {
            match value {
                0 => Some(BoundaryOperator::SingleLayer),
                1 => Some(BoundaryOperator::DoubleLayer),
                2 => Some(BoundaryOperator::AdjointDoubleLayer),
                3 => Some(BoundaryOperator::Hypersingular),
                4 => Some(BoundaryOperator::ElectricField),
                5 => Some(BoundaryOperator::MagneticField),
                _ => None,
            }
        }
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    #[repr(u8)]
    pub enum KernelType {
        Laplace = 0,
        Helmholtz = 1,
    }

    #[repr(C)]
    pub struct BoundaryAssemblerWrapper {
        pub assembler: *const c_void,
        pub itype: BoundaryOperator,
        pub ktype: KernelType,
        pub dtype: DType,
    }
    impl Drop for BoundaryAssemblerWrapper {
        fn drop(&mut self) {
            let Self {
                assembler,
                itype,
                ktype,
                dtype,
            } = self;
            match ktype {
                KernelType::Laplace => match itype {
                    BoundaryOperator::SingleLayer => match dtype {
                        DType::F32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        f32,
                                        SingleLayerBoundaryIntegrand<f32>,
                                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    >,
                            )
                        }),
                        DType::F64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        f64,
                                        SingleLayerBoundaryIntegrand<f64>,
                                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    BoundaryOperator::DoubleLayer => match dtype {
                        DType::F32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        f32,
                                        DoubleLayerBoundaryIntegrand<f32>,
                                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    >,
                            )
                        }),
                        DType::F64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        f64,
                                        DoubleLayerBoundaryIntegrand<f64>,
                                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    BoundaryOperator::AdjointDoubleLayer => match dtype {
                        DType::F32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        f32,
                                        AdjointDoubleLayerBoundaryIntegrand<f32>,
                                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    >,
                            )
                        }),
                        DType::F64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        f64,
                                        AdjointDoubleLayerBoundaryIntegrand<f64>,
                                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    BoundaryOperator::Hypersingular => match dtype {
                        DType::F32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        f32,
                                        LaplaceHypersingularBoundaryIntegrand<f32>,
                                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    >,
                            )
                        }),
                        DType::F64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        f64,
                                        LaplaceHypersingularBoundaryIntegrand<f64>,
                                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    _ => {
                        panic!("Invalid operator");
                    }
                },
                KernelType::Helmholtz => match itype {
                    BoundaryOperator::SingleLayer => match dtype {
                        DType::C32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        c32,
                                        SingleLayerBoundaryIntegrand<c32>,
                                        KernelEvaluator<c32, Laplace3dKernel<c32>>,
                                    >,
                            )
                        }),
                        DType::C64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        c64,
                                        SingleLayerBoundaryIntegrand<c64>,
                                        KernelEvaluator<c64, Laplace3dKernel<c64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    BoundaryOperator::DoubleLayer => match dtype {
                        DType::C32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        c32,
                                        DoubleLayerBoundaryIntegrand<c32>,
                                        KernelEvaluator<c32, Laplace3dKernel<c32>>,
                                    >,
                            )
                        }),
                        DType::C64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        c64,
                                        DoubleLayerBoundaryIntegrand<c64>,
                                        KernelEvaluator<c64, Laplace3dKernel<c64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    BoundaryOperator::AdjointDoubleLayer => match dtype {
                        DType::C32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        c32,
                                        AdjointDoubleLayerBoundaryIntegrand<c32>,
                                        KernelEvaluator<c32, Laplace3dKernel<c32>>,
                                    >,
                            )
                        }),
                        DType::C64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        c64,
                                        AdjointDoubleLayerBoundaryIntegrand<c64>,
                                        KernelEvaluator<c64, Laplace3dKernel<c64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    BoundaryOperator::Hypersingular => match dtype {
                        DType::C32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        c32,
                                        HelmholtzHypersingularBoundaryIntegrand<c32>,
                                        KernelEvaluator<c32, Laplace3dKernel<c32>>,
                                    >,
                            )
                        }),
                        DType::C64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut BoundaryAssembler<
                                        c64,
                                        HelmholtzHypersingularBoundaryIntegrand<c64>,
                                        KernelEvaluator<c64, Laplace3dKernel<c64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    _ => {
                        panic!("Invalid operator");
                    }
                },
            }
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn free_boundary_assembler(a: *mut BoundaryAssemblerWrapper) {
        assert!(!a.is_null());
        unsafe { drop(Box::from_raw(a)) }
    }

    #[repr(C)]
    pub struct SparseMatrixWrapper {
        pub matrix: *const c_void,
        pub dtype: DType,
    }
    impl Drop for SparseMatrixWrapper {
        fn drop(&mut self) {
            let Self { matrix, dtype } = self;
            match dtype {
                DType::F32 => drop(unsafe { Box::from_raw(*matrix as *mut SparseMatrixData<f32>) }),
                DType::F64 => drop(unsafe { Box::from_raw(*matrix as *mut SparseMatrixData<f64>) }),
                DType::C32 => drop(unsafe { Box::from_raw(*matrix as *mut SparseMatrixData<c32>) }),
                DType::C64 => drop(unsafe { Box::from_raw(*matrix as *mut SparseMatrixData<c64>) }),
            }
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn free_sparse_matrix(m: *mut SparseMatrixWrapper) {
        assert!(!m.is_null());
        unsafe { drop(Box::from_raw(m)) }
    }

    pub(crate) unsafe fn extract_sparse_matrix<T: RlstScalar + MatrixInverse>(
        matrix: *const SparseMatrixWrapper,
    ) -> *const SparseMatrixData<T> {
        (*matrix).matrix as *const SparseMatrixData<T>
    }

    #[no_mangle]
    pub unsafe extern "C" fn sparse_data_size(matrix: *mut SparseMatrixWrapper) -> usize {
        match (*matrix).dtype {
            DType::F32 => (*extract_sparse_matrix::<f32>(matrix)).data.len(),
            DType::F64 => (*extract_sparse_matrix::<f64>(matrix)).data.len(),
            DType::C32 => (*extract_sparse_matrix::<c32>(matrix)).data.len(),
            DType::C64 => (*extract_sparse_matrix::<c64>(matrix)).data.len(),
        }
    }

    unsafe fn sparse_data_internal<T: RlstScalar + MatrixInverse>(
        matrix: *mut SparseMatrixWrapper,
        data: *mut c_void,
    ) {
        let data = data as *mut T;
        for (i, j) in (*extract_sparse_matrix::<T>(matrix))
            .data
            .iter()
            .enumerate()
        {
            *data.add(i) = *j;
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn sparse_data(matrix: *mut SparseMatrixWrapper, data: *mut c_void) {
        match (*matrix).dtype {
            DType::F32 => sparse_data_internal::<f32>(matrix, data),
            DType::F64 => sparse_data_internal::<f64>(matrix, data),
            DType::C32 => sparse_data_internal::<c32>(matrix, data),
            DType::C64 => sparse_data_internal::<c64>(matrix, data),
        }
    }

    unsafe fn sparse_rows_internal<T: RlstScalar + MatrixInverse>(
        matrix: *mut SparseMatrixWrapper,
        rows: *mut usize,
    ) {
        for (i, j) in (*extract_sparse_matrix::<T>(matrix))
            .rows
            .iter()
            .enumerate()
        {
            *rows.add(i) = *j;
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn sparse_rows(matrix: *mut SparseMatrixWrapper, rows: *mut usize) {
        match (*matrix).dtype {
            DType::F32 => sparse_rows_internal::<f32>(matrix, rows),
            DType::F64 => sparse_rows_internal::<f64>(matrix, rows),
            DType::C32 => sparse_rows_internal::<c32>(matrix, rows),
            DType::C64 => sparse_rows_internal::<c64>(matrix, rows),
        }
    }

    unsafe fn sparse_cols_internal<T: RlstScalar + MatrixInverse>(
        matrix: *mut SparseMatrixWrapper,
        cols: *mut usize,
    ) {
        for (i, j) in (*extract_sparse_matrix::<T>(matrix))
            .cols
            .iter()
            .enumerate()
        {
            *cols.add(i) = *j;
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn sparse_cols(matrix: *mut SparseMatrixWrapper, cols: *mut usize) {
        match (*matrix).dtype {
            DType::F32 => sparse_cols_internal::<f32>(matrix, cols),
            DType::F64 => sparse_cols_internal::<f64>(matrix, cols),
            DType::C32 => sparse_cols_internal::<c32>(matrix, cols),
            DType::C64 => sparse_cols_internal::<c64>(matrix, cols),
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn sparse_shape0(matrix: *mut SparseMatrixWrapper) -> usize {
        match (*matrix).dtype {
            DType::F32 => (*extract_sparse_matrix::<f32>(matrix)).shape[0],
            DType::F64 => (*extract_sparse_matrix::<f64>(matrix)).shape[0],
            DType::C32 => (*extract_sparse_matrix::<c32>(matrix)).shape[0],
            DType::C64 => (*extract_sparse_matrix::<c64>(matrix)).shape[0],
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn sparse_shape1(matrix: *mut SparseMatrixWrapper) -> usize {
        match (*matrix).dtype {
            DType::F32 => (*extract_sparse_matrix::<f32>(matrix)).shape[1],
            DType::F64 => (*extract_sparse_matrix::<f64>(matrix)).shape[1],
            DType::C32 => (*extract_sparse_matrix::<c32>(matrix)).shape[1],
            DType::C64 => (*extract_sparse_matrix::<c64>(matrix)).shape[1],
        }
    }

    pub(crate) unsafe fn extract_boundary_assembler<
        T: RlstScalar + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
    >(
        assembler: *const BoundaryAssemblerWrapper,
    ) -> *const BoundaryAssembler<T, Integrand, Kernel> {
        (*assembler).assembler as *const BoundaryAssembler<T, Integrand, Kernel>
    }

    pub(crate) unsafe fn extract_boundary_assembler_mut<
        T: RlstScalar + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
    >(
        assembler: *const BoundaryAssemblerWrapper,
    ) -> *mut BoundaryAssembler<T, Integrand, Kernel> {
        (*assembler).assembler as *mut BoundaryAssembler<T, Integrand, Kernel>
    }

    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_has_quadrature_degree(
        assembler: *mut BoundaryAssemblerWrapper,
        cell: u8,
    ) -> bool {
        let cell = ReferenceCellType::from(cell).unwrap();
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        SingleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        SingleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        DoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        DoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        AdjointDoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        AdjointDoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        LaplaceHypersingularBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        LaplaceHypersingularBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        SingleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        SingleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        DoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        DoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        AdjointDoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        AdjointDoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        HelmholtzHypersingularBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        HelmholtzHypersingularBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
        .is_some()
    }

    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_set_quadrature_degree(
        assembler: *mut BoundaryAssemblerWrapper,
        cell: u8,
        degree: usize,
    ) {
        let cell = ReferenceCellType::from(cell).unwrap();
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        SingleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        SingleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        DoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        DoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        AdjointDoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        AdjointDoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        LaplaceHypersingularBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        LaplaceHypersingularBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        SingleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        SingleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        DoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        DoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        AdjointDoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        AdjointDoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        HelmholtzHypersingularBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        HelmholtzHypersingularBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_quadrature_degree(
        assembler: *mut BoundaryAssemblerWrapper,
        cell: u8,
    ) -> usize {
        let cell = ReferenceCellType::from(cell).unwrap();
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        SingleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        SingleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        DoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        DoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        AdjointDoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        AdjointDoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        LaplaceHypersingularBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        LaplaceHypersingularBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        SingleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        SingleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        DoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        DoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        AdjointDoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        AdjointDoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        HelmholtzHypersingularBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        HelmholtzHypersingularBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
        .unwrap()
    }

    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_has_singular_quadrature_degree(
        assembler: *mut BoundaryAssemblerWrapper,
        cell0: u8,
        cell1: u8,
    ) -> bool {
        let cells = (
            ReferenceCellType::from(cell0).unwrap(),
            ReferenceCellType::from(cell1).unwrap(),
        );
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        SingleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        SingleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        DoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        DoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        AdjointDoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        AdjointDoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        LaplaceHypersingularBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        LaplaceHypersingularBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        SingleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        SingleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        DoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        DoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        AdjointDoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        AdjointDoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        HelmholtzHypersingularBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        HelmholtzHypersingularBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
        .is_some()
    }

    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_set_singular_quadrature_degree(
        assembler: *mut BoundaryAssemblerWrapper,
        cell0: u8,
        cell1: u8,
        degree: usize,
    ) {
        let cells = (
            ReferenceCellType::from(cell0).unwrap(),
            ReferenceCellType::from(cell1).unwrap(),
        );
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        SingleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        SingleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        DoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        DoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        AdjointDoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        AdjointDoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        LaplaceHypersingularBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        LaplaceHypersingularBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        SingleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        SingleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        DoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        DoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        AdjointDoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        AdjointDoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        HelmholtzHypersingularBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        HelmholtzHypersingularBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_singular_quadrature_degree(cells, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_singular_quadrature_degree(
        assembler: *mut BoundaryAssemblerWrapper,
        cell0: u8,
        cell1: u8,
    ) -> usize {
        let cells = (
            ReferenceCellType::from(cell0).unwrap(),
            ReferenceCellType::from(cell1).unwrap(),
        );
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        SingleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        SingleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        DoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        DoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        AdjointDoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        AdjointDoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        LaplaceHypersingularBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        LaplaceHypersingularBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        SingleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        SingleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        DoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        DoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        AdjointDoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        AdjointDoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        HelmholtzHypersingularBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        HelmholtzHypersingularBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .singular_quadrature_degree(cells),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
        .unwrap()
    }

    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_set_batch_size(
        assembler: *mut BoundaryAssemblerWrapper,
        batch_size: usize,
    ) {
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        SingleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        SingleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        DoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        DoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        AdjointDoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        AdjointDoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler_mut::<
                        f32,
                        LaplaceHypersingularBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::F64 => (*extract_boundary_assembler_mut::<
                        f64,
                        LaplaceHypersingularBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        SingleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        SingleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        DoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        DoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        AdjointDoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        AdjointDoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler_mut::<
                        c32,
                        HelmholtzHypersingularBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::C64 => (*extract_boundary_assembler_mut::<
                        c64,
                        HelmholtzHypersingularBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_batch_size(
        assembler: *mut BoundaryAssemblerWrapper,
    ) -> usize {
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        SingleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .batch_size(),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        SingleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        DoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .batch_size(),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        DoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        AdjointDoubleLayerBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .batch_size(),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        AdjointDoubleLayerBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => (*extract_boundary_assembler::<
                        f32,
                        LaplaceHypersingularBoundaryIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .batch_size(),
                    DType::F64 => (*extract_boundary_assembler::<
                        f64,
                        LaplaceHypersingularBoundaryIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        SingleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .batch_size(),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        SingleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        DoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .batch_size(),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        DoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        AdjointDoubleLayerBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .batch_size(),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        AdjointDoubleLayerBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => (*extract_boundary_assembler::<
                        c32,
                        HelmholtzHypersingularBoundaryIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .batch_size(),
                    DType::C64 => (*extract_boundary_assembler::<
                        c64,
                        HelmholtzHypersingularBoundaryIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    unsafe fn boundary_assembler_assemble_into_dense_internal_real<
        T: RlstScalar<Real = T> + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut BoundaryAssemblerWrapper,
        output: *mut c_void,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) {
        let trial_dim = (*extract_space::<Space>(trial_space)).global_size();
        let test_dim = (*extract_space::<Space>(test_space)).global_size();
        let mut output = rlst_array_from_slice_mut2!(
            from_raw_parts_mut(output as *mut T, trial_dim * test_dim),
            [test_dim, trial_dim]
        );

        (*extract_boundary_assembler::<T, Integrand, Kernel>(assembler)).assemble_into_dense(
            &mut output,
            &*extract_space::<Space>(trial_space),
            &*extract_space::<Space>(test_space),
        )
    }
    unsafe fn boundary_assembler_assemble_into_dense_internal_complex<
        T: RlstScalar<Complex = T> + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut BoundaryAssemblerWrapper,
        output: *mut c_void,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) {
        let trial_dim = (*extract_space::<Space>(trial_space)).global_size();
        let test_dim = (*extract_space::<Space>(test_space)).global_size();
        let mut output = rlst_array_from_slice_mut2!(
            from_raw_parts_mut(output as *mut T, trial_dim * test_dim),
            [test_dim, trial_dim]
        );

        (*extract_boundary_assembler::<T, Integrand, Kernel>(assembler)).assemble_into_dense(
            &mut output,
            &*extract_space::<Space>(trial_space),
            &*extract_space::<Space>(test_space),
        )
    }
    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_assemble_into_dense(
        assembler: *mut BoundaryAssemblerWrapper,
        output: *mut c_void,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) {
        match (*assembler).dtype {
            DType::F32 => {
                assert_eq!((*trial_space).dtype, DType::F32);
                assert_eq!((*test_space).dtype, DType::F32);
            }
            DType::F64 => {
                assert_eq!((*trial_space).dtype, DType::F64);
                assert_eq!((*test_space).dtype, DType::F64);
            }
            DType::C32 => {
                assert_eq!((*trial_space).dtype, DType::F32);
                assert_eq!((*test_space).dtype, DType::F32);
            }
            DType::C64 => {
                assert_eq!((*trial_space).dtype, DType::F64);
                assert_eq!((*test_space).dtype, DType::F64);
            }
        }
        assert_eq!((*test_space).stype, (*trial_space).stype);
        assert_eq!((*test_space).gtype, (*trial_space).gtype);
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_real::<
                                    f32,
                                    SingleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_real::<
                                    f64,
                                    SingleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_real::<
                                    f32,
                                    DoubleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_real::<
                                    f64,
                                    DoubleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_real::<
                                    f32,
                                    AdjointDoubleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_real::<
                                    f64,
                                    AdjointDoubleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_real::<
                                    f32,
                                    LaplaceHypersingularBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_real::<
                                    f64,
                                    LaplaceHypersingularBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_complex::<
                                    c32,
                                    SingleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_complex::<
                                    c64,
                                    SingleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_complex::<
                                    c32,
                                    DoubleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_complex::<
                                    c64,
                                    DoubleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_complex::<
                                    c32,
                                    AdjointDoubleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_complex::<
                                    c64,
                                    AdjointDoubleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_complex::<
                                    c32,
                                    HelmholtzHypersingularBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_into_dense_internal_complex::<
                                    c64,
                                    HelmholtzHypersingularBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    unsafe fn boundary_assembler_assemble_singular_into_dense_internal_real<
        T: RlstScalar<Real = T> + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut BoundaryAssemblerWrapper,
        output: *mut c_void,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) {
        let trial_dim = (*extract_space::<Space>(trial_space)).global_size();
        let test_dim = (*extract_space::<Space>(test_space)).global_size();
        let mut output = rlst_array_from_slice_mut2!(
            from_raw_parts_mut(output as *mut T, trial_dim * test_dim),
            [test_dim, trial_dim]
        );

        (*extract_boundary_assembler::<T, Integrand, Kernel>(assembler))
            .assemble_singular_into_dense(
                &mut output,
                &*extract_space::<Space>(trial_space),
                &*extract_space::<Space>(test_space),
            )
    }
    unsafe fn boundary_assembler_assemble_singular_into_dense_internal_complex<
        T: RlstScalar<Complex = T> + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut BoundaryAssemblerWrapper,
        output: *mut c_void,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) {
        let trial_dim = (*extract_space::<Space>(trial_space)).global_size();
        let test_dim = (*extract_space::<Space>(test_space)).global_size();
        let mut output = rlst_array_from_slice_mut2!(
            from_raw_parts_mut(output as *mut T, trial_dim * test_dim),
            [test_dim, trial_dim]
        );

        (*extract_boundary_assembler::<T, Integrand, Kernel>(assembler))
            .assemble_singular_into_dense(
                &mut output,
                &*extract_space::<Space>(trial_space),
                &*extract_space::<Space>(test_space),
            )
    }
    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_assemble_singular_into_dense(
        assembler: *mut BoundaryAssemblerWrapper,
        output: *mut c_void,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) {
        match (*assembler).dtype {
            DType::F32 => {
                assert_eq!((*trial_space).dtype, DType::F32);
                assert_eq!((*test_space).dtype, DType::F32);
            }
            DType::F64 => {
                assert_eq!((*trial_space).dtype, DType::F64);
                assert_eq!((*test_space).dtype, DType::F64);
            }
            DType::C32 => {
                assert_eq!((*trial_space).dtype, DType::F32);
                assert_eq!((*test_space).dtype, DType::F32);
            }
            DType::C64 => {
                assert_eq!((*trial_space).dtype, DType::F64);
                assert_eq!((*test_space).dtype, DType::F64);
            }
        }
        assert_eq!((*test_space).stype, (*trial_space).stype);
        assert_eq!((*test_space).gtype, (*trial_space).gtype);
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_real::<
                                    f32,
                                    SingleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_real::<
                                    f64,
                                    SingleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_real::<
                                    f32,
                                    DoubleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_real::<
                                    f64,
                                    DoubleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_real::<
                                    f32,
                                    AdjointDoubleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_real::<
                                    f64,
                                    AdjointDoubleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_real::<
                                    f32,
                                    LaplaceHypersingularBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_real::<
                                    f64,
                                    LaplaceHypersingularBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_complex::<
                                    c32,
                                    SingleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_complex::<
                                    c64,
                                    SingleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_complex::<
                                    c32,
                                    DoubleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_complex::<
                                    c64,
                                    DoubleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_complex::<
                                    c32,
                                    AdjointDoubleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_complex::<
                                    c64,
                                    AdjointDoubleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_complex::<
                                    c32,
                                    HelmholtzHypersingularBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_into_dense_internal_complex::<
                                    c64,
                                    HelmholtzHypersingularBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    unsafe fn boundary_assembler_assemble_nonsingular_into_dense_internal_real<
        T: RlstScalar<Real = T> + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut BoundaryAssemblerWrapper,
        output: *mut c_void,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) {
        let trial_dim = (*extract_space::<Space>(trial_space)).global_size();
        let test_dim = (*extract_space::<Space>(test_space)).global_size();
        let mut output = rlst_array_from_slice_mut2!(
            from_raw_parts_mut(output as *mut T, trial_dim * test_dim),
            [test_dim, trial_dim]
        );
        let trial_colouring = (*extract_space::<Space>(trial_space)).cell_colouring();
        let test_colouring = (*extract_space::<Space>(test_space)).cell_colouring();

        (*extract_boundary_assembler::<T, Integrand, Kernel>(assembler))
            .assemble_nonsingular_into_dense(
                &mut output,
                &*extract_space::<Space>(trial_space),
                &*extract_space::<Space>(test_space),
                &trial_colouring,
                &test_colouring,
            )
    }
    unsafe fn boundary_assembler_assemble_nonsingular_into_dense_internal_complex<
        T: RlstScalar<Complex = T> + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut BoundaryAssemblerWrapper,
        output: *mut c_void,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) {
        let trial_dim = (*extract_space::<Space>(trial_space)).global_size();
        let test_dim = (*extract_space::<Space>(test_space)).global_size();
        let mut output = rlst_array_from_slice_mut2!(
            from_raw_parts_mut(output as *mut T, trial_dim * test_dim),
            [test_dim, trial_dim]
        );
        let trial_colouring = (*extract_space::<Space>(trial_space)).cell_colouring();
        let test_colouring = (*extract_space::<Space>(test_space)).cell_colouring();

        (*extract_boundary_assembler::<T, Integrand, Kernel>(assembler))
            .assemble_nonsingular_into_dense(
                &mut output,
                &*extract_space::<Space>(trial_space),
                &*extract_space::<Space>(test_space),
                &trial_colouring,
                &test_colouring,
            )
    }
    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_assemble_nonsingular_into_dense(
        assembler: *mut BoundaryAssemblerWrapper,
        output: *mut c_void,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) {
        match (*assembler).dtype {
            DType::F32 => {
                assert_eq!((*trial_space).dtype, DType::F32);
                assert_eq!((*test_space).dtype, DType::F32);
            }
            DType::F64 => {
                assert_eq!((*trial_space).dtype, DType::F64);
                assert_eq!((*test_space).dtype, DType::F64);
            }
            DType::C32 => {
                assert_eq!((*trial_space).dtype, DType::F32);
                assert_eq!((*test_space).dtype, DType::F32);
            }
            DType::C64 => {
                assert_eq!((*trial_space).dtype, DType::F64);
                assert_eq!((*test_space).dtype, DType::F64);
            }
        }
        assert_eq!((*test_space).stype, (*trial_space).stype);
        assert_eq!((*test_space).gtype, (*trial_space).gtype);
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_real::<
                                    f32,
                                    SingleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_real::<
                                    f64,
                                    SingleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_real::<
                                    f32,
                                    DoubleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_real::<
                                    f64,
                                    DoubleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_real::<
                                    f32,
                                    AdjointDoubleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_real::<
                                    f64,
                                    AdjointDoubleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_real::<
                                    f32,
                                    LaplaceHypersingularBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_real::<
                                    f64,
                                    LaplaceHypersingularBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_complex::<
                                    c32,
                                    SingleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_complex::<
                                    c64,
                                    SingleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_complex::<
                                    c32,
                                    DoubleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_complex::<
                                    c64,
                                    DoubleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_complex::<
                                    c32,
                                    AdjointDoubleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_complex::<
                                    c64,
                                    AdjointDoubleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_complex::<
                                    c32,
                                    HelmholtzHypersingularBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_nonsingular_into_dense_internal_complex::<
                                    c64,
                                    HelmholtzHypersingularBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, trial_space, test_space
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    unsafe fn boundary_assembler_assemble_singular_internal_real<
        T: RlstScalar<Real = T> + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut BoundaryAssemblerWrapper,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) -> *const SparseMatrixWrapper {
        let trial_dim = (*extract_space::<Space>(trial_space)).global_size();
        let test_dim = (*extract_space::<Space>(test_space)).global_size();
        Box::into_raw(Box::new(SparseMatrixWrapper {
            matrix: Box::into_raw(Box::new(
                (*extract_boundary_assembler::<T, Integrand, Kernel>(assembler)).assemble_singular(
                    [test_dim, trial_dim],
                    &*extract_space::<Space>(trial_space),
                    &*extract_space::<Space>(test_space),
                ),
            )) as *const c_void,
            dtype: (*assembler).dtype,
        }))
    }
    unsafe fn boundary_assembler_assemble_singular_internal_complex<
        T: RlstScalar<Complex = T> + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut BoundaryAssemblerWrapper,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) -> *const SparseMatrixWrapper {
        let trial_dim = (*extract_space::<Space>(trial_space)).global_size();
        let test_dim = (*extract_space::<Space>(test_space)).global_size();
        Box::into_raw(Box::new(SparseMatrixWrapper {
            matrix: Box::into_raw(Box::new(
                (*extract_boundary_assembler::<T, Integrand, Kernel>(assembler)).assemble_singular(
                    [test_dim, trial_dim],
                    &*extract_space::<Space>(trial_space),
                    &*extract_space::<Space>(test_space),
                ),
            )) as *const c_void,
            dtype: (*assembler).dtype,
        }))
    }
    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_assemble_singular(
        assembler: *mut BoundaryAssemblerWrapper,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) -> *const SparseMatrixWrapper {
        match (*assembler).dtype {
            DType::F32 => {
                assert_eq!((*trial_space).dtype, DType::F32);
                assert_eq!((*test_space).dtype, DType::F32);
            }
            DType::F64 => {
                assert_eq!((*trial_space).dtype, DType::F64);
                assert_eq!((*test_space).dtype, DType::F64);
            }
            DType::C32 => {
                assert_eq!((*trial_space).dtype, DType::F32);
                assert_eq!((*test_space).dtype, DType::F32);
            }
            DType::C64 => {
                assert_eq!((*trial_space).dtype, DType::F64);
                assert_eq!((*test_space).dtype, DType::F64);
            }
        }
        assert_eq!((*test_space).stype, (*trial_space).stype);
        assert_eq!((*test_space).gtype, (*trial_space).gtype);
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_real::<
                                    f32,
                                    SingleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_real::<
                                    f64,
                                    SingleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_real::<
                                    f32,
                                    DoubleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_real::<
                                    f64,
                                    DoubleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_real::<
                                    f32,
                                    AdjointDoubleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_real::<
                                    f64,
                                    AdjointDoubleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_real::<
                                    f32,
                                    LaplaceHypersingularBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_real::<
                                    f64,
                                    LaplaceHypersingularBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_complex::<
                                    c32,
                                    SingleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_complex::<
                                    c64,
                                    SingleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_complex::<
                                    c32,
                                    DoubleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_complex::<
                                    c64,
                                    DoubleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_complex::<
                                    c32,
                                    AdjointDoubleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_complex::<
                                    c64,
                                    AdjointDoubleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_complex::<
                                    c32,
                                    HelmholtzHypersingularBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_internal_complex::<
                                    c64,
                                    HelmholtzHypersingularBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    unsafe fn boundary_assembler_assemble_singular_correction_internal_real<
        T: RlstScalar<Real = T> + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut BoundaryAssemblerWrapper,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) -> *const SparseMatrixWrapper {
        let trial_dim = (*extract_space::<Space>(trial_space)).global_size();
        let test_dim = (*extract_space::<Space>(test_space)).global_size();
        Box::into_raw(Box::new(SparseMatrixWrapper {
            matrix: Box::into_raw(Box::new(
                (*extract_boundary_assembler::<T, Integrand, Kernel>(assembler))
                    .assemble_singular_correction(
                        [test_dim, trial_dim],
                        &*extract_space::<Space>(trial_space),
                        &*extract_space::<Space>(test_space),
                    ),
            )) as *const c_void,
            dtype: (*assembler).dtype,
        }))
    }
    unsafe fn boundary_assembler_assemble_singular_correction_internal_complex<
        T: RlstScalar<Complex = T> + MatrixInverse,
        Integrand: BoundaryIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut BoundaryAssemblerWrapper,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) -> *const SparseMatrixWrapper {
        let trial_dim = (*extract_space::<Space>(trial_space)).global_size();
        let test_dim = (*extract_space::<Space>(test_space)).global_size();
        Box::into_raw(Box::new(SparseMatrixWrapper {
            matrix: Box::into_raw(Box::new(
                (*extract_boundary_assembler::<T, Integrand, Kernel>(assembler))
                    .assemble_singular_correction(
                        [test_dim, trial_dim],
                        &*extract_space::<Space>(trial_space),
                        &*extract_space::<Space>(test_space),
                    ),
            )) as *const c_void,
            dtype: (*assembler).dtype,
        }))
    }
    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_assemble_singular_correction(
        assembler: *mut BoundaryAssemblerWrapper,
        trial_space: *const FunctionSpaceWrapper,
        test_space: *const FunctionSpaceWrapper,
    ) -> *const SparseMatrixWrapper {
        match (*assembler).dtype {
            DType::F32 => {
                assert_eq!((*trial_space).dtype, DType::F32);
                assert_eq!((*test_space).dtype, DType::F32);
            }
            DType::F64 => {
                assert_eq!((*trial_space).dtype, DType::F64);
                assert_eq!((*test_space).dtype, DType::F64);
            }
            DType::C32 => {
                assert_eq!((*trial_space).dtype, DType::F32);
                assert_eq!((*test_space).dtype, DType::F32);
            }
            DType::C64 => {
                assert_eq!((*trial_space).dtype, DType::F64);
                assert_eq!((*test_space).dtype, DType::F64);
            }
        }
        assert_eq!((*test_space).stype, (*trial_space).stype);
        assert_eq!((*test_space).gtype, (*trial_space).gtype);
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_real::<
                                    f32,
                                    SingleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_real::<
                                    f64,
                                    SingleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_real::<
                                    f32,
                                    DoubleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_real::<
                                    f64,
                                    DoubleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_real::<
                                    f32,
                                    AdjointDoubleLayerBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_real::<
                                    f64,
                                    AdjointDoubleLayerBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::F32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_real::<
                                    f32,
                                    LaplaceHypersingularBoundaryIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::F64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_real::<
                                    f64,
                                    LaplaceHypersingularBoundaryIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                BoundaryOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_complex::<
                                    c32,
                                    SingleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_complex::<
                                    c64,
                                    SingleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_complex::<
                                    c32,
                                    DoubleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_complex::<
                                    c64,
                                    DoubleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::AdjointDoubleLayer => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_complex::<
                                    c32,
                                    AdjointDoubleLayerBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_complex::<
                                    c64,
                                    AdjointDoubleLayerBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                BoundaryOperator::Hypersingular => match (*assembler).dtype {
                    DType::C32 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_complex::<
                                    c32,
                                    HelmholtzHypersingularBoundaryIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    DType::C64 => match (*trial_space).stype {
                        SpaceType::SerialFunctionSpace => match (*trial_space).gtype {
                            GridType::SerialSingleElementGrid => {
                                boundary_assembler_assemble_singular_correction_internal_complex::<
                                    c64,
                                    HelmholtzHypersingularBoundaryIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(assembler, trial_space, test_space)
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn boundary_assembler_dtype(
        assembler: *mut BoundaryAssemblerWrapper,
    ) -> u8 {
        (*assembler).dtype as u8
    }

    unsafe fn laplace_boundary_assembler_new_internal<T: RlstScalar + MatrixInverse>(
        operator: BoundaryOperator,
        dtype: DType,
    ) -> *const BoundaryAssemblerWrapper {
        Box::into_raw(Box::new(BoundaryAssemblerWrapper {
            assembler: match operator {
                BoundaryOperator::SingleLayer => Box::into_raw(Box::new(
                    BoundaryAssembler::<T, _, _>::new_laplace_single_layer(),
                )) as *const c_void,
                BoundaryOperator::DoubleLayer => Box::into_raw(Box::new(
                    BoundaryAssembler::<T, _, _>::new_laplace_double_layer(),
                )) as *const c_void,
                BoundaryOperator::AdjointDoubleLayer => Box::into_raw(Box::new(
                    BoundaryAssembler::<T, _, _>::new_laplace_adjoint_double_layer(),
                )) as *const c_void,
                BoundaryOperator::Hypersingular => Box::into_raw(Box::new(
                    BoundaryAssembler::<T, _, _>::new_laplace_hypersingular(),
                )) as *const c_void,
                _ => {
                    panic!("Invalid operator");
                }
            },
            itype: operator,
            ktype: KernelType::Laplace,
            dtype,
        }))
    }

    #[no_mangle]
    pub unsafe extern "C" fn laplace_boundary_assembler_new(
        operator: u8,
        dtype: u8,
    ) -> *const BoundaryAssemblerWrapper {
        let operator = BoundaryOperator::from(operator).unwrap();
        let dtype = DType::from(dtype).unwrap();
        match dtype {
            DType::F32 => laplace_boundary_assembler_new_internal::<f32>(operator, dtype),
            DType::F64 => laplace_boundary_assembler_new_internal::<f64>(operator, dtype),
            _ => {
                panic!("Invalid data type");
            }
        }
    }

    unsafe fn helmholtz_boundary_assembler_new_internal<
        T: RlstScalar<Complex = T> + MatrixInverse,
    >(
        wavenumber: T::Real,
        operator: BoundaryOperator,
        dtype: DType,
    ) -> *const BoundaryAssemblerWrapper {
        Box::into_raw(Box::new(BoundaryAssemblerWrapper {
            assembler: match operator {
                BoundaryOperator::SingleLayer => Box::into_raw(Box::new(
                    BoundaryAssembler::<T, _, _>::new_helmholtz_single_layer(wavenumber),
                )) as *const c_void,
                BoundaryOperator::DoubleLayer => Box::into_raw(Box::new(
                    BoundaryAssembler::<T, _, _>::new_helmholtz_double_layer(wavenumber),
                )) as *const c_void,
                BoundaryOperator::AdjointDoubleLayer => Box::into_raw(Box::new(
                    BoundaryAssembler::<T, _, _>::new_helmholtz_adjoint_double_layer(wavenumber),
                )) as *const c_void,
                BoundaryOperator::Hypersingular => Box::into_raw(Box::new(
                    BoundaryAssembler::<T, _, _>::new_helmholtz_hypersingular(wavenumber),
                )) as *const c_void,
                _ => {
                    panic!("Invalid operator");
                }
            },
            itype: operator,
            ktype: KernelType::Helmholtz,
            dtype,
        }))
    }

    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_boundary_assembler_new(
        wavenumber: *const c_void,
        operator: u8,
        dtype: u8,
    ) -> *const BoundaryAssemblerWrapper {
        let operator = BoundaryOperator::from(operator).unwrap();
        let dtype = DType::from(dtype).unwrap();
        match dtype {
            DType::C32 => helmholtz_boundary_assembler_new_internal::<c32>(
                *(wavenumber as *const f32),
                operator,
                dtype,
            ),
            DType::C64 => helmholtz_boundary_assembler_new_internal::<c64>(
                *(wavenumber as *const f64),
                operator,
                dtype,
            ),
            _ => {
                panic!("Invalid data type");
            }
        }
    }
}

pub mod potential_assembly {
    use super::boundary_assembly::KernelType;
    use super::function::{extract_space, FunctionSpaceWrapper, GridType, SpaceType};
    use super::DType;
    use crate::{
        assembly::kernels::KernelEvaluator,
        assembly::potential::integrands::{
            DoubleLayerPotentialIntegrand, SingleLayerPotentialIntegrand,
        },
        assembly::potential::PotentialAssembler,
        function::SerialFunctionSpace,
        traits::{
            FunctionSpace, KernelEvaluator as KernelEvaluatorTrait, PotentialAssembly,
            PotentialIntegrand,
        },
    };
    use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
    use ndelement::{ciarlet::CiarletElement, types::ReferenceCellType};
    use ndgrid::SingleElementGrid;
    use rlst::{
        c32, c64, rlst_array_from_slice2, rlst_array_from_slice_mut2, MatrixInverse, RlstScalar,
    };
    use std::ffi::c_void;
    use std::slice::{from_raw_parts, from_raw_parts_mut};

    #[derive(Debug, PartialEq, Clone, Copy)]
    #[repr(u8)]
    pub enum PotentialOperator {
        SingleLayer = 0,
        DoubleLayer = 1,
        ElectricField = 2,
        MagneticField = 3,
    }

    impl PotentialOperator {
        fn from(value: u8) -> Option<Self> {
            match value {
                0 => Some(PotentialOperator::SingleLayer),
                1 => Some(PotentialOperator::DoubleLayer),
                2 => Some(PotentialOperator::ElectricField),
                3 => Some(PotentialOperator::MagneticField),
                _ => None,
            }
        }
    }

    #[repr(C)]
    pub struct PotentialAssemblerWrapper {
        pub assembler: *const c_void,
        pub itype: PotentialOperator,
        pub ktype: KernelType,
        pub dtype: DType,
    }
    impl Drop for PotentialAssemblerWrapper {
        fn drop(&mut self) {
            let Self {
                assembler,
                itype,
                ktype,
                dtype,
            } = self;
            match ktype {
                KernelType::Laplace => match itype {
                    PotentialOperator::SingleLayer => match dtype {
                        DType::F32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut PotentialAssembler<
                                        f32,
                                        SingleLayerPotentialIntegrand<f32>,
                                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    >,
                            )
                        }),
                        DType::F64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut PotentialAssembler<
                                        f64,
                                        SingleLayerPotentialIntegrand<f64>,
                                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    PotentialOperator::DoubleLayer => match dtype {
                        DType::F32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut PotentialAssembler<
                                        f32,
                                        DoubleLayerPotentialIntegrand<f32>,
                                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    >,
                            )
                        }),
                        DType::F64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut PotentialAssembler<
                                        f64,
                                        DoubleLayerPotentialIntegrand<f64>,
                                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    _ => {
                        panic!("Invalid operator");
                    }
                },
                KernelType::Helmholtz => match itype {
                    PotentialOperator::SingleLayer => match dtype {
                        DType::C32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut PotentialAssembler<
                                        c32,
                                        SingleLayerPotentialIntegrand<c32>,
                                        KernelEvaluator<c32, Laplace3dKernel<c32>>,
                                    >,
                            )
                        }),
                        DType::C64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut PotentialAssembler<
                                        c64,
                                        SingleLayerPotentialIntegrand<c64>,
                                        KernelEvaluator<c64, Laplace3dKernel<c64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    PotentialOperator::DoubleLayer => match dtype {
                        DType::C32 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut PotentialAssembler<
                                        c32,
                                        DoubleLayerPotentialIntegrand<c32>,
                                        KernelEvaluator<c32, Laplace3dKernel<c32>>,
                                    >,
                            )
                        }),
                        DType::C64 => drop(unsafe {
                            Box::from_raw(
                                *assembler
                                    as *mut PotentialAssembler<
                                        c64,
                                        DoubleLayerPotentialIntegrand<c64>,
                                        KernelEvaluator<c64, Laplace3dKernel<c64>>,
                                    >,
                            )
                        }),
                        _ => {
                            panic!("Invalid data type");
                        }
                    },
                    _ => {
                        panic!("Invalid operator");
                    }
                },
            }
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn free_potential_assembler(a: *mut PotentialAssemblerWrapper) {
        assert!(!a.is_null());
        unsafe { drop(Box::from_raw(a)) }
    }

    pub(crate) unsafe fn extract_potential_assembler<
        T: RlstScalar + MatrixInverse,
        Integrand: PotentialIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
    >(
        assembler: *const PotentialAssemblerWrapper,
    ) -> *const PotentialAssembler<T, Integrand, Kernel> {
        (*assembler).assembler as *const PotentialAssembler<T, Integrand, Kernel>
    }

    pub(crate) unsafe fn extract_potential_assembler_mut<
        T: RlstScalar + MatrixInverse,
        Integrand: PotentialIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
    >(
        assembler: *const PotentialAssemblerWrapper,
    ) -> *mut PotentialAssembler<T, Integrand, Kernel> {
        (*assembler).assembler as *mut PotentialAssembler<T, Integrand, Kernel>
    }

    #[no_mangle]
    pub unsafe extern "C" fn potential_assembler_has_quadrature_degree(
        assembler: *mut PotentialAssemblerWrapper,
        cell: u8,
    ) -> bool {
        let cell = ReferenceCellType::from(cell).unwrap();
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_potential_assembler::<
                        f32,
                        SingleLayerPotentialIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_potential_assembler::<
                        f64,
                        SingleLayerPotentialIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_potential_assembler::<
                        f32,
                        DoubleLayerPotentialIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_potential_assembler::<
                        f64,
                        DoubleLayerPotentialIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_potential_assembler::<
                        c32,
                        SingleLayerPotentialIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_potential_assembler::<
                        c64,
                        SingleLayerPotentialIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_potential_assembler::<
                        c32,
                        DoubleLayerPotentialIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_potential_assembler::<
                        c64,
                        DoubleLayerPotentialIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
        .is_some()
    }

    #[no_mangle]
    pub unsafe extern "C" fn potential_assembler_set_quadrature_degree(
        assembler: *mut PotentialAssemblerWrapper,
        cell: u8,
        degree: usize,
    ) {
        let cell = ReferenceCellType::from(cell).unwrap();
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_potential_assembler_mut::<
                        f32,
                        SingleLayerPotentialIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::F64 => (*extract_potential_assembler_mut::<
                        f64,
                        SingleLayerPotentialIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_potential_assembler_mut::<
                        f32,
                        DoubleLayerPotentialIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::F64 => (*extract_potential_assembler_mut::<
                        f64,
                        DoubleLayerPotentialIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_potential_assembler_mut::<
                        c32,
                        SingleLayerPotentialIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::C64 => (*extract_potential_assembler_mut::<
                        c64,
                        SingleLayerPotentialIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_potential_assembler_mut::<
                        c32,
                        DoubleLayerPotentialIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    DType::C64 => (*extract_potential_assembler_mut::<
                        c64,
                        DoubleLayerPotentialIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_quadrature_degree(cell, degree),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn potential_assembler_quadrature_degree(
        assembler: *mut PotentialAssemblerWrapper,
        cell: u8,
    ) -> usize {
        let cell = ReferenceCellType::from(cell).unwrap();
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_potential_assembler::<
                        f32,
                        SingleLayerPotentialIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_potential_assembler::<
                        f64,
                        SingleLayerPotentialIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_potential_assembler::<
                        f32,
                        DoubleLayerPotentialIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::F64 => (*extract_potential_assembler::<
                        f64,
                        DoubleLayerPotentialIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_potential_assembler::<
                        c32,
                        SingleLayerPotentialIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_potential_assembler::<
                        c64,
                        SingleLayerPotentialIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_potential_assembler::<
                        c32,
                        DoubleLayerPotentialIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    DType::C64 => (*extract_potential_assembler::<
                        c64,
                        DoubleLayerPotentialIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .quadrature_degree(cell),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
        .unwrap()
    }

    #[no_mangle]
    pub unsafe extern "C" fn potential_assembler_set_batch_size(
        assembler: *mut PotentialAssemblerWrapper,
        batch_size: usize,
    ) {
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_potential_assembler_mut::<
                        f32,
                        SingleLayerPotentialIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::F64 => (*extract_potential_assembler_mut::<
                        f64,
                        SingleLayerPotentialIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_potential_assembler_mut::<
                        f32,
                        DoubleLayerPotentialIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::F64 => (*extract_potential_assembler_mut::<
                        f64,
                        DoubleLayerPotentialIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_potential_assembler_mut::<
                        c32,
                        SingleLayerPotentialIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::C64 => (*extract_potential_assembler_mut::<
                        c64,
                        SingleLayerPotentialIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_potential_assembler_mut::<
                        c32,
                        DoubleLayerPotentialIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    DType::C64 => (*extract_potential_assembler_mut::<
                        c64,
                        DoubleLayerPotentialIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .set_batch_size(batch_size),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn potential_assembler_batch_size(
        assembler: *mut PotentialAssemblerWrapper,
    ) -> usize {
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_potential_assembler::<
                        f32,
                        SingleLayerPotentialIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .batch_size(),
                    DType::F64 => (*extract_potential_assembler::<
                        f64,
                        SingleLayerPotentialIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => (*extract_potential_assembler::<
                        f32,
                        DoubleLayerPotentialIntegrand<f32>,
                        KernelEvaluator<f32, Laplace3dKernel<f32>>,
                    >(assembler))
                    .batch_size(),
                    DType::F64 => (*extract_potential_assembler::<
                        f64,
                        DoubleLayerPotentialIntegrand<f64>,
                        KernelEvaluator<f64, Laplace3dKernel<f64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_potential_assembler::<
                        c32,
                        SingleLayerPotentialIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .batch_size(),
                    DType::C64 => (*extract_potential_assembler::<
                        c64,
                        SingleLayerPotentialIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => (*extract_potential_assembler::<
                        c32,
                        DoubleLayerPotentialIntegrand<c32>,
                        KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                    >(assembler))
                    .batch_size(),
                    DType::C64 => (*extract_potential_assembler::<
                        c64,
                        DoubleLayerPotentialIntegrand<c64>,
                        KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                    >(assembler))
                    .batch_size(),
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    unsafe fn potential_assembler_assemble_into_dense_internal_real<
        T: RlstScalar<Real = T> + MatrixInverse,
        Integrand: PotentialIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut PotentialAssemblerWrapper,
        output: *mut c_void,
        space: *const FunctionSpaceWrapper,
        points: *const c_void,
        npts: usize,
    ) {
        let points = rlst_array_from_slice2!(
            from_raw_parts(points as *const T::Real, 3 * npts),
            [3, npts]
        );
        let dim = (*extract_space::<Space>(space)).global_size();
        let mut output = rlst_array_from_slice_mut2!(
            from_raw_parts_mut(output as *mut T, npts * dim),
            [npts, dim]
        );

        (*extract_potential_assembler::<T, Integrand, Kernel>(assembler)).assemble_into_dense(
            &mut output,
            &*extract_space::<Space>(space),
            &points,
        )
    }
    unsafe fn potential_assembler_assemble_into_dense_internal_complex<
        T: RlstScalar<Complex = T> + MatrixInverse,
        Integrand: PotentialIntegrand<T = T>,
        Kernel: KernelEvaluatorTrait<T = T>,
        Space: FunctionSpace<T = T> + Sync,
    >(
        assembler: *mut PotentialAssemblerWrapper,
        output: *mut c_void,
        space: *const FunctionSpaceWrapper,
        points: *const c_void,
        npts: usize,
    ) {
        let points = rlst_array_from_slice2!(
            from_raw_parts(points as *const T::Real, 3 * npts),
            [3, npts]
        );
        let dim = (*extract_space::<Space>(space)).global_size();
        let mut output = rlst_array_from_slice_mut2!(
            from_raw_parts_mut(output as *mut T, npts * dim),
            [npts, dim]
        );

        (*extract_potential_assembler::<T, Integrand, Kernel>(assembler)).assemble_into_dense(
            &mut output,
            &*extract_space::<Space>(space),
            &points,
        )
    }
    #[no_mangle]
    pub unsafe extern "C" fn potential_assembler_assemble_into_dense(
        assembler: *mut PotentialAssemblerWrapper,
        output: *mut c_void,
        space: *const FunctionSpaceWrapper,
        points: *const c_void,
        npts: usize,
    ) {
        match (*assembler).dtype {
            DType::F32 => {
                assert_eq!((*space).dtype, DType::F32);
            }
            DType::F64 => {
                assert_eq!((*space).dtype, DType::F64);
            }
            DType::C32 => {
                assert_eq!((*space).dtype, DType::F32);
            }
            DType::C64 => {
                assert_eq!((*space).dtype, DType::F64);
            }
        }
        match (*assembler).ktype {
            KernelType::Laplace => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::F32 => match (*space).stype {
                        SpaceType::SerialFunctionSpace => match (*space).gtype {
                            GridType::SerialSingleElementGrid => {
                                potential_assembler_assemble_into_dense_internal_real::<
                                    f32,
                                    SingleLayerPotentialIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, space, points, npts
                                );
                            }
                        },
                    },
                    DType::F64 => match (*space).stype {
                        SpaceType::SerialFunctionSpace => match (*space).gtype {
                            GridType::SerialSingleElementGrid => {
                                potential_assembler_assemble_into_dense_internal_real::<
                                    f64,
                                    SingleLayerPotentialIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, space, points, npts
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::F32 => match (*space).stype {
                        SpaceType::SerialFunctionSpace => match (*space).gtype {
                            GridType::SerialSingleElementGrid => {
                                potential_assembler_assemble_into_dense_internal_real::<
                                    f32,
                                    DoubleLayerPotentialIntegrand<f32>,
                                    KernelEvaluator<f32, Laplace3dKernel<f32>>,
                                    SerialFunctionSpace<
                                        f32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, space, points, npts
                                );
                            }
                        },
                    },
                    DType::F64 => match (*space).stype {
                        SpaceType::SerialFunctionSpace => match (*space).gtype {
                            GridType::SerialSingleElementGrid => {
                                potential_assembler_assemble_into_dense_internal_real::<
                                    f64,
                                    DoubleLayerPotentialIntegrand<f64>,
                                    KernelEvaluator<f64, Laplace3dKernel<f64>>,
                                    SerialFunctionSpace<
                                        f64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, space, points, npts
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
            KernelType::Helmholtz => match (*assembler).itype {
                PotentialOperator::SingleLayer => match (*assembler).dtype {
                    DType::C32 => match (*space).stype {
                        SpaceType::SerialFunctionSpace => match (*space).gtype {
                            GridType::SerialSingleElementGrid => {
                                potential_assembler_assemble_into_dense_internal_complex::<
                                    c32,
                                    SingleLayerPotentialIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, space, points, npts
                                );
                            }
                        },
                    },
                    DType::C64 => match (*space).stype {
                        SpaceType::SerialFunctionSpace => match (*space).gtype {
                            GridType::SerialSingleElementGrid => {
                                potential_assembler_assemble_into_dense_internal_complex::<
                                    c64,
                                    SingleLayerPotentialIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, space, points, npts
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                PotentialOperator::DoubleLayer => match (*assembler).dtype {
                    DType::C32 => match (*space).stype {
                        SpaceType::SerialFunctionSpace => match (*space).gtype {
                            GridType::SerialSingleElementGrid => {
                                potential_assembler_assemble_into_dense_internal_complex::<
                                    c32,
                                    DoubleLayerPotentialIntegrand<c32>,
                                    KernelEvaluator<c32, Helmholtz3dKernel<c32>>,
                                    SerialFunctionSpace<
                                        c32,
                                        SingleElementGrid<f32, CiarletElement<f32>>,
                                    >,
                                >(
                                    assembler, output, space, points, npts
                                );
                            }
                        },
                    },
                    DType::C64 => match (*space).stype {
                        SpaceType::SerialFunctionSpace => match (*space).gtype {
                            GridType::SerialSingleElementGrid => {
                                potential_assembler_assemble_into_dense_internal_complex::<
                                    c64,
                                    DoubleLayerPotentialIntegrand<c64>,
                                    KernelEvaluator<c64, Helmholtz3dKernel<c64>>,
                                    SerialFunctionSpace<
                                        c64,
                                        SingleElementGrid<f64, CiarletElement<f64>>,
                                    >,
                                >(
                                    assembler, output, space, points, npts
                                );
                            }
                        },
                    },
                    _ => {
                        panic!("Invalid data type");
                    }
                },
                _ => {
                    panic!("Invalid operator");
                }
            },
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn potential_assembler_dtype(
        assembler: *mut PotentialAssemblerWrapper,
    ) -> u8 {
        (*assembler).dtype as u8
    }

    unsafe fn laplace_potential_assembler_new_internal<T: RlstScalar + MatrixInverse>(
        operator: PotentialOperator,
        dtype: DType,
    ) -> *const PotentialAssemblerWrapper {
        Box::into_raw(Box::new(PotentialAssemblerWrapper {
            assembler: match operator {
                PotentialOperator::SingleLayer => {
                    Box::into_raw(Box::new(
                        PotentialAssembler::<T, _, _>::new_laplace_single_layer(),
                    )) as *const c_void
                }
                PotentialOperator::DoubleLayer => {
                    Box::into_raw(Box::new(
                        PotentialAssembler::<T, _, _>::new_laplace_double_layer(),
                    )) as *const c_void
                }
                _ => {
                    panic!("Invalid operator");
                }
            },
            itype: operator,
            ktype: KernelType::Laplace,
            dtype,
        }))
    }

    #[no_mangle]
    pub unsafe extern "C" fn laplace_potential_assembler_new(
        operator: u8,
        dtype: u8,
    ) -> *const PotentialAssemblerWrapper {
        let operator = PotentialOperator::from(operator).unwrap();
        let dtype = DType::from(dtype).unwrap();
        match dtype {
            DType::F32 => laplace_potential_assembler_new_internal::<f32>(operator, dtype),
            DType::F64 => laplace_potential_assembler_new_internal::<f64>(operator, dtype),
            _ => {
                panic!("Invalid data type");
            }
        }
    }

    unsafe fn helmholtz_potential_assembler_new_internal<
        T: RlstScalar<Complex = T> + MatrixInverse,
    >(
        wavenumber: T::Real,
        operator: PotentialOperator,
        dtype: DType,
    ) -> *const PotentialAssemblerWrapper {
        Box::into_raw(Box::new(PotentialAssemblerWrapper {
            assembler: match operator {
                PotentialOperator::SingleLayer => Box::into_raw(Box::new(
                    PotentialAssembler::<T, _, _>::new_helmholtz_single_layer(wavenumber),
                )) as *const c_void,
                PotentialOperator::DoubleLayer => Box::into_raw(Box::new(
                    PotentialAssembler::<T, _, _>::new_helmholtz_double_layer(wavenumber),
                )) as *const c_void,
                _ => {
                    panic!("Invalid operator");
                }
            },
            itype: operator,
            ktype: KernelType::Helmholtz,
            dtype,
        }))
    }

    #[no_mangle]
    pub unsafe extern "C" fn helmholtz_potential_assembler_new(
        wavenumber: *const c_void,
        operator: u8,
        dtype: u8,
    ) -> *const PotentialAssemblerWrapper {
        let operator = PotentialOperator::from(operator).unwrap();
        let dtype = DType::from(dtype).unwrap();
        match dtype {
            DType::C32 => helmholtz_potential_assembler_new_internal::<c32>(
                *(wavenumber as *const f32),
                operator,
                dtype,
            ),
            DType::C64 => helmholtz_potential_assembler_new_internal::<c64>(
                *(wavenumber as *const f64),
                operator,
                dtype,
            ),
            _ => {
                panic!("Invalid data type");
            }
        }
    }
}
