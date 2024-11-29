"""Function space."""

import typing
import numpy as np
from bempp._bempprs import lib as _lib, ffi as _ffi
from ndgrid._ndgridrs import lib as _gridlib
from ndgrid.grid import Grid
from ndgrid.ownership import Owned, Ghost
from ndelement.ciarlet import ElementFamily, CiarletElement
from ndelement.reference_cell import ReferenceCellType
from _cffi_backend import _CDataBase

_dtypes = {
    0: np.float32,
    1: np.float64,
}
_rtypes = {
    np.float32: _lib.DType_F32,
    np.float64: _lib.DType_F64,
    np.complex64: _lib.DType_C32,
    np.complex128: _lib.DType_C64,
}
_cells_ndgrid_to_bempp = {
    getattr(_gridlib, f"ReferenceCellType_{cell}"): getattr(_lib, f"ReferenceCellType_{cell}")
    for cell in [
        "Point",
        "Interval",
        "Triangle",
        "Quadrilateral",
        "Tetrahedron",
        "Hexahedron",
        "Prism",
        "Pyramid",
    ]
}
_dtype_ndgrid_to_bempp = {
    getattr(_gridlib, f"DType_{d}"): getattr(_lib, f"DType_{d}")
    for d in ["F32", "F64", "C32", "C64"]
}


class FunctionSpace(object):
    """Function space."""

    def __init__(self, grid: Grid, family: ElementFamily, rs_space: _CDataBase, owned: bool = True):
        """Initialise."""
        self._rs_space = rs_space
        self._owned = owned
        self._grid = grid
        self._family = family

    def __del__(self):
        """Delete."""
        if self._owned:
            _lib.space_t_free(self._rs_space)
            _lib.element_family_t_free_bempp(self._family._bempp_rs_family)
            del self._family._bempp_rs_family
            _lib.grid_t_free_bempp(self._grid._bempp_rs_grid)
            del self._grid._bempp_rs_grid

    def element(self, entity: ReferenceCellType) -> CiarletElement:
        """Get the grid that this space is defined on."""
        return self._family.element(entity)

    def get_local_dof_numbers(self, entity_dim: int, entity_index: int) -> typing.List[int]:
        """Get the local DOF numbers associated with an entity."""
        dofs = np.empty(
            _lib.space_get_local_dof_numbers_size(self._rs_space, entity_dim, entity_index),
            dtype=np.uintp,
        )
        _lib.space_get_local_dof_numbers(
            self._rs_space, entity_dim, entity_index, _ffi.cast("uintptr_t*", dofs.ctypes.data)
        )
        return [int(i) for i in dofs]

    def cell_dofs(self, cell: int) -> typing.Optional[typing.List[int]]:
        """Get the local DOF numbers associated with a cell."""
        if not _lib.space_has_cell_dofs(self._rs_space, cell):
            return None
        dofs = np.empty(_lib.space_cell_dofs_size(self._rs_space, cell), dtype=np.uintp)
        _lib.space_cell_dofs(self._rs_space, cell, _ffi.cast("uintptr_t*", dofs.ctypes.data))
        return [int(i) for i in dofs]

    def global_dof_index(self, local_dof_index: int) -> typing.Optional[typing.List[int]]:
        """Get the global DOF number for a local DOF."""
        return _lib.space_global_dof_index(self._rs_space, local_dof_index)

    def ownership(self, local_dof_index) -> typing.Union[Owned, Ghost]:
        """The ownership of a local DOF."""
        if _lib.space_is_owned(self._rs_space, local_dof_index):
            return Owned()
        else:
            return Ghost(
                _lib.space_ownership_process(self._rs_space, local_dof_index),
                _lib.space_ownership_index(self._rs_space, local_dof_index),
            )

    @property
    def dtype(self):
        """Data type."""
        return _dtypes[_lib.space_dtype(self._rs_space)]

    @property
    def local_size(self) -> int:
        """Number of DOFs on current process."""
        return _lib.space_local_size(self._rs_space)

    @property
    def global_size(self) -> int:
        """Number of DOFs on all processes."""
        return _lib.space_global_size(self._rs_space)

    @property
    def is_serial(self) -> bool:
        """Indicates whether the function space is stored in serial."""
        return _lib.space_is_serial(self._rs_space)

    @property
    def grid(self) -> Grid:
        """Get the grid that this space is defined on."""
        return self._grid


def function_space(grid: Grid, family: ElementFamily) -> FunctionSpace:
    """Create a function space."""
    if not hasattr(grid, "_bempp_rs_grid"):
        grid._bempp_rs_grid = recreate_grid(grid)
    if not hasattr(family, "_bempp_rs_family"):
        family._bempp_rs_family = recreate_element_family(family)

    return FunctionSpace(
        grid, family, _lib.function_space(grid._bempp_rs_grid, family._bempp_rs_family)
    )


def recreate_element_family(family: ElementFamily):
    return _lib.create_lagrange_family_bempp(
        family.degree, getattr(_lib, f"Continuity_{family.continuity.name}"), _rtypes[family.dtype]
    )


def recreate_grid(grid: Grid):
    if _gridlib.grid_type(grid._rs_grid) == _gridlib.GridType_SingleElementGrid:
        cdata = _gridlib.single_element_grid_cdata(grid._rs_grid)
        entity_types = np.array(
            [_cells_ndgrid_to_bempp[cdata.entity_types[i]] for i in range(cdata.tdim + 1)],
            dtype=np.uint8,
        )
        bempp_grid = _lib.single_element_grid_borrowed_create_bempp(
            cdata.tdim,
            cdata.id_sizes,
            cdata.id_pointers,
            _ffi.cast("ReferenceCellType*", entity_types.ctypes.data),
            cdata.entity_counts,
            cdata.downward_connectivity,
            cdata.downward_connectivity_shape0,
            cdata.upward_connectivity,
            cdata.upward_connectivity_lens,
            cdata.points,
            cdata.gdim,
            cdata.npoints,
            _dtype_ndgrid_to_bempp[cdata.dtype],
            cdata.cells,
            cdata.points_per_cell,
            cdata.ncells,
            cdata.geometry_degree,
        )
        _gridlib.internal_data_container_free(cdata.internal_storage)
        return bempp_grid
    else:
        raise ValueError("Unsupported grid type")
