"""Function space."""

import typing
import numpy as np
from bempp._bempprs import lib as _lib, ffi as _ffi
from ndelement._ndelementrs import ffi as _elementffi
from ndgrid._ndgridrs import ffi as _gridffi
from ndgrid.grid import Grid
from ndgrid.ownership import Owned, Ghost
from ndelement.ciarlet import ElementFamily, CiarletElement
from ndelement.reference_cell import ReferenceCellType

_dtypes = {
    0: np.float32,
    1: np.float64,
}
_ctypes = {
    np.float32: "float",
    np.float64: "double",
}


class FunctionSpace(object):
    """Function space."""

    def __init__(self, rs_space):
        """Initialise."""
        self._rs_space = rs_space

    def __del__(self):
        """Delete."""
        _lib.free_space(self._rs_space)

    def element(self, entity: ReferenceCellType) -> CiarletElement:
        """Get the grid that this space is defined on."""
        return CiarletElement(
            _elementffi.cast(
                "CiarletElementWrapper*", _lib.space_element(self._rs_space, entity.value)
            ),
            owned=False,
        )

    # TODO: test
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

    # TODO: test
    def cell_dofs(self, cell: int) -> typing.Optional[typing.List[int]]:
        """Get the local DOF numbers associated with a cell."""
        if not _lib.space_has_cell_dofs(self._rs_space, cell):
            return None
        dofs = np.empty(_lib.space_cell_dofs_size(self._rs_space, cell), dtype=np.uintp)
        _lib.space_cell_dofs(self._rs_space, cell, _ffi.cast("uintptr_t*", dofs.ctypes.data))
        return [int(i) for i in dofs]

    # TODO: test
    def global_dof_index(self, local_dof_index: int) -> typing.Optional[typing.List[int]]:
        """Get the global DOF number for a local DOF."""
        return _lib.space_global_dof_index(self._rs_space, local_dof_index)

    # TODO: test
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
    def _ctype(self):
        """C data type."""
        return _ctypes[self.dtype]

    @property
    def local_size(self) -> int:
        """Number of DOFs on current process."""
        return _lib.space_local_size(self._rs_space)

    @property
    def global_size(self) -> int:
        """Number of DOFs on all processes."""
        return _lib.space_global_size(self._rs_space)

    # TODO: test
    @property
    def is_serial(self) -> bool:
        """Indicates whether the function space is stored in serial."""
        return _lib.space_serial(self._rs_space)

    @property
    def grid(self) -> Grid:
        """Get the grid that this space is defined on."""
        return Grid(_gridffi.cast("GridWrapper*", _lib.space_grid(self._rs_space)), owned=False)


def function_space(grid: Grid, family: ElementFamily) -> FunctionSpace:
    return FunctionSpace(
        _lib.space_new(_ffi.cast("void*", grid._rs_grid), _ffi.cast("void*", family._rs_family))
    )
