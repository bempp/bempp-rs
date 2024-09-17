"""Function space."""

import numpy as np
from bempp._bempprs import lib as _lib, ffi as _ffi
from ndgrid._ndgridrs import ffi as _gridffi
from ndgrid.grid import Grid
from ndelement.ciarlet import ElementFamily

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
        return Grid(_lib.space_grid(self._rs_space), owned=False)
        return Grid(_gridffi.cast("GridWrapper*", _lib.space_grid(self._rs_space)), owned=False)


def function_space(grid: Grid, family: ElementFamily) -> FunctionSpace:
    return FunctionSpace(
        _lib.space_new(_ffi.cast("void*", grid._rs_grid), _ffi.cast("void*", family._rs_family))
    )
