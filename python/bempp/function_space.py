"""Function space."""

from bempp._bempprs import lib as _lib, ffi as _ffi
from ndgrid.grid import Grid
from ndelement.ciarlet import ElementFamily


class FunctionSpace(object):
    """Function space."""

    def __init__(self, rs_space):
        """Initialise."""
        self._rs_space = rs_space

    def __del__(self):
        """Delete."""
        _lib.free_space(self._rs_space)

    @property
    def local_size(self) -> int:
        """Number of DOFs on current process."""
        return _lib.space_local_size(self._rs_space)

    @property
    def global_size(self) -> int:
        """Number of DOFs on all processes."""
        return _lib.space_global_size(self._rs_space)


def function_space(grid: Grid, family: ElementFamily) -> FunctionSpace:
    return FunctionSpace(
        _lib.space_new(_ffi.cast("void*", grid._rs_grid), _ffi.cast("void*", family._rs_family))
    )
