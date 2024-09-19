"""Potential assembly."""

import typing
import numpy as np
import numpy.typing as npt
from bempp._bempprs import lib as _lib, ffi as _ffi
from bempp.function_space import FunctionSpace
from ndelement.reference_cell import ReferenceCellType
from enum import Enum
from _cffi_backend import _CDataBase

_dtypes = {
    0: np.float32,
    1: np.float64,
}
_ctypes = {
    np.float32: "float",
    np.float64: "double",
}


def _dtype_value(dtype):
    """Get the u8 identifier for a data type."""
    for i, j in _dtypes.items():
        if j == dtype:
            return i
    raise TypeError("Invalid data type")


class OperatorType(Enum):
    """Operator type."""

    SingleLayer = 0
    DoubleLayer = 1


class PotentialAssembler(object):
    """Potential assembler."""

    def __init__(self, rs_assembler: _CDataBase, owned: bool = True):
        """Initialise."""
        self._rs_assembler = rs_assembler
        self._owned = owned

    def __del__(self):
        """Delete."""
        if self._owned:
            _lib.free_potential_assembler(self._rs_assembler)

    def assemble_into_dense(
        self,
        space: FunctionSpace,
        points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Assemble operator into a dense matrix."""
        assert space.dtype == points.dtype == self.dtype
        output = np.zeros((points.shape[0], space.global_size), dtype=self.dtype, order="F")
        _lib.potential_assembler_assemble_into_dense(
            self._rs_assembler,
            _ffi.cast("void*", output.ctypes.data),
            space._rs_space,
            _ffi.cast("void*", points.ctypes.data),
            points.shape[0],
        )
        return output

    def set_quadrature_degree(self, cell: ReferenceCellType, degree: int):
        """Set the (non-singular) quadrature degree for a cell type."""
        if not _lib.potential_assembler_has_quadrature_degree(self._rs_assembler, cell.value):
            raise ValueError(f"Invalid cell type: {cell}")
        _lib.potential_assembler_set_quadrature_degree(self._rs_assembler, cell.value, degree)

    def quadrature_degree(self, cell: ReferenceCellType) -> int:
        """Get the (non-singular) quadrature degree for a cell type."""
        if not _lib.potential_assembler_has_quadrature_degree(self._rs_assembler, cell.value):
            raise ValueError(f"Invalid cell type: {cell}")
        return _lib.potential_assembler_quadrature_degree(self._rs_assembler, cell.value)

    def set_batch_size(self, batch_size: int):
        """Set the batch size."""
        _lib.potential_assembler_set_batch_size(self._rs_assembler, batch_size)

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return _lib.potential_assembler_batch_size(self._rs_assembler)

    @property
    def dtype(self):
        """Data type."""
        return _dtypes[_lib.potential_assembler_dtype(self._rs_assembler)]

    @property
    def _ctype(self):
        """C data type."""
        return _ctypes[self.dtype]


def create_laplace_assembler(
    operator_type: OperatorType, dtype: typing.Type[np.floating] = np.float64
):
    """Create a Laplace assembler."""
    return PotentialAssembler(
        _lib.laplace_potential_assembler_new(operator_type.value, _dtype_value(dtype))
    )
