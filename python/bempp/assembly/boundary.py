"""Boundary assembly."""

import typing
import numpy as np
from bempp._bempprs import lib as _lib
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
    AdjointDoubleLayer = 2
    Hypersingular = 3
    ElectricField = 4
    MagneticField = 5


class BoundaryAssembler(object):
    """Boundary assembler."""

    def __init__(self, rs_assembler: _CDataBase, owned: bool = True):
        """Initialise."""
        self._rs_assembler = rs_assembler
        self._owned = owned

    def __del__(self):
        """Delete."""
        if self._owned:
            _lib.free_boundary_assembler(self._rs_assembler)

    def set_quadrature_degree(self, cell: ReferenceCellType, degree: int):
        """Set the (non-singular) quadrature degree for a cell type."""
        if not _lib.boundary_assembler_has_quadrature_degree(self._rs_assembler, cell.value):
            raise ValueError(f"Invalid cell type: {cell}")
        _lib.boundary_assembler_set_quadrature_degree(self._rs_assembler, cell.value, degree)

    def quadrature_degree(self, cell: ReferenceCellType) -> int:
        """Get the (non-singular) quadrature degree for a cell type."""
        if not _lib.boundary_assembler_has_quadrature_degree(self._rs_assembler, cell.value):
            raise ValueError(f"Invalid cell type: {cell}")
        return _lib.boundary_assembler_quadrature_degree(self._rs_assembler, cell.value)

    def set_singular_quadrature_degree(
        self, cell0: ReferenceCellType, cell1: ReferenceCellType, degree: int
    ):
        """Set the singular quadrature degree for a cell type."""
        if not _lib.boundary_assembler_has_singular_quadrature_degree(
            self._rs_assembler, cell0.value, cell1.value
        ):
            raise ValueError(f"Invalid cell pair: {cell0} and {cell1}")
        _lib.boundary_assembler_set_singular_quadrature_degree(
            self._rs_assembler, cell0.value, cell1.value, degree
        )

    def singular_quadrature_degree(self, cell0: ReferenceCellType, cell1: ReferenceCellType) -> int:
        """Get the singular_quadrature degree for a cell type."""
        if not _lib.boundary_assembler_has_singular_quadrature_degree(
            self._rs_assembler, cell0.value, cell1.value
        ):
            raise ValueError(f"Invalid cell pair: {cell0} and {cell1}")
        return _lib.boundary_assembler_singular_quadrature_degree(
            self._rs_assembler, cell0.value, cell1.value
        )

    def set_batch_size(self, batch_size: int):
        """Set the batch size."""
        _lib.boundary_assembler_set_batch_size(self._rs_assembler, batch_size)

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return _lib.boundary_assembler_batch_size(self._rs_assembler)

    @property
    def dtype(self):
        """Data type."""
        return _dtypes[_lib.boundary_assembler_dtype(self._rs_assembler)]

    @property
    def _ctype(self):
        """C data type."""
        return _ctypes[self.dtype]


def create_laplace_assembler(
    operator_type: OperatorType, dtype: typing.Type[np.floating] = np.float64
):
    """Create a Laplace assembler."""
    return BoundaryAssembler(
        _lib.laplace_boundary_assembler_new(operator_type.value, _dtype_value(dtype))
    )
