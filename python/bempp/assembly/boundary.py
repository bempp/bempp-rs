"""Boundary assembly."""

import typing
import numpy as np
import numpy.typing as npt
from bempp._bempprs import lib as _lib, ffi as _ffi
from bempp.function_space import FunctionSpace
from ndelement.reference_cell import ReferenceCellType
from enum import Enum
from _cffi_backend import _CDataBase
from scipy.sparse import coo_matrix

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


def _convert_to_scipy(rs_sparse_mat: _CDataBase, dtype: typing.Type[np.floating]) -> coo_matrix:
    """Convert a pointer to a sparse matrix in Rust to a SciPy COO matrix."""
    shape = (_lib.sparse_shape0(rs_sparse_mat), _lib.sparse_shape1(rs_sparse_mat))
    size = _lib.sparse_data_size(rs_sparse_mat)
    data = np.empty(size, dtype=dtype)
    rows = np.empty(size, dtype=np.uintp)
    cols = np.empty(size, dtype=np.uintp)
    _lib.sparse_data(rs_sparse_mat, _ffi.cast("void*", data.ctypes.data))
    _lib.sparse_rows(rs_sparse_mat, _ffi.cast("uintptr_t*", rows.ctypes.data))
    _lib.sparse_cols(rs_sparse_mat, _ffi.cast("uintptr_t*", cols.ctypes.data))
    _lib.free_sparse_matrix(rs_sparse_mat)
    return coo_matrix((data, (rows, cols)), shape=shape)


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

    def assemble_into_dense(
        self, trial_space: FunctionSpace, test_space: FunctionSpace
    ) -> npt.NDArray[np.floating]:
        """Assemble operator into a dense matrix."""
        assert trial_space.dtype == test_space.dtype == self.dtype
        output = np.zeros(
            (test_space.global_size, trial_space.global_size), dtype=self.dtype, order="F"
        )
        _lib.boundary_assembler_assemble_into_dense(
            self._rs_assembler,
            _ffi.cast("void*", output.ctypes.data),
            trial_space._rs_space,
            test_space._rs_space,
        )
        return output

    def assemble_singular_into_dense(
        self, trial_space: FunctionSpace, test_space: FunctionSpace
    ) -> npt.NDArray[np.floating]:
        """Assemble the singular part of an operator into a dense matrix."""
        assert trial_space.dtype == test_space.dtype == self.dtype
        output = np.zeros(
            (test_space.global_size, trial_space.global_size), dtype=self.dtype, order="F"
        )
        _lib.boundary_assembler_assemble_singular_into_dense(
            self._rs_assembler,
            _ffi.cast("void*", output.ctypes.data),
            trial_space._rs_space,
            test_space._rs_space,
        )
        return output

    def assemble_nonsingular_into_dense(
        self, trial_space: FunctionSpace, test_space: FunctionSpace
    ) -> npt.NDArray[np.floating]:
        """Assemble the non-singular part of an operator into a dense matrix."""
        assert trial_space.dtype == test_space.dtype == self.dtype
        output = np.zeros(
            (test_space.global_size, trial_space.global_size), dtype=self.dtype, order="F"
        )
        _lib.boundary_assembler_assemble_nonsingular_into_dense(
            self._rs_assembler,
            _ffi.cast("void*", output.ctypes.data),
            trial_space._rs_space,
            test_space._rs_space,
        )
        return output

    def assemble_singular(
        self, trial_space: FunctionSpace, test_space: FunctionSpace
    ) -> coo_matrix:
        """Assemble the singular part of an operator into a CSR matrix."""
        assert trial_space.dtype == test_space.dtype == self.dtype
        return _convert_to_scipy(
            _lib.boundary_assembler_assemble_singular(
                self._rs_assembler,
                trial_space._rs_space,
                test_space._rs_space,
            ),
            self.dtype,
        )

    def assemble_singular_correction(
        self, trial_space: FunctionSpace, test_space: FunctionSpace
    ) -> coo_matrix:
        """Assemble the singular correction of an operator into a CSR matrix."""
        assert trial_space.dtype == test_space.dtype == self.dtype
        return _convert_to_scipy(
            _lib.boundary_assembler_assemble_singular_correction(
                self._rs_assembler,
                trial_space._rs_space,
                test_space._rs_space,
            ),
            self.dtype,
        )

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
