"""Boundary assembly."""

import typing
import numpy as np
import numpy.typing as npt
from bempp._bempprs import lib as _lib, ffi as _ffi
from bempp.function_space import FunctionSpace
from ndelement.reference_cell import ReferenceCellType
from enum import Enum
from _cffi_backend import _CDataBase
from scipy.sparse import csr_matrix

_rtypes = {
    np.float32: _lib.DType_F32,
    np.float64: _lib.DType_F64,
    np.complex64: _lib.DType_C32,
    np.complex128: _lib.DType_C64,
}
_dtypes = {j: i for i, j in _rtypes.items()}


def _convert_to_scipy(rs_sparse_mat: _CDataBase, dtype: typing.Type[np.floating]) -> csr_matrix:
    """Convert a pointer to a sparse matrix in Rust to a SciPy CSR matrix."""
    shape = (_lib.matrix_shape0(rs_sparse_mat), _lib.matrix_shape1(rs_sparse_mat))
    data = np.empty(_lib.matrix_csr_data_size(rs_sparse_mat), dtype=dtype)
    indices = np.empty(_lib.matrix_csr_indices_size(rs_sparse_mat), dtype=np.uintp)
    indptr = np.empty(_lib.matrix_csr_indptr_size(rs_sparse_mat), dtype=np.uintp)
    _lib.matrix_csr_data_copy(rs_sparse_mat, _ffi.cast("void*", data.ctypes.data))
    _lib.matrix_csr_indices_copy(rs_sparse_mat, _ffi.cast("uintptr_t*", indices.ctypes.data))
    _lib.matrix_csr_indptr_copy(rs_sparse_mat, _ffi.cast("uintptr_t*", indptr.ctypes.data))
    _lib.matrix_t_free(rs_sparse_mat)
    return csr_matrix((data, indices, indptr), shape=shape)


class OperatorType(Enum):
    """Operator type."""

    SingleLayer = _lib.OperatorType_SingleLayer
    DoubleLayer = _lib.OperatorType_DoubleLayer
    AdjointDoubleLayer = _lib.OperatorType_AdjointDoubleLayer
    Hypersingular = _lib.OperatorType_Hypersingular
    ElectricField = _lib.OperatorType_ElectricField
    MagneticField = _lib.OperatorType_MagneticField


class BoundaryAssemblerOptions(object):
    """Boundary assembler options."""

    def __init__(self):
        self._read_only = False
        self._rs_options = _lib.boundary_assembler_options_new()

    def set_regular_quadrature_degree(self, cell: ReferenceCellType, degree: int):
        """Set the (non-singular) quadrature degree for a cell type."""
        if self._read_only:
            raise ValueError("Cannot edit options already being used by an assembler")
        if not _lib.boundary_assembler_options_has_regular_quadrature_degree(
            self._rs_options, cell.value
        ):
            raise ValueError(f"Invalid cell type: {cell}")
        _lib.boundary_assembler_options_set_regular_quadrature_degree(
            self._rs_options, cell.value, degree
        )

    def regular_quadrature_degree(self, cell: ReferenceCellType) -> int:
        """Get the (non-singular) quadrature degree for a cell type."""
        if not _lib.boundary_assembler_options_has_regular_quadrature_degree(
            self._rs_options, cell.value
        ):
            raise ValueError(f"Invalid cell type: {cell}")
        return _lib.boundary_assembler_options_get_regular_quadrature_degree(
            self._rs_options, cell.value
        )

    def set_singular_quadrature_degree(
        self, cell0: ReferenceCellType, cell1: ReferenceCellType, degree: int
    ):
        """Set the singular quadrature degree for a cell type."""
        if self._read_only:
            raise ValueError("Cannot edit options already being used by an assembler")
        if not _lib.boundary_assembler_options_has_singular_quadrature_degree(
            self._rs_options, cell0.value, cell1.value
        ):
            raise ValueError(f"Invalid cell pair: {cell0} and {cell1}")
        _lib.boundary_assembler_options_set_singular_quadrature_degree(
            self._rs_options, cell0.value, cell1.value, degree
        )

    def singular_quadrature_degree(self, cell0: ReferenceCellType, cell1: ReferenceCellType) -> int:
        """Get the singular quadrature degree for a cell type."""
        if not _lib.boundary_assembler_options_has_singular_quadrature_degree(
            self._rs_options, cell0.value, cell1.value
        ):
            raise ValueError(f"Invalid cell pair: {cell0} and {cell1}")
        return _lib.boundary_assembler_options_get_singular_quadrature_degree(
            self._rs_options, cell0.value, cell1.value
        )

    def set_batch_size(self, batch_size: int):
        """Set the batch size."""
        if self._read_only:
            raise ValueError("Cannot edit options already being used by an assembler")
        _lib.boundary_assembler_options_set_batch_size(self._rs_options, batch_size)

    def batch_size(self) -> int:
        """Get the batch size."""
        return _lib.boundary_assembler_options_get_batch_size(self._rs_options)


class BoundaryAssembler(object):
    """Boundary assembler."""

    def __init__(
        self, options: BoundaryAssemblerOptions, rs_assembler: _CDataBase, owned: bool = True
    ):
        """Initialise."""
        self.options = options
        self.options._read_only = True
        self._rs_assembler = rs_assembler
        self._owned = owned

    def __del__(self):
        """Delete."""
        if self._owned:
            _lib.boundary_assembler_t_free(self._rs_assembler)

    def assemble(
        self, trial_space: FunctionSpace, test_space: FunctionSpace
    ) -> npt.NDArray[np.floating]:
        """Assemble operator into a dense matrix."""
        assert trial_space.dtype == test_space.dtype == self.dtype
        output = np.zeros(
            (test_space.global_size, trial_space.global_size), dtype=self.dtype, order="F"
        )
        _lib.boundary_assembler_assemble_into_memory(
            self._rs_assembler,
            trial_space._rs_space,
            test_space._rs_space,
            _ffi.cast("void*", output.ctypes.data),
        )
        return output

    def assemble_singular(
        self, trial_space: FunctionSpace, test_space: FunctionSpace
    ) -> csr_matrix:
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
    ) -> csr_matrix:
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

    @property
    def dtype(self):
        """Data type."""
        return _dtypes[_lib.boundary_assembler_dtype(self._rs_assembler)]


def create_laplace_assembler(
    operator_type: OperatorType,
    options: BoundaryAssemblerOptions = BoundaryAssemblerOptions(),
    dtype: typing.Type[np.floating] = np.float64,
):
    """Create a Laplace assembler."""
    return BoundaryAssembler(
        options,
        _lib.laplace_boundary_assembler_new(
            options._rs_options, operator_type.value, _rtypes[dtype]
        ),
    )
