//! Common utility functions
use crate::traits::CellGeometry;
pub(crate) use green_kernels::types::GreenKernelEvalType;
use ndgrid::traits::Grid;
use rlst::{Array, BaseArray, MatrixInverse, RlstScalar, VectorContainer};

pub(crate) type RlstArray<T, const DIM: usize> =
    Array<T, BaseArray<T, VectorContainer<T>, DIM>, DIM>;

pub(crate) fn equal_grids<TestGrid: Grid, TrialGrid: Grid>(
    test_grid: &TestGrid,
    trial_grid: &TrialGrid,
) -> bool {
    std::ptr::addr_of!(*test_grid) as usize == std::ptr::addr_of!(*trial_grid) as usize
}

/// Raw 2D data
pub(crate) struct RawData2D<T: RlstScalar + MatrixInverse> {
    /// Array containting data
    pub(crate) data: *mut T,
    /// Shape of data
    pub(crate) shape: [usize; 2],
}

unsafe impl<T: RlstScalar + MatrixInverse> Sync for RawData2D<T> {}

/// Data for a sparse matrix
pub struct SparseMatrixData<T: RlstScalar + MatrixInverse> {
    /// Data
    pub data: Vec<T>,
    /// Rows
    pub rows: Vec<usize>,
    /// Columns
    pub cols: Vec<usize>,
    /// Shape of the matrix
    pub shape: [usize; 2],
}

impl<T: RlstScalar + MatrixInverse> SparseMatrixData<T> {
    /// Create new sparse matrix
    pub fn new(shape: [usize; 2]) -> Self {
        Self {
            data: vec![],
            rows: vec![],
            cols: vec![],
            shape,
        }
    }
    /// Create new sparse matrix with a known size
    pub fn new_known_size(shape: [usize; 2], size: usize) -> Self {
        Self {
            data: Vec::with_capacity(size),
            rows: Vec::with_capacity(size),
            cols: Vec::with_capacity(size),
            shape,
        }
    }
    /// Add another sparse matrix to this matrix
    pub fn add(&mut self, other: SparseMatrixData<T>) {
        debug_assert!(self.shape[0] == other.shape[0]);
        debug_assert!(self.shape[1] == other.shape[1]);
        self.rows.extend(&other.rows);
        self.cols.extend(&other.cols);
        self.data.extend(&other.data);
    }
}

unsafe impl<T: RlstScalar + MatrixInverse> Sync for SparseMatrixData<T> {}

pub(crate) struct AssemblerGeometry<'a, T: RlstScalar<Real = T>> {
    points: &'a RlstArray<T, 2>,
    normals: &'a RlstArray<T, 2>,
    jacobians: &'a RlstArray<T, 2>,
    jdets: &'a [T],
}

impl<'a, T: RlstScalar<Real = T>> AssemblerGeometry<'a, T> {
    pub(crate) fn new(
        points: &'a RlstArray<T, 2>,
        normals: &'a RlstArray<T, 2>,
        jacobians: &'a RlstArray<T, 2>,
        jdets: &'a [T],
    ) -> Self {
        Self {
            points,
            normals,
            jacobians,
            jdets,
        }
    }
}

impl<'a, T: RlstScalar<Real = T>> CellGeometry for AssemblerGeometry<'a, T> {
    type T = T;
    fn points(&self) -> &RlstArray<T, 2> {
        self.points
    }
    fn normals(&self) -> &RlstArray<T, 2> {
        self.normals
    }
    fn jacobians(&self) -> &RlstArray<T, 2> {
        self.jacobians
    }
    fn jdets(&self) -> &[T] {
        self.jdets
    }
}
