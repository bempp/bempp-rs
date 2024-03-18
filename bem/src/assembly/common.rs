//! Common utility functions
use rlst::RlstScalar;

/// Raw 2D data
pub struct RawData2D<T: RlstScalar> {
    /// Array containting data
    pub data: *mut T,
    /// Shape of data
    pub shape: [usize; 2],
}

unsafe impl<T: RlstScalar> Sync for RawData2D<T> {}

/// Data for a sparse matrix
pub struct SparseMatrixData<T: RlstScalar> {
    /// Data
    pub data: Vec<T>,
    /// Rows
    pub rows: Vec<usize>,
    /// Columns
    pub cols: Vec<usize>,
    /// Shape of the matrix
    pub shape: [usize; 2],
}

impl<T: RlstScalar> SparseMatrixData<T> {
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
    /// Compute the sum of this sparse matrix and another sparse matrix
    pub fn sum(&self, other: SparseMatrixData<T>) -> SparseMatrixData<T> {
        debug_assert!(self.shape[0] == other.shape[0]);
        debug_assert!(self.shape[1] == other.shape[1]);
        let mut out = SparseMatrixData::<T>::new(self.shape);
        out.rows.extend(&self.rows);
        out.cols.extend(&self.cols);
        out.data.extend(&self.data);
        out.rows.extend(&other.rows);
        out.cols.extend(&other.cols);
        out.data.extend(&other.data);
        out
    }
}

unsafe impl<T: RlstScalar> Sync for SparseMatrixData<T> {}
