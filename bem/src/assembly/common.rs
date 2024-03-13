//! Common utility functions
use rlst_dense::types::RlstScalar;

pub(crate) struct RawData2D<T: RlstScalar> {
    pub(crate) data: *mut T,
    pub(crate) shape: [usize; 2],
}

unsafe impl<T: RlstScalar> Sync for RawData2D<T> {}

pub(crate) struct SparseMatrixData<T: RlstScalar> {
    pub(crate) data: Vec<T>,
    pub(crate) rows: Vec<usize>,
    pub(crate) cols: Vec<usize>,
    pub(crate) shape: [usize; 2],
}

impl<T: RlstScalar> SparseMatrixData<T> {
    pub(crate) fn new(shape: [usize; 2]) -> Self {
        Self {
            data: vec![],
            rows: vec![],
            cols: vec![],
            shape,
        }
    }
    pub(crate) fn new_known_size(shape: [usize; 2], size: usize) -> Self {
        Self {
            data: Vec::with_capacity(size),
            rows: Vec::with_capacity(size),
            cols: Vec::with_capacity(size),
            shape,
        }
    }
    pub(crate) fn add(&mut self, other: SparseMatrixData<T>) {
        debug_assert!(self.shape[0] == other.shape[0]);
        debug_assert!(self.shape[1] == other.shape[1]);
        self.rows.extend(&other.rows);
        self.cols.extend(&other.cols);
        self.data.extend(&other.data);
    }
    pub(crate) fn sum(&self, other: SparseMatrixData<T>) -> SparseMatrixData<T> {
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
