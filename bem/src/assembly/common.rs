use rlst_dense::types::RlstScalar;

pub struct RawData2D<T: RlstScalar> {
    pub data: *mut T,
    pub shape: [usize; 2],
}

unsafe impl<T: RlstScalar> Sync for RawData2D<T> {}

pub struct SparseMatrixData<T: RlstScalar> {
    pub data: Vec<T>,
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub shape: [usize; 2],
}

impl<T: RlstScalar> SparseMatrixData<T> {
    pub fn new(shape: [usize; 2]) -> Self {
        Self {
            data: vec![],
            rows: vec![],
            cols: vec![],
            shape,
        }
    }
    pub fn new_known_size(shape: [usize; 2], size: usize) -> Self {
        Self {
            data: Vec::with_capacity(size),
            rows: Vec::with_capacity(size),
            cols: Vec::with_capacity(size),
            shape,
        }
    }
    pub fn add(&mut self, other: SparseMatrixData<T>) {
        debug_assert!(self.shape[0] == other.shape[0]);
        debug_assert!(self.shape[1] == other.shape[1]);
        self.rows.extend(&other.rows);
        self.cols.extend(&other.cols);
        self.data.extend(&other.data);
    }
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
