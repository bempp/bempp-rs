use super::MatrixT;
use c_api_tools::concretise_types;
use ndgrid::types::{Array2D, Array2DBorrowed};
use rlst::{c32, c64, CsrMatrix, Shape};

#[concretise_types(
    gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
    field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["CsrMatrix<{{dtype}}>", "Array2D<{{dtype}}>", "Array2DBorrowed<{{dtype}}>"]),
)]
pub fn matrix_shape0<Mat: Shape<2>>(matrix: &Mat) -> usize {
    matrix.shape()[0]
}

#[concretise_types(
    gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
    field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["CsrMatrix<{{dtype}}>", "Array2D<{{dtype}}>", "Array2DBorrowed<{{dtype}}>"]),
)]
pub fn matrix_shape1<Mat: Shape<2>>(matrix: &Mat) -> usize {
    matrix.shape()[1]
}

pub mod dense_matrix {
    use super::super::MatrixT;
    use c_api_tools::concretise_types;
    use ndgrid::types::{Array2D, Array2DBorrowed};
    use rlst::{c32, c64, RawAccess, RlstScalar};
    use std::ffi::c_void;

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["Array2D<{{dtype}}>", "Array2DBorrowed<{{dtype}}>"]),
    )]
    pub fn matrix_dense_data<T: RlstScalar>(matrix: &impl RawAccess<Item = T>) -> *const c_void {
        matrix.data().as_ptr() as *const c_void
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["Array2D<{{dtype}}>", "Array2DBorrowed<{{dtype}}>"]),
    )]
    pub unsafe fn matrix_dense_data_copy<T: RlstScalar>(
        matrix: &impl RawAccess<Item = T>,
        data: *mut c_void,
    ) {
        let data = data as *mut T;
        for (i, j) in matrix.data().iter().enumerate() {
            *data.add(i) = *j;
        }
    }
}

pub mod csr_matrix {
    use super::MatrixT;
    use c_api_tools::concretise_types;
    use rlst::{c32, c64, CsrMatrix, RlstScalar};
    use std::ffi::c_void;
    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["CsrMatrix<{{dtype}}>"]),
    )]
    pub fn matrix_csr_data_size<T: RlstScalar>(matrix: &CsrMatrix<T>) -> usize {
        matrix.data().len()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["CsrMatrix<{{dtype}}>"]),
    )]
    pub fn matrix_csr_data<T: RlstScalar>(matrix: &CsrMatrix<T>) -> *const c_void {
        matrix.data().as_ptr() as *const c_void
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["CsrMatrix<{{dtype}}>"]),
    )]
    pub unsafe fn matrix_csr_data_copy<T: RlstScalar>(matrix: &CsrMatrix<T>, data: *mut c_void) {
        let data = data as *mut T;
        for (i, j) in matrix.data().iter().enumerate() {
            *data.add(i) = *j;
        }
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["CsrMatrix<{{dtype}}>"]),
    )]
    pub fn matrix_csr_indices_size<T: RlstScalar>(matrix: &CsrMatrix<T>) -> usize {
        matrix.indices().len()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["CsrMatrix<{{dtype}}>"]),
    )]
    pub fn matrix_csr_indices<T: RlstScalar>(matrix: &CsrMatrix<T>) -> *const usize {
        matrix.indices().as_ptr()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["CsrMatrix<{{dtype}}>"]),
    )]
    pub unsafe fn matrix_csr_indices_copy<T: RlstScalar>(
        matrix: &CsrMatrix<T>,
        indices: *mut usize,
    ) {
        for (i, j) in matrix.indices().iter().enumerate() {
            *indices.add(i) = *j;
        }
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["CsrMatrix<{{dtype}}>"]),
    )]
    pub fn matrix_csr_indptr_size<T: RlstScalar>(matrix: &CsrMatrix<T>) -> usize {
        matrix.indptr().len()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["CsrMatrix<{{dtype}}>"]),
    )]
    pub fn matrix_csr_indptr<T: RlstScalar>(matrix: &CsrMatrix<T>) -> *const usize {
        matrix.indptr().as_ptr()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "matrix", wrapper = "MatrixT", replace_with = ["CsrMatrix<{{dtype}}>"]),
    )]
    pub unsafe fn matrix_csr_indptr_copy<T: RlstScalar>(matrix: &CsrMatrix<T>, indptr: *mut usize) {
        for (i, j) in matrix.indptr().iter().enumerate() {
            *indptr.add(i) = *j;
        }
    }
}
