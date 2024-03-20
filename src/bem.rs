//! Function spaces and BEM assembly

pub mod assembly;
pub mod function_space;

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
