//! Finite elements

pub mod ciarlet;
pub mod polynomials;
pub mod reference_cell;

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
