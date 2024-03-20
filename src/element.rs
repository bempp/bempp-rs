//! Finite elements

pub mod element;
pub mod polynomials;
pub mod reference_cell;
//pub mod map;

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
