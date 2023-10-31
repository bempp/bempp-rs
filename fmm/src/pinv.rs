//! Implementation of Moore-Penrose PseudoInverse
use num::{Float, Zero};
use rlst::algorithms::linalg::LinAlg;
use rlst::algorithms::traits::svd::{Mode, Svd};
use rlst::dense::{
    base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix, Dynamic, Shape,
};
// use rlst_common::traits::*;
use rlst::common::traits::{Eval, Transpose};
use rlst::common::types::{RlstError, RlstResult, Scalar};
use rlst::dense::MatrixD;

pub type PinvMatrix = Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic>, Dynamic>;

type PinvReturnType<T> = RlstResult<(
    Vec<<T as Scalar>::Real>,
    MatrixD<<T as Scalar>::Real>,
    MatrixD<<T as Scalar>::Real>,
)>;

/// Compute the (Moore-Penrose) pseudo-inverse of a matrix.
///
/// Calculate a generalised inverse using its singular value decomposition `U @ S @ V*`.
/// If `s` is the maximum singular value, then the signifance cut-off value is determined by
/// `atol + rtol * s`. Any singular value below this is assumed insignificant.
///
/// # Arguments
/// * `mat` - (M, N) matrix to be inverted.
/// * `atol` - Absolute threshold term, default is 0.
/// * `rtol` - Relative threshold term, default value is max(M, N) * eps
pub fn pinv<T: Scalar<Real = f64> + Float>(
    mat: &PinvMatrix,
    atol: Option<T::Real>,
    rtol: Option<T::Real>,
) -> PinvReturnType<T> {
    let shape = mat.shape();

    if shape.0 == 0 || shape.1 == 0 {
        return Err(RlstError::MatrixIsEmpty(shape));
    }

    // If we have a vector return error
    if shape.0 == 1 || shape.1 == 1 {
        Err(RlstError::SingleDimensionError {
            expected: 2,
            actual: 1,
        })
    } else {
        // For matrices compute the full SVD
        let (mut s, u, vt) = mat.linalg().svd(Mode::All, Mode::All)?;
        let u = u.unwrap();
        let vt = vt.unwrap();

        let eps = T::real(T::epsilon());
        let max_dim = T::real(std::cmp::max(shape.0, shape.1));

        let atol = atol.unwrap_or(T::Real::zero());
        let rtol = rtol.unwrap_or(max_dim * eps);

        let max_s = T::real(s[0]);
        let threshold = atol + rtol * max_s;
        // Filter singular values below this threshold
        for s in s.iter_mut() {
            if *s > threshold {
                *s = T::real(1.0) / *s;
            } else {
                *s = T::real(0.)
            }
        }

        // Return pseudo-inverse in component form
        let v = vt.transpose().eval();
        let ut = u.transpose().eval();

        Ok((s, ut, v))
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use approx::assert_relative_eq;
    use rlst::common::traits::ColumnMajorIterator;
    use rlst::common::traits::NewLikeSelf;
    use rlst::dense::{rlst_dynamic_mat, rlst_rand_mat, Dot};

    #[test]
    fn test_pinv() {
        let dim: usize = 5;
        let mat = rlst_rand_mat![f64, (dim, dim)];

        let (s, ut, v) = pinv::<f64>(&mat, None, None).unwrap();

        let ut = ut;
        let v = v;

        let mut mat_s = rlst_dynamic_mat![f64, (s.len(), s.len())];
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i];
        }

        let inv = v.dot(&mat_s).dot(&ut);

        let actual = inv.dot(&mat);

        // Expect the identity matrix
        let mut expected = actual.new_like_self();
        for i in 0..dim {
            expected[[i, i]] = 1.0
        }

        for (a, e) in actual.iter_col_major().zip(expected.iter_col_major()) {
            assert_relative_eq!(a, e, epsilon = 1E-13);
        }
    }
}
