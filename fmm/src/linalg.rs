use std::ops::Mul;

use nalgebra as na;

// Moore-Penrose pseudoinverse
pub fn pinv(
    matrix: na::DMatrix<f64>,
) -> (
    na::Matrix<f64, na::Dyn, na::Dyn, na::VecStorage<f64, na::Dyn, na::Dyn>>,
    na::Matrix<f64, na::Dyn, na::Dyn, na::VecStorage<f64, na::Dyn, na::Dyn>>,
    na::Matrix<f64, na::Dyn, na::Dyn, na::VecStorage<f64, na::Dyn, na::Dyn>>,
) {
    let svd = na::linalg::SVD::new(matrix, true, true);

    let max_s = svd.singular_values.max();
    // svd.singular_values;
    let mut s_inv = na::Matrix::from(svd.singular_values.clone_owned());

    for (i, &s) in svd.singular_values.iter().enumerate() {
        // Heuristic
        if s > 4. * max_s * f64::EPSILON {
            s_inv[i] = 1. / s;
        } else {
            s_inv[i] = 0.;
        }
    }

    let v = svd.v_t.unwrap().transpose();
    let ut = svd.u.unwrap().transpose();

    let mut s_inv_mat = na::DMatrix::<f64>::zeros(s_inv.len(), s_inv.len());

    s_inv_mat.set_diagonal(&s_inv);
    return (v, s_inv_mat, ut);
}

mod test {

    use super::*;

    use float_cmp::approx_eq;
    use float_cmp::assert_approx_eq;
    use rand::prelude::*;
    use rand::SeedableRng;

    #[test]
    fn test_pinv() {
        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0..1.0);

        // Setup a random square matrix, of dimension 'dim'
        let mut data: Vec<f64> = Vec::new();
        let dim = 5;
        let npoints = (dim as usize).pow(2);
        for _ in 0..npoints {
            data.push(between.sample(&mut range))
        }

        let data = na::DMatrix::from_vec(dim, dim, data);
        let data2 = data.clone();
        let (a, b, c) = pinv(data);

        // Test dimensions of computed inverse are correct
        let inv = a.mul(b).mul(c);
        assert_eq!(inv.ncols(), dim);
        assert_eq!(inv.nrows(), dim);

        // Test that the inverse is approximately correct
        let res = inv.mul(data2);

        let id = na::DMatrix::<f64>::identity(dim, dim);
        for (a, b) in res.row_iter().zip(id.row_iter()) {
            for (c, d) in a.column_iter().zip(b.column_iter()) {
                let c = c[0];
                let d = d[0];
                assert_approx_eq!(f64, c, d, epsilon = 1e-14);
            }
        }
    }
}
