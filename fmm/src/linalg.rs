// use nalgebra as na;
use ndarray::*;
use ndarray_linalg::*;

const F64_EPSILON: f64 = 2.2204460492503131E-16f64;

// Moore-Penrose pseudoinverse
pub fn pinv<T: Scalar + Lapack>(
    array: &Array2<T>,
) -> (
    ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<<T as Scalar>::Real>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>,
) {
    let (u, mut s, vt): (_, Array1<_>, _) = array.svd(true, true).unwrap();

    let u = u.unwrap();
    let vt = vt.unwrap();

    let max_s = s[0];

    println!("N SING VALS {:?}", s.len());

    // Hacky, should really work with type check at runtime.
    for s in s.iter_mut() {
        if *s > T::real(4.) * max_s * T::real(F64_EPSILON) {
            *s = T::real(1.) / *s;
        } else {
            *s = T::real(0);
        }
    }

    let v = vt.t();
    let ut = u.t();

    let s_inv_mat = Array2::from_diag(&s);

    // Return components
    (v.to_owned(), s_inv_mat.to_owned(), ut.to_owned())
}

mod test {

    use super::*;

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
        let nvals = (dim as usize).pow(2);
        for _ in 0..nvals {
            data.push(between.sample(&mut range))
        }

        let data = Array1::from_vec(data).into_shape((dim, dim)).unwrap();
        let data2 = data.clone();

        let (a, b, c) = pinv(&data);

        // Test dimensions of computed inverse are correct
        let inv = a.dot(&b).dot(&c);
        assert_eq!(inv.ncols(), dim);
        assert_eq!(inv.nrows(), dim);

        // Test that the inverse is approximately correct
        let res = inv.dot(&data2);

        let ones = Array1::from_vec(vec![1.; dim]);
        let id = Array2::from_diag(&ones);

        for (a, b) in id.iter().zip(res.iter()) {
            assert_approx_eq!(f64, *a, *b, epsilon = 1e-14);
        }
    }
}
