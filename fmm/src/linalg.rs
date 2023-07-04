// //! Temporary home of linear algebra utilities. TODO: Replace with routines from Householder.
// use ndarray::*;
// // use ndarray_linalg::*;

// use bempp_traits::types::Scalar;

// use rlst;
// use rlst::dense::{base_matrix::BaseMatrix, VectorContainer};


// // const F64_EPSILON: f64 = 2.220_446_049_250_313E-16f64;
// // type D = Dim<[usize; 2]>;
// // type Type1<T> = ArrayBase<OwnedRepr<T>, D>;
// // type Type2<T> = ArrayBase<OwnedRepr<<T as Scalar>::Real>, D>;

// // /// Calculate the Moore-Penrose pseudoinverse.
// // pub fn pinv<T: Scalar + Lapack>(array: &Array2<T>) -> (Type1<T>, Type2<T>, Type1<T>) {
// //     let (u, mut s, vt): (_, Array1<_>, _) = array.svd(true, true).unwrap();

// //     let u = u.unwrap();
// //     // Truncate u
// //     let vt = vt.unwrap();

// //     let max_s = s[0];

// //     // Hacky, should really work with type check at runtime.
// //     for s in s.iter_mut() {
// //         if *s > T::real(4.) * max_s * T::real(F64_EPSILON) {
// //             *s = T::real(1.) / *s;
// //         } else {
// //             *s = T::real(0);
// //         }
// //     }

// //     let v = vt.t();
// //     let ut = u.t();

// //     let s_inv_mat = Array2::from_diag(&s);

// //     // Return components
// //     (v.to_owned(), s_inv_mat.to_owned(), ut.to_owned())
// // }

// type Type1 =  Matrix<<T as Scalar>::Real, BaseMatrix<<T as Scalar>::Real, VectorContainer<<T as Scalar>::Real>, Dynamic, Dynamic>, Dynamic, Dynamic>;

// pub fn pinv(array: Type1) {

//     let (s, u, vt) = array.linalg.svd().unwrap(Mode::All, Mode::All);

//     let max_s = s[0];

//     for s in s.iter_mut() {
        
//     }

// }

// // pub fn matrix_rank<T: Scalar + Lapack>(array: &Array2<T>) -> usize {
// //     let (_, s, _): (_, Array1<_>, _) = array.svd(false, false).unwrap();
// //     let shape = array.shape();
// //     let max_dim = shape.iter().max().unwrap();

// //     let tol = s[0] * T::real(*max_dim as f64) * T::real(F64_EPSILON);

// //     let significant: Vec<bool> = s.iter().map(|sv| sv > &tol).filter(|sv| *sv).collect();
// //     let rank = significant.iter().len();

// //     rank
// // }

// #[allow(unused_imports)]
// mod test {

//     use super::*;

//     use rlst;
//     use rlst::common::tools::PrettyPrint;
//     use rlst::dense::rlst_rand_mat;


//     #[test]
//     fn test_pinv() {
//         // let mut range = StdRng::seed_from_u64(0);
//         // let between = rand::distributions::Uniform::from(0.0..1.0);

//         // // Setup a random square matrix, of dimension 'dim'
//         // let mut data: Vec<f64> = Vec::new();
//         // let dim: usize = 5;
//         // let nvals = dim.pow(2);
//         // for _ in 0..nvals {
//         //     data.push(between.sample(&mut range))
//         // }

//         // let data = Array1::from_vec(data).into_shape((dim, dim)).unwrap();

//         let dim = 5;
//         let data = rlst_rand_mat![f64, (dim, dim)];

//         pinv(&data);
//         // let (a, b, c) = pinv(&data);

//         // // Test dimensions of computed inverse are correct
//         // let inv = a.dot(&b).dot(&c);
//         // assert_eq!(inv.ncols(), dim);
//         // assert_eq!(inv.nrows(), dim);

//         // // Test that the inverse is approximately correct
//         // let res = inv.dot(&data);

//         // let ones = Array1::from_vec(vec![1.; dim]);
//         // let id = Array2::from_diag(&ones);

//         // for (a, b) in id.iter().zip(res.iter()) {
//         //     assert_approx_eq!(f64, *a, *b, epsilon = 1e-14);
//         // }
//     }
// }
