//! Functionality for handling 3D arrays
//! # Warning
//! This module awaits deprecation with the addition of N-Dimensional tensor handling in rlst.
use num::traits::Num;

use rlst::{
    rlst_dynamic_array3, Array, BaseArray, RandomAccessByRef, RandomAccessMut, RlstScalar, Shape,
    VectorContainer,
};

/// Pad an Array3D from a given `pad_index` with an amount of zeros specified by `pad_size` to the right of each axis.
///
/// # Arguments
/// * `arr` - An array to be padded.
/// * `pad_size` - The amount of padding to be added along each axis.
/// * `pad_index` - The position in the array to start the padding from.
pub fn pad3<T: RlstScalar>(
    arr: &Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>,
    pad_size: (usize, usize, usize),
    pad_index: (usize, usize, usize),
) -> Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>
where
    T: Clone + Copy + Num,
{
    let [m, n, o] = arr.shape();

    let (x, y, z) = pad_index;
    let (p, q, r) = pad_size;

    // Check that there is enough space for pad
    assert!(x + p <= m + p && y + q <= n + q && z + r <= o + r);

    let mut padded = rlst_dynamic_array3!(T, [p + m, q + n, r + o]);

    for i in 0..m {
        for j in 0..n {
            for k in 0..o {
                *padded.get_mut([x + i, y + j, z + k]).unwrap() = *arr.get([i, j, k]).unwrap();
            }
        }
    }

    padded
}

/// Flip an Array3D along each axis, returns a new array.
///
/// # Arguments
/// * `arr` - An array to be flipped.
pub fn flip3<T: RlstScalar>(
    arr: &Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>,
) -> Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>
where
    T: Clone + Copy + Num,
{
    let mut flipped = rlst_dynamic_array3!(T, arr.shape());

    let [m, n, o] = arr.shape();

    for i in 0..m {
        for j in 0..n {
            for k in 0..o {
                *flipped.get_mut([i, j, k]).unwrap() =
                    *arr.get([m - i - 1, n - j - 1, o - k - 1]).unwrap();
            }
        }
    }

    flipped
}

#[cfg(test)]
mod test {

    use super::*;
    use approx::*;

    #[test]
    fn test_flip3() {
        let n = 2;
        let mut arr = rlst_dynamic_array3!(f64, [n, n, n]);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    *arr.get_mut([i, j, k]).unwrap() = (i + j * n + k * n * n) as f64;
                }
            }
        }
        let result = flip3(&arr);
        assert_relative_eq!(*result.get([0, 0, 0]).unwrap(), 7.0);
        assert_relative_eq!(*result.get([0, 0, 1]).unwrap(), 3.0);
        assert_relative_eq!(*result.get([0, 1, 0]).unwrap(), 5.0);
        assert_relative_eq!(*result.get([0, 1, 1]).unwrap(), 1.0);
        assert_relative_eq!(*result.get([1, 0, 0]).unwrap(), 6.0);
        assert_relative_eq!(*result.get([1, 0, 1]).unwrap(), 2.0);
        assert_relative_eq!(*result.get([1, 1, 0]).unwrap(), 4.0);
        assert_relative_eq!(*result.get([1, 1, 1]).unwrap(), 0.0);
    }

    #[test]
    fn test_pad3() {
        let dim = 3;
        // Initialise input data
        let mut input = rlst_dynamic_array3!(f64, [dim, dim, dim]);
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    *input.get_mut([i, j, k]).unwrap() = (i + j * dim + k * dim * dim + 1) as f64
                }
            }
        }

        // Test padding at edge of each axis
        let pad_size = (2, 3, 4);
        let pad_index = (0, 0, 0);
        let padded = pad3(&input, pad_size, pad_index);

        let [m, n, o] = padded.shape();

        // Check dimension
        assert_eq!(m, dim + pad_size.0);
        assert_eq!(n, dim + pad_size.1);
        assert_eq!(o, dim + pad_size.2);

        // Check that padding has been correctly applied
        for i in dim..m {
            for j in dim..n {
                for k in dim..o {
                    assert_eq!(*padded.get([i, j, k]).unwrap(), 0f64)
                }
            }
        }

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    assert_eq!(
                        *padded.get([i, j, k]).unwrap(),
                        *input.get([i, j, k]).unwrap()
                    )
                }
            }
        }

        // Test padding at the start of each axis
        let pad_index = (2, 2, 2);

        let padded = pad3(&input, pad_size, pad_index);

        // Check that padding has been correctly applied
        for i in 0..pad_index.0 {
            for j in 0..pad_index.1 {
                for k in 0..pad_index.2 {
                    assert_eq!(*padded.get([i, j, k]).unwrap(), 0f64)
                }
            }
        }

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    assert_eq!(
                        *padded
                            .get([i + pad_index.0, j + pad_index.1, k + pad_index.2])
                            .unwrap(),
                        *input.get([i, j, k]).unwrap()
                    );
                }
            }
        }
    }
}
