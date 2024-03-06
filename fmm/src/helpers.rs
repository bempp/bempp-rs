use crate::traits::FmmScalar;
use cauchy::Scalar;

/// Euclidean algorithm to find greatest divisor of `n` less than or equal to `max_chunk_size`
pub fn find_chunk_size(n: usize, max_chunk_size: usize) -> usize {
    let max_divisor = max_chunk_size;
    for divisor in (1..=max_divisor).rev() {
        if n % divisor == 0 {
            return divisor;
        }
    }
    1 // If no divisor is found greater than 1, return 1 as the GCD
}

pub fn homogenous_kernel_scale<T: FmmScalar<Real = T>>(level: u64) -> T {
    let numerator = T::from(1).unwrap();
    let denominator = T::from(2.).unwrap();
    let power = T::from(level).unwrap();
    let denominator = <T as Scalar>::powf(denominator, power);
    numerator / denominator
}
