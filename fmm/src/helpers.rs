use rlst_dense::types::RlstScalar;

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

/// Scaling to apply to homogenous scale invariant kernels at a given octree level.
pub fn homogenous_kernel_scale<T: RlstScalar<Real = T>>(level: u64) -> T {
    let numerator = T::from(1).unwrap();
    let denominator = T::from(2.).unwrap();
    let power = T::from(level).unwrap();
    let denominator = <T as RlstScalar>::powf(denominator, power);
    numerator / denominator
}

/// Scaling to apply to M2L operators calculated using homogenous scale invariant kernels at a given octree level.
pub fn m2l_scale<T: RlstScalar<Real = T>>(level: u64) -> T {
    if level < 2 {
        panic!("M2L only perfomed on level 2 and below")
    }

    if level == 2 {
        T::from(1. / 2.).unwrap()
    } else {
        let two = T::from(2.0).unwrap();
        <T as RlstScalar>::powf(two, T::from(level - 3).unwrap())
    }
}
