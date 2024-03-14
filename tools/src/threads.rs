//! Thread tools

use rayon::ThreadPoolBuilder;

/// Create a pool
pub fn create_pool(num_threads: usize) -> rayon::ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap()
}
