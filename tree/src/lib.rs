//! # Distributed Octrees in Rust
//!
//! Octrees parallelized using Rust and MPI for Scientific Computing.
//!
//! ## References
//! \[1\] Sundar, Hari, Rahul S. Sampath, and George Biros. "Bottom-up construction and 2: 1
//! balance refinement of linear octrees in parallel." SIAM Journal on Scientific Computing 30.5
//! (2008): 2675-2708.
//!
//! \[2\] Sundar, Hari, Dhairya Malhotra, and George Biros. "Hyksort: a new variant of
//! hypercube quicksort on distributed memory architectures." Proceedings of the 27th
//! international ACM conference on international conference on supercomputing. (2013).
//!
//! \[3\] Lashuk, Ilya, et al. "A massively parallel adaptive fast-multipole method on heterogeneous
//! architectures." Proceedings of the Conference on High Performance Computing Networking, Storage
//! and Analysis. IEEE (2009).
//!
//! \[4\] Chan, T. "Closest-point problems simplified on the RAM", ACM-SIAM Symposium on Discrete
//! Algorithms (2002)
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod constants;
pub mod implementations;
pub mod types;
