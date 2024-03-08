//! Crate wide constants

/// Maximum chunk size to use to process leaf boxes during P2M kernel.
pub const P2M_MAX_CHUNK_SIZE: usize = 256;

/// Maximum chunk size to use to process boxes by level during M2M kernel.
pub const M2M_MAX_CHUNK_SIZE: usize = 256;

/// Maximum chunk size to use to process boxes by level during L2L kernel.
pub const L2L_MAX_CHUNK_SIZE: usize = 256;

/// Amount to dilate box radius by in octree for inner surfaces
pub const ALPHA_INNER: f64 = 1.05;

/// Amount to dilate box radius by in octree for outer surfaces
pub const ALPHA_OUTER: f64 = 2.95;

/// Number of siblings for each node in octree
pub const NSIBLINGS: usize = 8;

/// Maximum number of boxes in a 1 box deep halo around a given box in 3D
pub const NHALO: usize = 26;
