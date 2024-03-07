pub const ALPHA_INNER: f64 = 1.05;

pub const ALPHA_OUTER: f64 = 2.95;

/// Number of siblings for each node in octree
pub const NSIBLINGS: usize = 8;

pub const NSIBLINGS_SQUARED: usize = 64;

pub const NCORNERS: usize = 8;

pub const NTRANSFER_VECTORS_KIFMM: usize = 316;

/// Maximum number of boxes in a 1 box deep halo around a given box in 3D
pub const NHALO: usize = 26;