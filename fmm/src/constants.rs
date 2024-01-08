//! Crate wide constants

/// Maximum chunk size to use to process leaf boxes during P2M kernel.
pub const P2M_MAX_CHUNK_SIZE: usize = 256;

/// Maximum chunk size to use to process boxes by level during M2M kernel.
pub const M2M_MAX_CHUNK_SIZE: usize = 256;

/// Maximum chunk size to use to process boxes by level during L2L kernel.
pub const L2L_MAX_CHUNK_SIZE: usize = 256;
