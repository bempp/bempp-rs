//! Data structures and methods for Morton Keys.
pub type KeyType = u64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
/// Representation of a Morton key with an 'anchor' specifying the origin of the node it encodes
/// with respect to the deepest level of the octree, as well as 'morton', a bit-interleaved single
/// integer representation.
pub struct MortonKey {
    // Lower left corner of key
    pub anchor: [KeyType; 3],
    // Morton encoded represententation
    pub morton: KeyType,
}

/// Vector of **MortonKeys**.
#[derive(Clone, Debug, Default)]
pub struct MortonKeys {
    pub keys: Vec<MortonKey>,
    pub index: usize,
}
