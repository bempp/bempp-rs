//! Data structures for Morton Keys.
/// Key type
pub type KeyType = u64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
/// Representation of a Morton key with an 'anchor' specifying the origin of the node it encodes
/// with respect to the deepest level of the octree, as well as 'morton', a bit-interleaved single
/// integer representation.
pub struct MortonKey {
    /// The anchor is the index coordinate of the key, with respect to the origin of the Domain.
    pub anchor: [KeyType; 3],
    /// The Morton encoded anchor.
    pub morton: KeyType,
}

/// Container of **MortonKeys**.
#[derive(Clone, Debug, Default)]
pub struct MortonKeys {
    /// A vector of MortonKeys
    pub keys: Vec<MortonKey>,

    /// index for implementing the Iterator trait.
    pub index: usize,
}
