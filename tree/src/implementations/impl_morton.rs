use itertools::{izip, Itertools};
use std::{
    cmp::Ordering,
    collections::HashSet,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
};

use crate::{
    constants::{
        BYTE_DISPLACEMENT, BYTE_MASK, DEEPEST_LEVEL, DIRECTIONS, LEVEL_DISPLACEMENT, LEVEL_MASK,
        LEVEL_SIZE, NINE_BIT_MASK, X_LOOKUP_DECODE, X_LOOKUP_ENCODE, Y_LOOKUP_DECODE,
        Y_LOOKUP_ENCODE, Z_LOOKUP_DECODE, Z_LOOKUP_ENCODE,
    },
    types::{
        domain::Domain,
        morton::{KeyType, MortonKey, MortonKeys},
        point::PointType,
    },
};

/// Linearize (remove overlaps) a vector of keys. The input must be sorted. Algorithm 7 in [1].
pub fn linearize_keys(keys: &Vec<MortonKey>) -> Vec<MortonKey> {
    let nkeys = keys.len();

    // Then we remove the ancestors.
    let mut new_keys = Vec::<MortonKey>::with_capacity(keys.len());

    // Now check pairwise for ancestor relationship and only add to new vector if item
    // is not an ancestor of the next item. Add final element.
    keys.iter()
        .enumerate()
        .tuple_windows::<((_, _), (_, _))>()
        .for_each(|((_, a), (j, b))| {
            if !a.is_ancestor(b) {
                new_keys.push(*a);
            }
            if j == (nkeys - 1) {
                new_keys.push(*b);
            }
        });

    new_keys
}

/// Complete the region between two keys with the minimum spanning nodes, algorithm 6 in [1].
pub fn complete_region(a: &MortonKey, b: &MortonKey) -> Vec<MortonKey> {
    let mut a_ancestors: HashSet<MortonKey> = a.ancestors();
    let mut b_ancestors: HashSet<MortonKey> = b.ancestors();

    a_ancestors.remove(a);
    b_ancestors.remove(b);

    let mut minimal_tree: Vec<MortonKey> = Vec::new();
    let mut work_list: Vec<MortonKey> = a.finest_ancestor(b).children().into_iter().collect();

    while !work_list.is_empty() {
        let current_item = work_list.pop().unwrap();
        if (current_item > *a) & (current_item < *b) & !b_ancestors.contains(&current_item) {
            minimal_tree.push(current_item);
        } else if (a_ancestors.contains(&current_item)) | (b_ancestors.contains(&current_item)) {
            let mut children = current_item.children();
            work_list.append(&mut children);
        }
    }

    minimal_tree.sort();
    minimal_tree
}

impl MortonKeys {
    /// Complete the region between all elements in an vector of Morton keys that doesn't
    /// necessarily span the domain defined by its least and greatest nodes.
    pub fn complete(&mut self) {
        let a = self.keys.iter().min().unwrap();
        let b = self.keys.iter().max().unwrap();
        let mut completion = complete_region(a, b);
        completion.push(*a);
        completion.push(*b);
        completion.sort();
        self.keys = completion;
    }

    /// Wrapper for linearize_keys over all keys in vector of Morton keys.
    pub fn linearize(&mut self) {
        self.keys.sort();
        self.keys = linearize_keys(&self.keys);
    }

    /// Wrapper for sorting a tree, by its keys.
    pub fn sort(&mut self) {
        self.keys.sort();
    }

    /// Enforce a 2:1 balance for a vector of Morton keys, and remove any overlaps.
    pub fn balance(&mut self) {
        let mut balanced: HashSet<MortonKey> = self.keys.iter().cloned().collect();

        for level in (0..DEEPEST_LEVEL).rev() {
            let work_list: Vec<MortonKey> = balanced
                .iter()
                .filter(|key| key.level() == level)
                .cloned()
                .collect();

            for key in work_list {
                let neighbors = key.neighbors();

                for neighbor in neighbors {
                    let parent = neighbor.parent();
                    if !balanced.contains(&neighbor) && !balanced.contains(&neighbor) {
                        balanced.insert(parent);

                        if parent.level() > 0 {
                            for sibling in parent.siblings() {
                                balanced.insert(sibling);
                            }
                        }
                    }
                }
            }
        }

        let mut balanced = MortonKeys {
            keys: balanced.into_iter().collect(),
        };
        balanced.sort();
        balanced.linearize();
        self.keys = balanced.keys;
    }
}

impl Deref for MortonKeys {
    type Target = Vec<MortonKey>;

    fn deref(&self) -> &Self::Target {
        &self.keys
    }
}

impl DerefMut for MortonKeys {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.keys
    }
}

/// Serialize a Morton Key for VTK visualization.
pub fn serialize_morton_key(key: &MortonKey, domain: &Domain) -> Vec<f64> {
    let anchor = key.anchor;

    let mut serialized = Vec::<PointType>::with_capacity(24);

    let disp = 1 << (LEVEL_DISPLACEMENT + 1 - key.level() as usize);

    let anchors = [
        [anchor[0], anchor[1], anchor[2]],
        [disp + anchor[0], anchor[1], anchor[2]],
        [anchor[0], disp + anchor[1], anchor[2]],
        [disp + anchor[0], disp + anchor[1], anchor[2]],
        [anchor[0], anchor[1], disp + anchor[2]],
        [disp + anchor[0], anchor[1], disp + anchor[2]],
        [anchor[0], disp + anchor[1], disp + anchor[2]],
        [disp + anchor[0], disp + anchor[1], disp + anchor[2]],
    ];

    for anchor in anchors.iter() {
        let coords = MortonKey::from_anchor(anchor).to_coordinates(domain);
        for index in 0..3 {
            serialized.push(coords[index]);
        }
    }

    serialized
}

/// Return the level associated with a key.
fn find_level(morton: KeyType) -> KeyType {
    morton & LEVEL_MASK
}

/// Helper function for decoding keys.
fn decode_key_helper(key: KeyType, lookup_table: &[KeyType; 512]) -> KeyType {
    const N_LOOPS: KeyType = 7; // 8 bytes in 64 bit key
    let mut coord: KeyType = 0;

    for index in 0..N_LOOPS {
        coord |= lookup_table[((key >> (index * 9)) & NINE_BIT_MASK) as usize] << (3 * index);
    }

    coord
}

/// Decode a given key.
///
/// Returns the anchor for the given Morton key
fn decode_key(morton: KeyType) -> [KeyType; 3] {
    let key = morton >> LEVEL_DISPLACEMENT;

    let x = decode_key_helper(key, &X_LOOKUP_DECODE);
    let y = decode_key_helper(key, &Y_LOOKUP_DECODE);
    let z = decode_key_helper(key, &Z_LOOKUP_DECODE);

    [x, y, z]
}

/// Map a point to the anchor of the enclosing box.
///
/// Returns the 3 integeger coordinates of the enclosing box.
///
/// # Arguments
/// `point` - The (x, y, z) coordinates of the point to map.
/// `level` - The level of the tree at which the point will be mapped.
/// `origin` - The origin of the bounding box.
/// `diameter` - The diameter of the bounding box in each dimension.
fn point_to_anchor(
    point: &[PointType; 3],
    level: KeyType,
    origin: &[PointType; 3],
    diameter: &[PointType; 3],
) -> [KeyType; 3] {
    let mut anchor: [KeyType; 3] = [0, 0, 0];

    let level_size = (1 << level) as PointType;

    for (anchor_value, point_value, &origin_value, &diameter_value) in
        izip!(&mut anchor, point, origin, diameter)
    {
        *anchor_value =
            ((point_value - origin_value) * level_size / diameter_value).floor() as KeyType
    }

    anchor
}

/// Encode an anchor.
///
/// Returns the Morton key associated with the given anchor.
///
/// # Arguments
/// `anchor` - A vector with 4 elements defining the integer coordinates and level.
fn encode_anchor(anchor: &[KeyType; 3], level: KeyType) -> KeyType {
    let x = anchor[0];
    let y = anchor[1];
    let z = anchor[2];

    let key: KeyType = X_LOOKUP_ENCODE[((x >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
        | Y_LOOKUP_ENCODE[((y >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
        | Z_LOOKUP_ENCODE[((z >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];

    let key = (key << 24)
        | X_LOOKUP_ENCODE[(x & BYTE_MASK) as usize]
        | Y_LOOKUP_ENCODE[(y & BYTE_MASK) as usize]
        | Z_LOOKUP_ENCODE[(z & BYTE_MASK) as usize];

    let key = key << LEVEL_DISPLACEMENT;
    key | level
}

impl MortonKey {
    /// Return the anchor
    pub fn anchor(&self) -> &[KeyType; 3] {
        &self.anchor
    }

    /// Return the Morton representation
    pub fn morton(&self) -> KeyType {
        self.morton
    }

    /// Return the level
    pub fn level(&self) -> KeyType {
        find_level(self.morton)
    }

    /// Return a `MortonKey` type from a Morton index
    pub fn from_morton(morton: KeyType) -> Self {
        let anchor = decode_key(morton);

        MortonKey { anchor, morton }
    }

    /// Return a `MortonKey` type from the anchor on the deepest level
    pub fn from_anchor(anchor: &[KeyType; 3]) -> Self {
        let morton = encode_anchor(anchor, DEEPEST_LEVEL);

        MortonKey {
            anchor: anchor.to_owned(),
            morton,
        }
    }

    /// Return a `MortonKey` associated with the box that encloses the point on the deepest level
    pub fn from_point(point: &[PointType; 3], domain: &Domain) -> Self {
        let anchor = point_to_anchor(point, DEEPEST_LEVEL, &domain.origin, &domain.diameter);
        MortonKey::from_anchor(&anchor)
    }

    /// Return the parent
    pub fn parent(&self) -> Self {
        let level = self.level();
        let morton = self.morton >> LEVEL_DISPLACEMENT;

        let parent_level = level - 1;
        let bit_multiplier = DEEPEST_LEVEL - parent_level;

        // Zeros out the last 3 * bit_multiplier bits of the Morton index
        let parent_morton_without_level = (morton >> (3 * bit_multiplier)) << (3 * bit_multiplier);

        let parent_morton = (parent_morton_without_level << LEVEL_DISPLACEMENT) | parent_level;

        MortonKey::from_morton(parent_morton)
    }

    /// Return the first child
    pub fn first_child(&self) -> Self {
        MortonKey {
            anchor: self.anchor,
            morton: 1 + self.morton,
        }
    }

    /// Return the first child on the deepest level
    pub fn finest_first_child(&self) -> Self {
        MortonKey {
            anchor: self.anchor,
            morton: DEEPEST_LEVEL - self.level() + self.morton,
        }
    }

    /// Return the last child on the deepest level
    pub fn finest_last_child(&self) -> Self {
        if self.level() < DEEPEST_LEVEL {
            let mut level_diff = DEEPEST_LEVEL - self.level();
            let mut flc = *self.children().iter().max().unwrap();

            while level_diff > 1 {
                let tmp = flc;
                flc = *tmp.children().iter().max().unwrap();
                level_diff -= 1;
            }

            flc
        } else {
            *self
        }
    }

    /// Return all children in order of their Morton indices
    pub fn children(&self) -> Vec<MortonKey> {
        let level = self.level();
        let morton = self.morton() >> LEVEL_DISPLACEMENT;

        let mut children_morton: [KeyType; 8] = [0; 8];
        let mut children: Vec<MortonKey> = Vec::with_capacity(8);
        let bit_shift = 3 * (DEEPEST_LEVEL - level - 1);
        for (index, item) in children_morton.iter_mut().enumerate() {
            *item =
                ((morton | (index << bit_shift) as KeyType) << LEVEL_DISPLACEMENT) | (level + 1);
        }

        for &child_morton in children_morton.iter() {
            children.push(MortonKey::from_morton(child_morton))
        }

        children
    }

    /// Return all children of the parent of the current Morton index
    pub fn siblings(&self) -> Vec<MortonKey> {
        self.parent().children()
    }

    /// Check if the key is ancestor of `other`.
    pub fn is_ancestor(&self, other: &MortonKey) -> bool {
        let ancestors = other.ancestors();
        ancestors.contains(self)
    }

    /// Check if key is descendent of another key
    pub fn is_descendent(&self, other: &MortonKey) -> bool {
        other.is_ancestor(self)
    }

    /// Return set of all ancestors
    pub fn ancestors(&self) -> HashSet<MortonKey> {
        let mut ancestors = HashSet::<MortonKey>::new();

        let mut current = *self;

        ancestors.insert(current);

        while current.level() > 0 {
            current = current.parent();
            ancestors.insert(current);
        }

        ancestors
    }

    /// Find the finest ancestor of key and another key
    pub fn finest_ancestor(&self, other: &MortonKey) -> MortonKey {
        if self == other {
            *other
        } else {
            let my_ancestors = self.ancestors();
            let mut current = other.parent();
            while !my_ancestors.contains(&current) {
                current = current.parent()
            }
            current
        }
    }

    /// Return a point with the coordinates of the anchor
    pub fn to_coordinates(&self, domain: &Domain) -> [PointType; 3] {
        let mut coord: [PointType; 3] = [0.0; 3];

        for (anchor_value, coord_ref, origin_value, diameter_value) in
            izip!(self.anchor, &mut coord, &domain.origin, &domain.diameter)
        {
            *coord_ref = origin_value
                + diameter_value * (anchor_value as PointType) / (LEVEL_SIZE as PointType);
        }

        coord
    }

    /// Serialized representation of a box associated with a key.
    ///
    /// Returns a vector with 24 entries, associated with the 8 x,y,z coordinates
    /// of the box associated with the key.
    /// If the lower left corner of the box is (0, 0, 0). Then the points are numbered in the
    /// following order.
    /// 1. (0, 0, 0)
    /// 2. (1, 0, 0)
    /// 3. (0, 1, 0)
    /// 4. (1, 1, 0)
    /// 5. (0, 0, 1)
    /// 6. (1, 0, 1)
    /// 7. (0, 1, 1)
    /// 8. (1, 1, 1)
    ///
    /// # Arguments
    /// * `domain` - The domain descriptor.
    pub fn box_coordinates(&self, domain: &Domain) -> Vec<f64> {
        let mut serialized = Vec::<f64>::with_capacity(24);
        let level = self.level();
        let step = (1 << (DEEPEST_LEVEL - level)) as u64;

        let anchors = [
            [self.anchor[0], self.anchor[1], self.anchor[2]],
            [step + self.anchor[0], self.anchor[1], self.anchor[2]],
            [self.anchor[0], step + self.anchor[1], self.anchor[2]],
            [step + self.anchor[0], step + self.anchor[1], self.anchor[2]],
            [self.anchor[0], self.anchor[1], step + self.anchor[2]],
            [step + self.anchor[0], self.anchor[1], step + self.anchor[2]],
            [self.anchor[0], step + self.anchor[1], step + self.anchor[2]],
            [
                step + self.anchor[0],
                step + self.anchor[1],
                step + self.anchor[2],
            ],
        ];

        for anchor in anchors.iter() {
            let mut coord: [PointType; 3] = [0.0; 3];
            for (&anchor_value, coord_ref, origin_value, diameter_value) in
                izip!(anchor, &mut coord, &domain.origin, &domain.diameter)
            {
                *coord_ref = origin_value
                    + diameter_value * (anchor_value as PointType) / (LEVEL_SIZE as PointType);
            }

            for component in &coord {
                serialized.push(*component);
            }
        }

        serialized
    }

    /// Return the anchor of the ancestor or descendent at the given level
    /// Note that if `level > self.level()` then the returned anchor is the
    /// same as `self.anchor`. The anchor
    pub fn anchor_at_level(&self, level: KeyType) -> [KeyType; 3] {
        let level_diff = (self.level() as i32) - (level as i32);

        if level_diff <= 0 {
            self.anchor().to_owned()
        } else {
            let mut parent = self.to_owned();
            for _ in 0..level_diff {
                parent = parent.parent();
            }

            parent.anchor().to_owned()
        }
    }

    /// Find key in a given direction.
    ///
    /// Returns the key obtained by moving direction\[j\] boxes into direction j
    /// starting from the anchor associated with the given key.
    /// Negative steps are possible. If the result is out of bounds,
    /// i.e. anchor\[j\] + direction\[j\] is negative or larger than the number of boxes
    /// across each dimension, `None` is returned. Otherwise, `Some(new_key)` is returned,
    /// where `new_key` is the Morton key after moving into the given direction.
    ///
    /// # Arguments
    /// * `direction` - A vector describing how many boxes we move along each coordinate direction.
    ///               Negative values are possible (meaning that we move backwards).
    pub fn find_key_in_direction(&self, direction: &[i64; 3]) -> Option<MortonKey> {
        let level = self.level();

        let max_number_of_boxes: i64 = 1 << DEEPEST_LEVEL;
        let step_multiplier: i64 = (1 << (DEEPEST_LEVEL - level)) as i64;

        let x: i64 = self.anchor[0] as i64;
        let y: i64 = self.anchor[1] as i64;
        let z: i64 = self.anchor[2] as i64;

        let x = x + step_multiplier * direction[0];
        let y = y + step_multiplier * direction[1];
        let z = z + step_multiplier * direction[2];

        if (x >= 0)
            & (y >= 0)
            & (z >= 0)
            & (x < max_number_of_boxes)
            & (y < max_number_of_boxes)
            & (z < max_number_of_boxes)
        {
            let new_anchor: [KeyType; 3] = [x as KeyType, y as KeyType, z as KeyType];
            let new_morton = encode_anchor(&new_anchor, level);
            Some(MortonKey {
                anchor: new_anchor,
                morton: new_morton,
            })
        } else {
            None
        }
    }

    /// Find all neighbors for to a given key.
    pub fn neighbors(&self) -> Vec<MortonKey> {
        DIRECTIONS
            .iter()
            .map(|d| self.find_key_in_direction(d))
            .filter(|d| !d.is_none())
            .map(|d| d.unwrap())
            .collect()
    }

    pub fn serialize(&self, domain: &Domain) -> Vec<f64> {
        serialize_morton_key(self, domain)
    }
}

impl PartialEq for MortonKey {
    fn eq(&self, other: &Self) -> bool {
        self.morton == other.morton
    }
}
impl Eq for MortonKey {}

impl Ord for MortonKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.morton.cmp(&other.morton)
    }
}

impl PartialOrd for MortonKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        //    less_than(self, other)
        Some(self.morton.cmp(&other.morton))
    }
}

impl Hash for MortonKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.morton.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Subroutine in less than function, equivalent to comparing floor of log_2(x). Adapted from [3].
    fn most_significant_bit(x: u64, y: u64) -> bool {
        (x < y) & (x < (x ^ y))
    }

    /// Implementation of Algorithm 12 in [1]. to compare the ordering of two **Morton Keys**. If key
    /// `a` is less than key `b`, this function evaluates to true.
    fn less_than(a: &MortonKey, b: &MortonKey) -> Option<bool> {
        // If anchors match, the one at the coarser level has the lesser Morton id.
        let same_anchor = (a.anchor[0] == b.anchor[0])
            & (a.anchor[1] == b.anchor[1])
            & (a.anchor[2] == b.anchor[2]);

        match same_anchor {
            true => {
                if a.level() < b.level() {
                    Some(true)
                } else {
                    Some(false)
                }
            }
            false => {
                let x = vec![
                    a.anchor[0] ^ b.anchor[0],
                    a.anchor[1] ^ b.anchor[1],
                    a.anchor[2] ^ b.anchor[2],
                ];

                let mut argmax = 0;

                for dim in 1..3 {
                    if most_significant_bit(x[argmax as usize], x[dim as usize]) {
                        argmax = dim
                    }
                }

                match argmax {
                    0 => {
                        if a.anchor[0] < b.anchor[0] {
                            Some(true)
                        } else {
                            Some(false)
                        }
                    }
                    1 => {
                        if a.anchor[1] < b.anchor[1] {
                            Some(true)
                        } else {
                            Some(false)
                        }
                    }
                    2 => {
                        if a.anchor[2] < b.anchor[2] {
                            Some(true)
                        } else {
                            Some(false)
                        }
                    }
                    _ => None,
                }
            }
        }
    }

    #[test]
    fn test_z_encode_table() {
        for (mut index, actual) in Z_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: KeyType = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift)) as KeyType;
                index = index >> 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_y_encode_table() {
        for (mut index, actual) in Y_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: KeyType = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift + 1)) as KeyType;
                index = index >> 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_x_encode_table() {
        for (mut index, actual) in X_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: KeyType = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift + 2)) as KeyType;
                index = index >> 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_z_decode_table() {
        for (index, &actual) in Z_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: KeyType = (index & 1) as KeyType;
            expected |= (((index >> 3) & 1) << 1) as KeyType;
            expected |= (((index >> 6) & 1) << 2) as KeyType;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_y_decode_table() {
        for (index, &actual) in Y_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: KeyType = ((index >> 1) & 1) as KeyType;
            expected |= (((index >> 4) & 1) << 1) as KeyType;
            expected |= (((index >> 7) & 1) << 2) as KeyType;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_x_decode_table() {
        for (index, &actual) in X_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: KeyType = ((index >> 2) & 1) as KeyType;
            expected |= (((index >> 5) & 1) << 1) as KeyType;
            expected |= (((index >> 8) & 1) << 2) as KeyType;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_encoding_decoding() {
        let anchor: [KeyType; 3] = [65535, 65535, 65535];

        let actual = decode_key(encode_anchor(&anchor, DEEPEST_LEVEL));

        assert_eq!(anchor, actual);
    }

    #[test]
    fn test_sorting() {
        let npoints = 1000;
        let mut range = rand::thread_rng();
        let mut points: Vec<[PointType; 3]> = Vec::new();

        for _ in 0..npoints {
            points.push([range.gen(), range.gen(), range.gen()]);
        }

        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        let mut keys: Vec<MortonKey> = points
            .iter()
            .map(|p| MortonKey::from_point(&p, &domain))
            .collect();

        keys.sort();
        let mut tree = MortonKeys { keys };
        tree.linearize();

        // Test that Z order is maintained when sorted
        for i in 0..(tree.keys.len() - 1) {
            let a = tree.keys[i];
            let b = tree.keys[i + 1];

            assert!(less_than(&a, &b).unwrap() | (a == b));
        }
    }

    #[test]
    fn test_find_children() {
        let key = MortonKey {
            morton: 0,
            anchor: [0, 0, 0],
        };
        let displacement = 1 << (DEEPEST_LEVEL - key.level() - 1);

        let expected: Vec<MortonKey> = vec![
            MortonKey {
                anchor: [0, 0, 0],
                morton: 1,
            },
            MortonKey {
                anchor: [displacement, 0, 0],
                morton: 0b100000000000000000000000000000000000000000000000000000000000001,
            },
            MortonKey {
                anchor: [0, displacement, 0],
                morton: 0b10000000000000000000000000000000000000000000000000000000000001,
            },
            MortonKey {
                anchor: [0, 0, displacement],
                morton: 0b1000000000000000000000000000000000000000000000000000000000001,
            },
            MortonKey {
                anchor: [displacement, displacement, 0],
                morton: 0b110000000000000000000000000000000000000000000000000000000000001,
            },
            MortonKey {
                anchor: [displacement, 0, displacement],
                morton: 0b101000000000000000000000000000000000000000000000000000000000001,
            },
            MortonKey {
                anchor: [0, displacement, displacement],
                morton: 0b11000000000000000000000000000000000000000000000000000000000001,
            },
            MortonKey {
                anchor: [displacement, displacement, displacement],
                morton: 0b111000000000000000000000000000000000000000000000000000000000001,
            },
        ];

        let children = key.children();

        for child in &children {
            println!("child {:?} expected {:?}", child, expected);
            assert!(expected.contains(child));
        }
    }

    #[test]
    fn test_ancestors() {
        let domain: Domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let point = [0.5, 0.5, 0.5];

        let key = MortonKey::from_point(&point, &domain);

        let mut ancestors: Vec<MortonKey> = key.ancestors().into_iter().collect();
        ancestors.sort();

        // Test that all ancestors found
        let mut current_level = 0;
        for &ancestor in &ancestors {
            assert!(ancestor.level() == current_level);
            current_level += 1;
        }

        // Test that the ancestors include the key at the leaf level
        assert!(ancestors.contains(&key));
    }

    #[test]
    pub fn test_finest_ancestor() {
        // Trivial case
        let key: MortonKey = MortonKey {
            anchor: [0, 0, 0],
            morton: 0,
        };
        let result = key.finest_ancestor(&key);
        let expected: MortonKey = MortonKey {
            anchor: [0, 0, 0],
            morton: 0,
        };
        assert!(result == expected);

        // Standard case
        let displacement = 1 << (DEEPEST_LEVEL - key.level() - 1);
        let a: MortonKey = MortonKey {
            anchor: [0, 0, 0],
            morton: 16,
        };
        let b: MortonKey = MortonKey {
            anchor: [displacement, displacement, displacement],
            morton: 0b111000000000000000000000000000000000000000000000000000000000001,
        };
        let result = a.finest_ancestor(&b);
        let expected: MortonKey = MortonKey {
            anchor: [0, 0, 0],
            morton: 0,
        };
        assert!(result == expected);
    }

    #[test]
    pub fn test_neighbors() {
        let point = [0.5, 0.5, 0.5];
        let domain: Domain = Domain {
            diameter: [1., 1., 1.],
            origin: [0., 0., 0.],
        };
        let key = MortonKey::from_point(&point, &domain);

        // Simple case, at the leaf level
        {
            let mut result = key.neighbors();
            result.sort();

            // Test that we get the expected number of neighbors
            assert!(result.len() == 26);

            // Test that the displacements are correct
            let displacement = 1 << (DEEPEST_LEVEL - key.level()) as i64;
            let anchor = key.anchor;
            let expected: [[i64; 3]; 26] = [
                [-displacement, -displacement, -displacement],
                [-displacement, -displacement, 0],
                [-displacement, -displacement, displacement],
                [-displacement, 0, -displacement],
                [-displacement, 0, 0],
                [-displacement, 0, displacement],
                [-displacement, displacement, -displacement],
                [-displacement, displacement, 0],
                [-displacement, displacement, displacement],
                [0, -displacement, -displacement],
                [0, -displacement, 0],
                [0, -displacement, displacement],
                [0, 0, -displacement],
                [0, 0, displacement],
                [0, displacement, -displacement],
                [0, displacement, 0],
                [0, displacement, displacement],
                [displacement, -displacement, -displacement],
                [displacement, -displacement, 0],
                [displacement, -displacement, displacement],
                [displacement, 0, -displacement],
                [displacement, 0, 0],
                [displacement, 0, displacement],
                [displacement, displacement, -displacement],
                [displacement, displacement, 0],
                [displacement, displacement, displacement],
            ];

            let mut expected: Vec<MortonKey> = expected
                .iter()
                .map(|n| {
                    [
                        (n[0] + (anchor[0] as i64)) as u64,
                        (n[1] + (anchor[1] as i64)) as u64,
                        (n[2] + (anchor[2] as i64)) as u64,
                    ]
                })
                .map(|anchor| MortonKey::from_anchor(&anchor))
                .collect();
            expected.sort();

            for i in 0..26 {
                assert!(expected[i] == result[i]);
            }
        }

        // More complex case, in the middle of the tree
        {
            let parent = key.parent().parent().parent();
            let mut result = parent.neighbors();
            result.sort();

            // Test that we get the expected number of neighbors
            assert!(result.len() == 26);

            // Test that the displacements are correct
            let displacement = 1 << (DEEPEST_LEVEL - parent.level()) as i64;
            let anchor = key.anchor;
            let expected: [[i64; 3]; 26] = [
                [-displacement, -displacement, -displacement],
                [-displacement, -displacement, 0],
                [-displacement, -displacement, displacement],
                [-displacement, 0, -displacement],
                [-displacement, 0, 0],
                [-displacement, 0, displacement],
                [-displacement, displacement, -displacement],
                [-displacement, displacement, 0],
                [-displacement, displacement, displacement],
                [0, -displacement, -displacement],
                [0, -displacement, 0],
                [0, -displacement, displacement],
                [0, 0, -displacement],
                [0, 0, displacement],
                [0, displacement, -displacement],
                [0, displacement, 0],
                [0, displacement, displacement],
                [displacement, -displacement, -displacement],
                [displacement, -displacement, 0],
                [displacement, -displacement, displacement],
                [displacement, 0, -displacement],
                [displacement, 0, 0],
                [displacement, 0, displacement],
                [displacement, displacement, -displacement],
                [displacement, displacement, 0],
                [displacement, displacement, displacement],
            ];

            let mut expected: Vec<MortonKey> = expected
                .iter()
                .map(|n| {
                    [
                        (n[0] + (anchor[0] as i64)) as u64,
                        (n[1] + (anchor[1] as i64)) as u64,
                        (n[2] + (anchor[2] as i64)) as u64,
                    ]
                })
                .map(|anchor| MortonKey::from_anchor(&anchor))
                .map(|key| MortonKey {
                    anchor: key.anchor,
                    morton: ((key.morton >> LEVEL_DISPLACEMENT) << LEVEL_DISPLACEMENT)
                        | parent.level(),
                })
                .collect();
            expected.sort();

            for i in 0..26 {
                assert!(expected[i] == result[i]);
            }
        }
    }
}
