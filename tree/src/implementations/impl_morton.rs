//! Implementations of constructors and transformation methods for Morton keys, as well as traits for sorting, and handling containers of Morton keys.
use itertools::{izip, Itertools};
use std::{
    cmp::Ordering,
    collections::HashSet,
    error::Error,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
    vec,
};

use crate::{
    constants::{
        BYTE_DISPLACEMENT, BYTE_MASK, DEEPEST_LEVEL, DIRECTIONS, LEVEL_DISPLACEMENT, LEVEL_MASK,
        LEVEL_SIZE, NINE_BIT_MASK, X_LOOKUP_DECODE, X_LOOKUP_ENCODE, Y_LOOKUP_DECODE,
        Y_LOOKUP_ENCODE, Z_LOOKUP_DECODE, Z_LOOKUP_ENCODE,
    },
    implementations::helpers::find_corners,
    types::{
        domain::Domain,
        morton::{KeyType, MortonKey, MortonKeys},
        point::PointType,
    },
};

use bempp_traits::tree::MortonKeyInterface;

/// Remove overlaps in an iterable of keys, prefer smallest keys if overlaps.
/// Returns an owned vector of Morton Keys, hence requires a copy.
///
/// # Arguments
/// * `keys` - A slice of Morton Keys to be linearised.
fn linearize_keys(keys: &[MortonKey]) -> Vec<MortonKey> {
    let depth = keys.iter().map(|k| k.level()).max().unwrap();
    let mut key_set: HashSet<MortonKey> = keys.iter().cloned().collect();

    for level in (0..=depth).rev() {
        let work_set: Vec<&MortonKey> = keys.iter().filter(|&&k| k.level() == level).collect();

        for work_item in work_set.iter() {
            let mut ancestors = work_item.ancestors();
            ancestors.remove(work_item);
            for ancestor in ancestors.iter() {
                if key_set.contains(ancestor) {
                    key_set.remove(ancestor);
                }
            }
        }
    }

    let result: Vec<MortonKey> = key_set.into_iter().collect();
    result
}

/// Enforce a 2:1 balance on a slice of Morton Keys. This method relies on `complete'
/// Trees, i.e. there are no gaps between keys provided to the method in the iterator.
/// Returns a set of Morton Keys.
///
/// # Arguments
/// * `keys` - A slice of Morton Keys to be balanced.
fn balance_keys(keys: &[MortonKey]) -> HashSet<MortonKey> {
    let mut balanced: HashSet<MortonKey> = keys.iter().cloned().collect();
    for level in (0..=DEEPEST_LEVEL).rev() {
        let work_list: Vec<MortonKey> = balanced
            .iter()
            .filter(|&key| key.level() == level)
            .cloned()
            .collect();

        for key in work_list.iter() {
            let neighbors = key.neighbors();
            for neighbor in neighbors {
                let parent = neighbor.parent();

                if !balanced.contains(&neighbor) && !balanced.contains(&parent) {
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
    balanced
}

/// Complete the region between two keys with the minimum spanning boxes that cover the region.
/// Returns an owned vector of Morton Keys.
///
/// # Arguments
/// * `a` - Morton Key specifying the start of a region to be completed.
/// * `b` - Morton Key specifying the end of a region to be completed.
pub fn complete_region(a: &MortonKey, b: &MortonKey) -> Vec<MortonKey> {
    let mut a_ancestors: HashSet<MortonKey> = a.ancestors();
    let mut b_ancestors: HashSet<MortonKey> = b.ancestors();

    // Remove endpoints from ancestors
    a_ancestors.remove(a);
    b_ancestors.remove(b);

    let mut minimal_tree: Vec<MortonKey> = Vec::new();
    let mut work_list: Vec<MortonKey> = a.finest_ancestor(b).children().into_iter().collect();

    while let Some(current_item) = work_list.pop() {
        if (current_item > *a) & (current_item < *b) & !b_ancestors.contains(&current_item) {
            minimal_tree.push(current_item);
        } else if (a_ancestors.contains(&current_item)) | (b_ancestors.contains(&current_item)) {
            let mut children = current_item.children();
            work_list.append(&mut children);
        }
    }

    // Sort the minimal tree before returning
    minimal_tree.sort();
    minimal_tree
}

impl MortonKeys {
    pub fn new() -> MortonKeys {
        MortonKeys {
            keys: Vec::new(),
            index: 0,
        }
    }

    pub fn add(&mut self, item: MortonKey) {
        self.keys.push(item);
    }

    /// Complete the region between all elements in an vector of Morton keys that doesn't
    /// necessarily span the domain defined by its least and greatest nodes.
    pub fn complete(&mut self) {
        let a = self.keys.iter().min().unwrap();
        let b = self.keys.iter().max().unwrap();
        let completion = complete_region(a, b);
        let start_val = vec![*a];
        let end_val = vec![*b];
        self.keys = start_val
            .into_iter()
            .chain(completion)
            .chain(end_val)
            .collect();
    }

    /// Wrapper for linearising a container of Morton Keys.
    pub fn linearize(&mut self) {
        self.keys = linearize_keys(&self.keys);
    }

    /// Wrapper for sorting a container of Morton Keys.
    pub fn sort(&mut self) {
        self.keys.sort();
    }

    /// Enforce a 2:1 balance for a vector of Morton keys, and remove any overlaps.
    pub fn balance(&mut self) {
        self.keys = balance_keys(self).into_iter().collect();
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

impl Iterator for MortonKeys {
    type Item = MortonKey;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.keys.len() {
            return None;
        }

        self.index += 1;
        self.keys.get(self.index).copied()
    }
}

impl FromIterator<MortonKey> for MortonKeys {
    fn from_iter<I: IntoIterator<Item = MortonKey>>(iter: I) -> Self {
        let mut c = MortonKeys::new();

        for i in iter {
            c.add(i);
        }
        c
    }
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
/// Returns the 3 integer coordinates of the enclosing box.
///
/// # Arguments
/// * `point` - The (x, y, z) coordinates of the point to map.
/// * `level` - The level of the tree at which the point will be mapped.
/// * `domain` - The computational domain defined by the point set.
pub fn point_to_anchor(
    point: &[PointType; 3],
    level: KeyType,
    domain: &Domain,
) -> Result<[KeyType; 3], Box<dyn Error>> {
    // Check if point is in the domain

    let mut contained = Vec::new();
    for (&p, d, o) in izip!(point, domain.diameter, domain.origin) {
        contained.push((o <= p) && (p <= o + d));
    }
    let contained = contained.iter().all(|&x| x);

    match contained {
        true => {
            let mut anchor = [KeyType::default(); 3];

            let side_length: Vec<f64> = domain
                .diameter
                .iter()
                .map(|d| d / ((1 << level) as f64))
                .collect();

            let scaling_factor = 1 << (DEEPEST_LEVEL - level);

            for (a, p, o, s) in izip!(&mut anchor, point, &domain.origin, &side_length) {
                *a = (((p - o) / s).floor()) as KeyType * scaling_factor;
            }
            Ok(anchor)
        }
        false => {
            panic!("Point not in Domain")
        }
    }
}

/// Encode an anchor.
/// Returns the Morton key associated with the given anchor.
///
/// # Arguments
/// * `anchor` - A vector with 3 elements defining the integer coordinates.
/// * `level` - The level of the tree the anchor is encoded to.
pub fn encode_anchor(anchor: &[KeyType; 3], level: KeyType) -> KeyType {
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
    /// Construct a `MortonKey` type from a Morton index
    pub fn from_morton(morton: KeyType) -> Self {
        let anchor = decode_key(morton);

        MortonKey { anchor, morton }
    }

    /// Construct a `MortonKey` type from the anchor at a given level
    pub fn from_anchor(anchor: &[KeyType; 3], level: u64) -> Self {
        let morton = encode_anchor(anchor, level);

        MortonKey {
            anchor: *anchor,
            morton,
        }
    }

    /// Construct a `MortonKey` associated with the box that encloses the point on the deepest level.
    ///
    /// # Arguments
    /// * `point` - Cartesian coordinate for a given point.
    /// * `domain` - Domain associated with a given tree encoding.
    /// * `level` - level of octree on which to find the encoding.
    pub fn from_point(point: &[PointType; 3], domain: &Domain, level: u64) -> Self {
        let anchor = point_to_anchor(point, level, domain).unwrap();
        MortonKey::from_anchor(&anchor, level)
    }

    /// Find the transfer vector between two Morton keys in component form.
    ///
    /// # Arguments
    /// * `other` - A Morton Key with which to calculate a transfer vector to.
    pub fn find_transfer_vector_components(&self, &other: &MortonKey) -> [i64; 3] {
        // Only valid for keys at level 2 and below
        if self.level() < 2 || other.level() < 2 {
            panic!("Transfer vectors only computed for keys at levels deeper than 2")
        }

        let level_diff = DEEPEST_LEVEL - self.level();

        let a = decode_key(self.morton);
        let b = decode_key(other.morton);

        // Compute transfer vector
        let mut x = a[0] as i64 - b[0] as i64;
        let mut y = a[1] as i64 - b[1] as i64;
        let mut z = a[2] as i64 - b[2] as i64;

        // Convert to an absolute transfer vector, wrt to key level.
        x /= 2_i64.pow(level_diff as u32);
        y /= 2_i64.pow(level_diff as u32);
        z /= 2_i64.pow(level_diff as u32);

        [x, y, z]
    }

    /// Subroutine for converting components of a transfer vector into a unique, positive, checksum.
    ///
    /// # Arguments
    /// * `components` - A three vector corresponding to a transfer vector.
    pub fn find_transfer_vector_from_components(components: &[i64]) -> usize {
        let mut x = components[0];
        let mut y = components[1];
        let mut z = components[2];

        fn positive_map(num: &mut i64) {
            if *num < 0 {
                *num = 2 * (-1 * *num) + 1;
            } else {
                *num *= 2;
            }
        }

        // Compute checksum via mapping to positive integers.
        positive_map(&mut x);
        positive_map(&mut y);
        positive_map(&mut z);

        let mut checksum = x;
        checksum = (checksum << 16) | y;
        checksum = (checksum << 16) | z;

        checksum as usize
    }

    /// Checksum encoding unique transfer vector between this key, and another.
    /// ie. the vector other->self returned as an unsigned integer.
    ///
    /// # Arguments
    /// * `other` - A Morton Key with which to calculate a transfer vector to.
    pub fn find_transfer_vector(&self, &other: &MortonKey) -> usize {
        let tmp = self.find_transfer_vector_components(&other);
        MortonKey::find_transfer_vector_from_components(&tmp)
    }

    /// The physical diameter of a box specified by this Morton Key, calculated with respect to
    /// a Domain. Returns a stack allocated array for the size of the box width in each corresponding
    /// dimension.
    ///
    /// # Arguments
    /// `domain` - The physical domain with which we calculate the diameter with respect to.
    pub fn diameter(&self, domain: &Domain) -> [f64; 3] {
        domain
            .diameter
            .map(|x| 0.5f64.powf(self.level() as f64) * x)
    }

    /// The physical centre of a box specified by this Morton Key, calculated with respect to
    /// a Domain. Returns a stack allocated array for the coordinates of the centre,
    ///
    /// # Arguments
    /// * `domain` - The physical domain with which we calculate the centre with respect to.
    pub fn centre(&self, domain: &Domain) -> [f64; 3] {
        let mut result = [0f64; 3];

        let anchor_coordinate = self.to_coordinates(domain);
        let diameter = self.diameter(domain);

        for (i, (c, d)) in anchor_coordinate.iter().zip(diameter).enumerate() {
            result[i] = c + d / 2.0;
        }

        result
    }

    /// The anchor corresponding to this key.
    pub fn anchor(&self) -> &[KeyType; 3] {
        &self.anchor
    }

    /// The Morton Key in index form.
    pub fn morton(&self) -> KeyType {
        self.morton
    }

    /// The level of this key.
    pub fn level(&self) -> KeyType {
        self.morton & LEVEL_MASK
    }

    /// Return the parent of a Morton Key.
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

    /// Return the first child of a Morton Key.
    pub fn first_child(&self) -> Self {
        MortonKey {
            anchor: self.anchor,
            morton: 1 + self.morton,
        }
    }

    /// Return the first child of a Morton Key on the deepest level.
    pub fn finest_first_child(&self) -> Self {
        MortonKey {
            anchor: self.anchor,
            morton: DEEPEST_LEVEL - self.level() + self.morton,
        }
    }

    /// Return the last child of a Morton Key on the deepest level.
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

    /// Return all children of a Morton Key in sorted order.
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

        children.sort();
        children
    }

    /// Return all children of the parent of this Morton Key.
    pub fn siblings(&self) -> Vec<MortonKey> {
        self.parent().children()
    }

    /// Check if the key is ancestor of `other`.
    pub fn is_ancestor(&self, other: &MortonKey) -> bool {
        let ancestors = other.ancestors();
        ancestors.contains(self)
    }

    /// Check if key is descendant of another key.
    pub fn is_descendant(&self, other: &MortonKey) -> bool {
        other.is_ancestor(self)
    }

    /// Return set of all ancestors of this Morton Key.
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

    /// Return descendants `n` levels down from a key.
    ///
    /// # Arguments
    /// * `n` - The level below the key's level to return descendants from.
    pub fn descendants(&self, n: u64) -> Result<Vec<MortonKey>, Box<dyn Error>> {
        let valid: bool = self.level() + n <= DEEPEST_LEVEL;

        match valid {
            false => {
                panic!("Cannot find descendants below level {:?}", DEEPEST_LEVEL)
            }
            true => {
                let mut descendants = vec![*self];
                for _ in 0..n {
                    let mut tmp = Vec::<MortonKey>::new();
                    for key in descendants {
                        tmp.append(&mut key.children());
                    }
                    descendants = tmp;
                }
                Ok(descendants)
            }
        }
    }

    /// Find the finest ancestor of key and another key.
    ///
    /// # Arguments
    /// * `other` - The key with which we are calculating the shared finest ancestor with respect to.
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

    /// Return the coordinates of the anchor for this Morton Key.
    ///
    /// # Arguments
    /// * `domain` - The domain with which we are calculating with respect to.
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
    /// * `domain` - The domain with which we are calculating with respect to.
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

    /// Find all neighbors for to a given key. Filter out 'invalid' neighbours that lie outside of the tree domain.
    pub fn neighbors(&self) -> Vec<MortonKey> {
        DIRECTIONS
            .iter()
            .map(|d| self.find_key_in_direction(d))
            .filter(|d| !d.is_none())
            .map(|d| d.unwrap())
            .collect()
    }

    /// Find all neighbors for to a given key, even if they lie outside of the tree domain.
    pub fn all_neighbors(&self) -> Vec<Option<MortonKey>> {
        DIRECTIONS
            .iter()
            .map(|d| self.find_key_in_direction(d))
            .collect()
    }

    /// Check if two keys are adjacent with respect to each other when they are known to on the same tree level.
    pub fn is_adjacent_same_level(&self, other: &MortonKey) -> bool {
        // Calculate distance between centres of each node
        let da = 1 << (DEEPEST_LEVEL - self.level());
        let db = 1 << (DEEPEST_LEVEL - other.level());
        let ra = (da as f64) * 0.5;
        let rb = (db as f64) * 0.5;

        let ca: Vec<f64> = self.anchor.iter().map(|&x| (x as f64) + ra).collect();
        let cb: Vec<f64> = other.anchor.iter().map(|&x| (x as f64) + rb).collect();

        let distance: Vec<f64> = ca.iter().zip(cb.iter()).map(|(a, b)| b - a).collect();

        let min = -ra - rb;
        let max = ra + rb;
        let mut result = true;

        for &d in distance.iter() {
            if d > max || d < min {
                result = false
            }
        }

        result
    }

    /// Check if two keys are adjacent with respect to each other
    pub fn is_adjacent(&self, other: &MortonKey) -> bool {
        let ancestors = self.ancestors();
        let other_ancestors = other.ancestors();

        // If either key overlaps they cannot be adjacent.
        if ancestors.contains(other) || other_ancestors.contains(self) {
            false
        } else {
            self.is_adjacent_same_level(other)
        }
    }

    /// Compute the convolution grid centered at a given MortonKey and its respective surface grid. This method is used
    /// in the FFT sparsification of the Multipole to Local translation operator for Fast Multipole Methods constructed
    /// on regular grids, such as the kiFMM or bbFMM. Returns an owned vector corresponding to the coordinates of the
    /// convolution grid in column major order [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N], as well as a
    /// vector of grid indices.
    ///
    /// # Arguments
    /// * `order` - the order of expansions used in constructing the surface grid
    /// * `domain` - The physical domain with which Morton Keys are being constructed with respect to.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    /// * `conv_point` - The corner point on the surface grid against which the surface and  convolution grids are aligned.
    /// * `conv_point_corner_index` - The index of the corner of the surface grid that the conv point lies on.
    pub fn convolution_grid(
        &self,
        order: usize,
        domain: &Domain,
        alpha: f64,
        conv_point_corner: &[f64],
        conv_point_corner_index: usize,
    ) -> (Vec<f64>, Vec<usize>) {
        // Number of convolution points along each axis
        let n = 2 * order - 1;

        let dim: usize = 3;
        let ncoeffs = n.pow(dim as u32);
        let mut grid = vec![0f64; dim * ncoeffs];

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let conv_index = i + j * n + k * n * n;
                    grid[conv_index] = i as f64;
                    grid[(dim - 2) * ncoeffs + conv_index] = j as f64;
                    grid[(dim - 1) * ncoeffs + conv_index] = k as f64;
                }
            }
        }

        // Map conv points to indices
        let conv_idxs = grid.iter().clone().map(|&x| x as usize).collect();

        let diameter = self
            .diameter(domain)
            .iter()
            .map(|x| x * alpha)
            .collect_vec();

        // Shift and scale to embed surface grid inside convolution grid
        // Scale
        grid.iter_mut().for_each(|point| {
            *point *= 1.0 / ((n - 1) as f64); // normalize
            *point *= diameter[0]; // find diameter
            *point *= 2.0; // convolution grid is 2x as large
        });

        // Shift convolution grid to align with a specified corner of surface grid
        let corners = find_corners(&grid);

        let surface_point = [
            corners[conv_point_corner_index],
            corners[8 + conv_point_corner_index],
            corners[16 + conv_point_corner_index],
        ];

        let diff = conv_point_corner
            .iter()
            .zip(surface_point)
            .map(|(a, b)| a - b)
            .collect_vec();

        for i in 0..dim {
            grid[i * ncoeffs..(i + 1) * ncoeffs]
                .iter_mut()
                .for_each(|value| *value += diff[i]);
        }

        (grid, conv_idxs)
    }

    /// Compute surface grid for a given expansion order. Used in the discretisation of Fast Multipole
    /// Methods which require a regular grid, such as the kiFMM or bbFMM.
    /// Returns a tuple, the first element is an owned vector of the physical coordinates of the
    /// surface grid in column major order [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N].
    /// The second element is a vector of indices corresponding to each of these coordinates.
    ///
    /// # Arguments
    /// * `order` - The expansion order being used in the FMM simulation.
    pub fn surface_grid(order: usize) -> (Vec<f64>, Vec<usize>) {
        let dim = 3;
        let n_coeffs = 6 * (order - 1).pow(2) + 2;

        // Implicitly in column major order
        let mut surface: Vec<f64> = vec![0f64; dim * n_coeffs];

        // Bounds of the surface grid
        let lower = 0;
        let upper = order - 1;

        // Orders the surface grid, implicitly column major
        let mut idx = 0;

        for k in 0..order {
            for j in 0..order {
                for i in 0..order {
                    if (i >= lower && j >= lower && (k == lower || k == upper))
                        || (j >= lower && k >= lower && (i == lower || i == upper))
                        || (k >= lower && i >= lower && (j == lower || j == upper))
                    {
                        surface[idx] = i as f64;
                        surface[(dim - 2) * n_coeffs + idx] = j as f64;
                        surface[(dim - 1) * n_coeffs + idx] = k as f64;
                        idx += 1;
                    }
                }
            }
        }

        // Map surface points to multi-indices
        let surface_idxs = surface.iter().clone().map(|&x| x as usize).collect();
        // Shift and scale surface so that it's centered at the origin and has side length of 1
        surface.iter_mut().for_each(|point| {
            *point *= 2.0 / (order as f64 - 1.0);
        });

        surface.iter_mut().for_each(|point| *point -= 1.0);

        (surface, surface_idxs)
    }

    /// Compute a surface grid centered at this Morton Key, used in the discretisation of Fast Multipole
    /// Methods which require a regular grid, such as the kiFMM or bbFMM.
    ///
    /// # Arguments
    /// * `surface` - A general surface grid, computed for a given expansion order computed with the
    /// associated function `surface_grid`.
    /// * `domain` - The physical domain with which Morton Keys are being constructed with respect to.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    pub fn scale_surface(&self, surface: Vec<f64>, domain: &Domain, alpha: f64) -> Vec<f64> {
        let dim = 3;
        // Translate box to specified centre, and scale
        let scaled_diameter = self.diameter(domain);
        let dilated_diameter = scaled_diameter.map(|d| d * alpha);

        let mut scaled_surface = vec![0f64; surface.len()];

        let centre = self.centre(domain);

        let ncoeffs = surface.len() / 3;
        for i in 0..ncoeffs {
            scaled_surface[i] = (surface[i] * (dilated_diameter[0] / 2.0)) + centre[0];
            scaled_surface[(dim - 2) * ncoeffs + i] =
                (surface[(dim - 2) * ncoeffs + i] * (dilated_diameter[1] / 2.0)) + centre[1];
            scaled_surface[(dim - 1) * ncoeffs + i] =
                (surface[(dim - 1) * ncoeffs + i] * (dilated_diameter[2] / 2.0)) + centre[2];
        }

        scaled_surface
    }

    /// Compute the surface grid, centered at this Morton Key, for a given expansion order and alpha parameter. This is used
    /// in the discretisation of Fast Multipole Methods which require a regular grid, such as the kiFMM or bbFMM.
    ///
    /// # Arguments
    /// * `domain` - The physical domain with which Morton Keys are being constructed with respect to.
    /// * `order` - The expansion order being used in the FMM simulation.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    pub fn compute_surface(&self, domain: &Domain, order: usize, alpha: f64) -> Vec<f64> {
        let (surface, _) = MortonKey::surface_grid(order);

        self.scale_surface(surface, domain, alpha)
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
        Some(self.morton.cmp(&other.morton))
    }
}

impl Hash for MortonKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.morton.hash(state);
    }
}

impl MortonKeyInterface for MortonKey {
    type NodeIndices = MortonKeys;

    fn children(&self) -> Self::NodeIndices {
        MortonKeys {
            keys: self.children(),
            index: 0,
        }
    }

    fn parent(&self) -> Self {
        self.parent()
    }

    fn neighbors(&self) -> Self::NodeIndices {
        MortonKeys {
            keys: self.neighbors(),
            index: 0,
        }
    }

    fn is_adjacent(&self, other: &Self) -> bool {
        self.is_adjacent(other)
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rlst::dense::{RawAccess, Shape};
    use std::vec;

    use crate::implementations::helpers::points_fixture;

    use super::*;

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
                index >>= 1;
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
                index >>= 1;
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
                index >>= 1;
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
    fn test_siblings() {
        // Test that we get the same siblings for a pair of siblings
        let a = [0, 0, 0];
        let b = [1, 1, 1];

        let a = MortonKey::from_anchor(&a, DEEPEST_LEVEL);
        let b = MortonKey::from_anchor(&b, DEEPEST_LEVEL);
        let mut sa = a.siblings();
        let mut sb = b.siblings();
        sa.sort();
        sb.sort();

        for (a, b) in sa.iter().zip(sb.iter()) {
            assert_eq!(a, b)
        }
    }

    #[test]
    fn test_sorting() {
        let npoints = 1000;
        let points = points_fixture(npoints, Some(-1.), Some(1.0));

        let domain = Domain::from_local_points(points.data());

        let mut keys: Vec<MortonKey> = Vec::new();

        for i in 0..points.shape().0 {
            let point = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];

            keys.push(MortonKey::from_point(&point, &domain, DEEPEST_LEVEL));
        }

        // Add duplicates to keys, to test ordering in terms of equality
        let mut cpy: Vec<MortonKey> = keys.to_vec();
        keys.append(&mut cpy);

        // Add duplicates to ensure equality is also sorted
        let mut replica = keys.to_vec();
        keys.append(&mut replica);
        keys.sort();

        // Test that Z order is maintained when sorted
        for i in 0..(keys.len() - 1) {
            let a = keys[i];
            let b = keys[i + 1];
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

        let key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);

        let mut ancestors: Vec<MortonKey> = key.ancestors().into_iter().collect();
        ancestors.sort();

        // Test that all ancestors found
        for (current_level, &ancestor) in ancestors.iter().enumerate() {
            assert!(ancestor.level() == current_level.try_into().unwrap());
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
        let key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);

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
                .map(|anchor| MortonKey::from_anchor(&anchor, DEEPEST_LEVEL))
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
                .map(|anchor| MortonKey::from_anchor(&anchor, DEEPEST_LEVEL))
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

    #[test]
    pub fn test_morton_keys_iterator() {
        let npoints = 1000;
        let domain = Domain {
            origin: [-1.01, -1.01, -1.01],
            diameter: [2.0, 2.0, 2.0],
        };
        let min = Some(-1.01);
        let max = Some(0.99);

        let points = points_fixture(npoints, min, max);

        let mut keys = Vec::new();

        for i in 0..points.shape().0 {
            let point = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];
            keys.push(MortonKey::from_point(&point, &domain, DEEPEST_LEVEL))
        }

        let keys = MortonKeys { keys, index: 0 };

        // test that we can call keys as an iterator
        keys.iter().sorted();

        // test that iterator index resets to 0
        assert!(keys.index == 0);
    }

    #[test]
    fn test_linearize_keys() {
        let key = MortonKey {
            morton: 15,
            anchor: [0, 0, 0],
        };

        let ancestors: Vec<MortonKey> = key.ancestors().into_iter().collect();
        let linearized = linearize_keys(&ancestors);

        assert_eq!(linearized.len(), 1);
        assert_eq!(linearized[0], key);
    }

    #[test]
    fn test_point_to_anchor() {
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        // Test points in the domain
        let point = [0.9999, 0.9999, 0.9999];
        let level = 2;
        let anchor = point_to_anchor(&point, level, &domain);
        let expected = [49152, 49152, 49152];

        for (i, a) in anchor.unwrap().iter().enumerate() {
            assert_eq!(a, &expected[i])
        }

        let domain = Domain {
            origin: [-0.7, -0.6, -0.5],
            diameter: [1., 1., 1.],
        };

        let point = [-0.499, -0.499, -0.499];
        let level = 1;
        let anchor = point_to_anchor(&point, level, &domain);
        let expected = [0, 0, 0];

        for (i, a) in anchor.unwrap().iter().enumerate() {
            assert_eq!(a, &expected[i])
        }
    }

    #[test]
    #[should_panic(expected = "Point not in Domain")]
    fn test_point_to_anchor_fails() {
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        // Test a point not in the domain
        let point = [1.9, 0.9, 0.9];
        let level = 2;
        let _anchor = point_to_anchor(&point, level, &domain);
    }

    #[test]
    #[should_panic(expected = "Point not in Domain")]
    fn test_point_to_anchor_fails_negative_domain() {
        let domain = Domain {
            origin: [-0.5, -0.5, -0.5],
            diameter: [1., 1., 1.],
        };

        // Test a point not in the domain
        let point = [-0.51, -0.5, -0.5];
        let level = 2;
        let _anchor = point_to_anchor(&point, level, &domain);
    }

    #[test]
    fn test_encode_anchor() {
        let anchor = [1, 0, 1];
        let level = 1;
        let morton = encode_anchor(&anchor, level);
        let expected = 0b101000000000000001;
        assert_eq!(expected, morton);

        let anchor = [3, 3, 3];
        let level = 2;
        let morton = encode_anchor(&anchor, level);
        let expected = 0b111111000000000000010;
        assert_eq!(expected, morton);
    }

    #[test]
    fn test_find_descendants() {
        let key = MortonKey {
            morton: 0,
            anchor: [0, 0, 0],
        };

        let descendants = key.descendants(1).unwrap();
        assert_eq!(descendants.len(), 8);

        // Ensure this also works for other keys in hierarchy
        let key = descendants[0];
        let descendants = key.descendants(2).unwrap();
        assert_eq!(descendants.len(), 64);
    }

    #[test]
    #[should_panic(expected = "Cannot find descendants below level 16")]
    fn test_find_descendants_panics() {
        let key = MortonKey {
            morton: 0,
            anchor: [0, 0, 0],
        };
        let _descendants = key.descendants(17);
    }

    #[test]
    fn test_complete_region() {
        let a: MortonKey = MortonKey {
            anchor: [0, 0, 0],
            morton: 16,
        };
        let b: MortonKey = MortonKey {
            anchor: [65535, 65535, 65535],
            morton: 0b111111111111111111111111111111111111111111111111000000000010000,
        };

        let region = complete_region(&a, &b);

        let fa = a.finest_ancestor(&b);

        let min = *region.iter().min().unwrap();
        let max = *region.iter().max().unwrap();

        // Test that bounds are satisfied
        assert!(a <= min);
        assert!(b >= max);

        // Test that FCA is an ancestor of all nodes in the result
        for node in region.iter() {
            let ancestors = node.ancestors();
            assert!(ancestors.contains(&fa));
        }

        // Test that completed region doesn't contain its bounds
        assert!(!region.contains(&a));
        assert!(!region.contains(&b));

        // Test that the compeleted region doesn't contain any overlaps
        for node in region.iter() {
            let mut ancestors = node.ancestors();
            ancestors.remove(node);
            for ancestor in ancestors.iter() {
                assert!(!region.contains(ancestor))
            }
        }

        // Test that the region is sorted
        for i in 0..region.iter().len() - 1 {
            let a = region[i];
            let b = region[i + 1];

            assert!(a <= b);
        }
    }

    #[test]
    pub fn test_balance() {
        let a = MortonKey::from_anchor(&[0, 0, 0], DEEPEST_LEVEL);
        let b = MortonKey::from_anchor(&[1, 1, 1], DEEPEST_LEVEL);

        let mut complete = complete_region(&a, &b);
        let start_val = vec![a];
        let end_val = vec![b];
        complete = start_val
            .into_iter()
            .chain(complete.into_iter())
            .chain(end_val.into_iter())
            .collect();
        let mut tree = MortonKeys {
            keys: complete,
            index: 0,
        };

        tree.balance();
        tree.linearize();
        tree.sort();

        // Test for overlaps in balanced tree
        for key in tree.iter() {
            if !tree.iter().contains(key) {
                let mut ancestors = key.ancestors();
                ancestors.remove(key);

                for ancestor in ancestors.iter() {
                    assert!(!tree.keys.contains(ancestor));
                }
            }
        }

        // Test that adjacent keys are 2:1 balanced
        for key in tree.iter() {
            let adjacent_levels: Vec<u64> = tree
                .iter()
                .cloned()
                .filter(|k| key.is_adjacent(k))
                .map(|a| a.level())
                .collect();

            for l in adjacent_levels.iter() {
                assert!(l.abs_diff(key.level()) <= 1);
            }
        }
    }

    #[test]
    fn test_is_adjacent() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain {
            origin: [-0.1, -0.1, 0.1],
            diameter: [1., 1., 1.],
        };

        let key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);

        let mut ancestors = key.ancestors();
        ancestors.remove(&key);

        // Test that overlapping nodes are not adjacent
        for a in ancestors.iter() {
            assert!(!key.is_adjacent(a))
        }

        // Test that siblings & neighbours are adjacent
        let siblings = key.siblings();
        let neighbors = key.neighbors();

        for s in siblings.iter() {
            if *s != key {
                assert!(key.is_adjacent(s));
            }
        }

        for n in neighbors.iter() {
            assert!(key.is_adjacent(n));
        }

        // Test keys on different levels
        let anchor_a = [0, 0, 0];
        let a = MortonKey::from_anchor(&anchor_a, DEEPEST_LEVEL - 1);
        let anchor_b = [2, 2, 2];
        let b = MortonKey::from_anchor(&anchor_b, DEEPEST_LEVEL);
        assert!(a.is_adjacent(&b));
    }

    #[test]
    fn test_encoding_is_always_absolute() {
        let point = [-0.099999, -0.099999, -0.099999];
        let domain: Domain = Domain {
            origin: [-0.1, -0.1, -0.1],
            diameter: [1., 1., 1.],
        };

        let a = MortonKey::from_point(&point, &domain, 1);
        let b = MortonKey::from_point(&point, &domain, 16);
        assert_ne!(a, b);
        assert_eq!(a.anchor, b.anchor);
    }

    #[test]
    fn test_transfer_vector() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        // Test scale independence of transfer vectors
        let a = MortonKey::from_point(&point, &domain, 2);
        let other = a.siblings()[2];
        let res_a = a.find_transfer_vector(&other);

        let b = MortonKey::from_point(&point, &domain, 16);
        let other = b.siblings()[2];
        let res_b = b.find_transfer_vector(&other);

        assert_eq!(res_a, res_b);

        // Test translational invariance of transfer vector
        let a = MortonKey::from_point(&point, &domain, 2);
        let other = a.siblings()[2];
        let res_a = a.find_transfer_vector(&other);

        let shifted_point = [0.1, 0.1, 0.1];
        let b = MortonKey::from_point(&shifted_point, &domain, 2);
        let other = b.siblings()[2];
        let res_b = b.find_transfer_vector(&other);

        assert_eq!(res_a, res_b);
    }

    #[test]
    #[should_panic(expected = "Transfer vectors only computed for keys at levels deeper than 2")]
    fn test_transfer_vector_panic() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let key = MortonKey::from_point(&point, &domain, 1);
        let sibling = key.siblings()[0];
        key.find_transfer_vector(&sibling);
    }

    #[test]
    fn test_surface_grid() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let key = MortonKey::from_point(&point, &domain, 0);

        let order = 2;
        let alpha = 1.;
        let dim = 3;
        let ncoeffs = 6 * (order - 1_usize).pow(2) + 2;

        // Test lengths
        let surface = key.compute_surface(&domain, order, alpha);
        assert_eq!(surface.len(), ncoeffs * dim);

        let (surface, surface_idxs) = MortonKey::surface_grid(order);
        assert_eq!(surface.len(), ncoeffs * dim);
        assert_eq!(surface_idxs.len(), ncoeffs * dim);

        let mut expected = vec![[0usize; 3]; ncoeffs];
        let lower = 0;
        let upper = order - 1;
        let mut idx = 0;
        for k in 0..order {
            for j in 0..order {
                for i in 0..order {
                    if (i >= lower && j >= lower && (k == lower || k == upper))
                        || (j >= lower && k >= lower && (i == lower || i == upper))
                        || (k >= lower && i >= lower && (j == lower || j == upper))
                    {
                        expected[idx] = [i, j, k];
                        idx += 1;
                    }
                }
            }
        }

        // Test ordering.
        for i in 0..ncoeffs {
            let point = vec![
                surface_idxs[i],
                surface_idxs[i + ncoeffs],
                surface_idxs[i + 2 * ncoeffs],
            ];
            assert_eq!(point, expected[i]);
        }

        // Test scaling
        let level = 2;
        let key = MortonKey::from_point(&point, &domain, level);
        let surface = key.compute_surface(&domain, order, alpha);

        let min_x = surface
            .iter()
            .take(ncoeffs)
            .fold(f64::INFINITY, |a, &b| a.min(b));

        let max_x = surface.iter().take(ncoeffs).fold(0f64, |a, &b| a.max(b));

        let diam_x = max_x - min_x;

        let expected = key.diameter(&domain)[0];
        assert_eq!(diam_x, expected);

        // Test shifting
        let point = [0.1, 0.2, 0.3];
        let level = 2;
        let key = MortonKey::from_point(&point, &domain, level);
        let surface = key.compute_surface(&domain, order, alpha);
        let expected = key.centre(&domain);

        let c_x = surface.iter().take(ncoeffs).fold(0f64, |a, &b| a + b) / (ncoeffs as f64);
        let c_y = surface
            .iter()
            .skip(ncoeffs)
            .take(ncoeffs)
            .fold(0f64, |a, &b| a + b)
            / (ncoeffs as f64);
        let c_z = surface
            .iter()
            .skip(2 * ncoeffs)
            .take(ncoeffs)
            .fold(0f64, |a, &b| a + b)
            / (ncoeffs as f64);

        let result = vec![c_x, c_y, c_z];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_convolution_grid() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        let order = 5;
        let alpha = 1.0;

        let key = MortonKey::from_point(&point, &domain, 0);

        let surface_grid = key.compute_surface(&domain, order, alpha);

        // Place convolution grid on max corner
        let corners = find_corners(&surface_grid);
        let ncorners = corners.len() / 3;
        let conv_point_corner_index = 7;

        let conv_point = vec![
            corners[conv_point_corner_index],
            corners[ncorners + conv_point_corner_index],
            corners[2 * ncorners + conv_point_corner_index],
        ];

        let (conv_grid, _) =
            key.convolution_grid(order, &domain, alpha, &conv_point, conv_point_corner_index);

        // Test that surface grid is embedded in convolution grid
        let mut surface = Vec::new();
        let nsurf = surface_grid.len() / 3;
        for i in 0..nsurf {
            surface.push([
                surface_grid[i],
                surface_grid[i + nsurf],
                surface_grid[i + 2 * nsurf],
            ])
        }

        let mut convolution = Vec::new();
        let nconv = conv_grid.len() / 3;
        for i in 0..nconv {
            convolution.push([conv_grid[i], conv_grid[i + nconv], conv_grid[i + 2 * nconv]])
        }
        assert!(surface.iter().all(|point| convolution.contains(point)));
    }
}
