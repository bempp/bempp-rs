//! Data structures and methods for Cartesian Points in 3D.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::path::Path;

use hdf5::H5Type;
use memoffset::offset_of;
use mpi::{
    datatype::{Equivalence, UncommittedUserDatatype, UserDatatype},
    Address,
};
use serde::{Deserialize, Serialize};

use crate::{
    data::{HDF5, JSON},
    types::morton::{KeyType, MortonKey},
};

pub type PointType = f64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, H5Type)]
/// A 3D cartesian point, described by coordinate, a unique global index, and the Morton Key for
/// the octree node in which it lies. The ordering of Points is determined by their Morton Key.
pub struct Point {
    pub coordinate: [PointType; 3],
    pub global_idx: usize,
    pub key: MortonKey,
}

/// Vector of **Points**.
pub type Points = Vec<Point>;

unsafe impl Equivalence for Point {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1, 1],
            &[
                offset_of!(Point, coordinate) as Address,
                offset_of!(Point, global_idx) as Address,
                offset_of!(Point, key) as Address,
            ],
            &[
                UncommittedUserDatatype::contiguous(3, &PointType::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(1, &usize::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::structured(
                    &[1, 1],
                    &[
                        offset_of!(MortonKey, anchor) as Address,
                        offset_of!(MortonKey, morton) as Address,
                    ],
                    &[
                        UncommittedUserDatatype::contiguous(3, &KeyType::equivalent_datatype())
                            .as_ref(),
                        UncommittedUserDatatype::contiguous(1, &KeyType::equivalent_datatype())
                            .as_ref(),
                    ],
                )
                .as_ref(),
            ],
        )
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Point {}

impl Ord for Point {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // less_than(&self.morton, &other.morton)
        Some(self.key.cmp(&other.key))
    }
}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl JSON for Vec<Point> {}

impl HDF5<Point> for Vec<Point> {
    fn write_hdf5<P: AsRef<Path>>(&self, filename: P) -> hdf5::Result<()> {
        let file = hdf5::File::create(filename)?;
        let points = file.new_dataset::<Point>().create("points")?;
        points.write(self)?;

        Ok(())
    }

    fn read_hdf5<P: AsRef<Path>>(filepath: P) -> hdf5::Result<Vec<Point>> {
        let file = hdf5::File::open(filepath)?;
        let points = file.dataset("points")?;
        let points: Vec<Point> = points.read_raw::<Point>()?;

        Ok(points)
    }
}
