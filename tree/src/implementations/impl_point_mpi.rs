
use memoffset::offset_of;
use mpi::{
    datatype::{Equivalence, UncommittedUserDatatype, UserDatatype},
    Address,
};
use crate::{
    types::{
        point::{PointType, Point},
        morton::{KeyType, MortonKey}
    }
};


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
