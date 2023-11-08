//! Implementation of an equivalent MPI type for point data.
use crate::types::{
    morton::{KeyType, MortonKey},
    point::{Point, PointType},
};
use memoffset::offset_of;
use mpi::{
    datatype::{Equivalence, UncommittedUserDatatype, UserDatatype},
    Address,
};
use num::Float;

unsafe impl<T: Float + Equivalence> Equivalence for Point<T> {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1, 1, 1],
            &[
                offset_of!(Point<T>, coordinate) as Address,
                offset_of!(Point<T>, global_idx) as Address,
                offset_of!(Point<T>, base_key) as Address,
                offset_of!(Point<T>, encoded_key) as Address,
            ],
            &[
                UncommittedUserDatatype::contiguous(3, &T::equivalent_datatype()).as_ref(),
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
