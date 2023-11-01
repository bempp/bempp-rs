//! Implementation of an equivalent MPI type for Morton keys.
use crate::types::morton::{KeyType, MortonKey};
use memoffset::offset_of;
use mpi::{
    datatype::{Equivalence, UncommittedUserDatatype, UserDatatype},
    Address,
};

unsafe impl Equivalence for MortonKey {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1],
            &[
                offset_of!(MortonKey, anchor) as Address,
                offset_of!(MortonKey, morton) as Address,
            ],
            &[
                UncommittedUserDatatype::contiguous(3, &KeyType::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(1, &KeyType::equivalent_datatype()).as_ref(),
            ],
        )
    }
}
