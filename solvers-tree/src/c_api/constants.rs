//! Wrappers for project constants.

use crate::constants;
use crate::types::morton::KeyType;

#[no_mangle]
pub static LEVEL_DISPLACEMENT: usize = constants::LEVEL_DISPLACEMENT;

#[no_mangle]
pub static LEVEL_MASK: KeyType = constants::LEVEL_MASK;

#[no_mangle]
pub static BYTE_MASK: KeyType = constants::BYTE_MASK;

#[no_mangle]
pub static BYTE_DISPLACEMENT: KeyType = constants::BYTE_DISPLACEMENT;

#[no_mangle]
pub static NINE_BIT_MASK: KeyType = constants::NINE_BIT_MASK;

#[no_mangle]
pub static DIRECTIONS: [[i64; 3]; 26] = constants::DIRECTIONS;

#[no_mangle]
pub static Z_LOOKUP_ENCODE: [KeyType; 256] = constants::Z_LOOKUP_ENCODE;

#[no_mangle]
pub static Y_LOOKUP_ENCODE: [KeyType; 256] = constants::Y_LOOKUP_ENCODE;

#[no_mangle]
pub static X_LOOKUP_ENCODE: [KeyType; 256] = constants::X_LOOKUP_ENCODE;

#[no_mangle]
pub static Z_LOOKUP_DECODE: [KeyType; 512] = constants::Z_LOOKUP_DECODE;

#[no_mangle]
pub static Y_LOOKUP_DECODE: [KeyType; 512] = constants::Y_LOOKUP_DECODE;

#[no_mangle]
pub static X_LOOKUP_DECODE: [KeyType; 512] = constants::X_LOOKUP_DECODE;
