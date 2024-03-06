//! Helper functions to handle charge data.
use std::collections::HashMap;

use rlst_dense::{array::Array, base_array::BaseArray, data_container::VectorContainer};

use crate::{
    traits::FmmScalar,
    types::{Charge, ChargeDict, GlobalIdx},
};

/// Zip together ordered list of global indices with their associated charges in a dictionary.
///
/// # Arguments
/// * `global_idxs` - Unique global index associated with charge.
/// * `charges` - The charge associated with each unique global index.
// pub fn build_charge_dict<T: FmmScalar>(
//     global_idxs: &[GlobalIdx],
//     charges: &[Charge<T>],
// ) -> ChargeDict<T> {
//     let mut res: ChargeDict<T> = HashMap::new();
//     for (&global_idx, &charge) in global_idxs.iter().zip(charges.iter()) {
//         res.insert(global_idx, charge);
//     }
//     res
// }

pub type Charges<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

pub type Coordinates<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;
