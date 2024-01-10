//! Helper functions to handle charge data.
use std::collections::HashMap;

use num::Float;
use rlst_common::types::Scalar;

use crate::types::{Charge, ChargeDict, GlobalIdx};

/// Zip together ordered list of global indices with their associated charges in a dictionary.
///
/// # Arguments
/// * `global_idxs` - Unique global index associated with charge.
/// * `charges` - The charge associated with each unique global index.
pub fn build_charge_dict<T>(global_idxs: &[GlobalIdx], charges: &[Charge<T>]) -> ChargeDict<T>
where
    T: Float + Scalar + Default,
{
    let mut res: ChargeDict<T> = HashMap::new();
    for (&global_idx, &charge) in global_idxs.iter().zip(charges.iter()) {
        res.insert(global_idx, charge);
    }
    res
}
