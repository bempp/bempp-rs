use std::collections::HashMap;

use crate::types::{Charge64, ChargeDict, GlobalIdx};

pub fn build_charge_dict(global_idxs: &[GlobalIdx], charges: &[Charge64]) -> ChargeDict {
    let mut res: ChargeDict = HashMap::new();
    for (&global_idx, &charge) in global_idxs.iter().zip(charges.iter()) {
        res.insert(global_idx, charge);
    }
    res
}
