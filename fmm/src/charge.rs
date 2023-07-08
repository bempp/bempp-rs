use std::collections::HashMap;

use crate::types::{ChargeDict, Charge, GlobalIdx};


pub fn build_charge_dict(global_idxs: &[GlobalIdx], charges: &[Charge])
-> ChargeDict
 {

    let mut res: ChargeDict = HashMap::new();
    for (&global_idx, &charge) in global_idxs.iter().zip(charges.iter()) {
        res.insert(global_idx, charge);
    }
    res
}