use std::collections::HashMap;

use cauchy::Scalar;
use num::Float;

use crate::types::{Charge, ChargeDict, GlobalIdx};

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
