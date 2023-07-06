use std::{
    cmp::{Eq, Ord, Ordering, PartialEq},
    hash::{Hash, Hasher},
};

use crate::types::{Charge, Charges};

impl Hash for Charge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.global_idx.hash(state);
    }
}

impl PartialEq for Charge {
    fn eq(&self, other: &Self) -> bool {
        self.global_idx == other.global_idx
    }
}

impl Eq for Charge {}

impl Ord for Charge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.global_idx.cmp(&other.global_idx)
    }
}

impl PartialOrd for Charge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // less_than(&self.morton, &other.morton)
        Some(self.global_idx.cmp(&other.global_idx))
    }
}

impl Charges {
    pub fn new() -> Charges {
        Charges {
            charges: Vec::new(),
            index: 0,
        }
    }

    pub fn add(&mut self, item: Charge) {
        self.charges.push(item);
    }

    pub fn sort(&mut self) {
        self.charges.sort();
    }
}

impl Iterator for Charges {
    type Item = Charge;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.charges.len() {
            return None;
        }

        self.index += 1;
        self.charges.get(self.index).cloned()
    }
}

impl FromIterator<Charge> for Charges {
    fn from_iter<I: IntoIterator<Item = Charge>>(iter: I) -> Self {
        let mut c = Charges::new();

        for i in iter {
            c.add(i);
        }
        c
    }
}
