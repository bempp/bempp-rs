#[derive(Clone, Debug, Default)]
pub struct Charge {
    /// Charge data
    pub data: f64,

    /// Global unique index.
    pub global_idx: usize
}

/// Container of **Points**.
#[derive(Clone, Debug, Default)]
pub struct Charges {
    /// A vector of Charges
    pub charges: Vec<Charge>,

    /// index for implementing the Iterator trait.
    pub index: usize,
}

