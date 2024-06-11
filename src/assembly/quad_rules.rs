//! Interface to quadrature rules

use std::collections::HashMap;

use num::cast;
use rlst::prelude::*;

use crate::{quadrature::simplex_rules::simplex_rule, traits::types::ReferenceCellType};

/// Stores a regular quadrature rule.
pub struct RegularQuadratureRule<T: RlstScalar> {
    /// Quadrature points.
    pub points: DynamicArray<T::Real, 2>,
    /// Quadrature weights.
    pub weights: DynamicArray<T::Real, 1>,
}

impl<T: RlstScalar> RegularQuadratureRule<T> {
    /// Return the number of quadrature points.
    pub fn number_of_points(&self) -> usize {
        self.weights.shape()[0]
    }
}

impl<T: RlstScalar> RegularQuadratureRule<T> {
    /// Create a new quadrature rule from quadrature options and cell type.
    pub fn new(opts: &QuadratureOptions, cell_type: ReferenceCellType) -> Self {
        let npts = opts.quadrature_degrees[&cell_type];

        let quad_rule = simplex_rule(cell_type, npts).unwrap();
        let mut points = rlst_dynamic_array2!(T::Real, [2, npts]);
        let mut weights = rlst_dynamic_array1!(T::Real, [npts]);
        for (elem_out, &elem_in) in itertools::izip!(points.iter_mut(), quad_rule.points.iter()) {
            *elem_out = cast::<f64, T::Real>(elem_in).unwrap();
        }
        for (elem_out, &elem_in) in itertools::izip!(weights.iter_mut(), quad_rule.weights.iter()) {
            *elem_out = cast::<f64, T::Real>(elem_in).unwrap();
        }
        Self { points, weights }
    }
}

/// Options for a batched assembler
pub struct QuadratureOptions {
    /// Number of points used in quadrature for non-singular integrals
    pub quadrature_degrees: HashMap<ReferenceCellType, usize>,
    /// Quadrature degrees to be used for singular integrals
    pub singular_quadrature_degrees: HashMap<(ReferenceCellType, ReferenceCellType), usize>,
}

impl Default for QuadratureOptions {
    fn default() -> Self {
        use ReferenceCellType::{Quadrilateral, Triangle};
        Self {
            quadrature_degrees: HashMap::from([(Triangle, 12), (Quadrilateral, 12)]),
            singular_quadrature_degrees: HashMap::from([
                ((Triangle, Triangle), 4),
                ((Quadrilateral, Quadrilateral), 4),
                ((Quadrilateral, Triangle), 4),
                ((Triangle, Quadrilateral), 4),
            ]),
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_number_of_quad_points() {
        let opts = QuadratureOptions::default();
        let quad_rule = RegularQuadratureRule::<f64>::new(&opts, ReferenceCellType::Triangle);
        println!(
            "Number of quadrature points {}",
            quad_rule.number_of_points()
        );
    }
}
