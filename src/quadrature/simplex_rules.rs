//! Get rules on simplices.

pub use crate::traits::types::ReferenceCell;

use crate::quadrature::simplex_rule_definitions::SIMPLEX_RULE_DEFINITIONS;
use crate::quadrature::types::NumericalQuadratureDefinition;
use crate::quadrature::types::QuadratureError;

/// Return a simplex rule for a given number of points.
///
/// If the rule does not exist `Err(())` is returned.
pub fn simplex_rule(
    cell_type: ReferenceCell,
    npoints: usize,
) -> Result<NumericalQuadratureDefinition, QuadratureError> {
    let dim: usize = match cell_type {
        ReferenceCell::Point => 0,
        ReferenceCell::Interval => 1,
        ReferenceCell::Triangle => 2,
        ReferenceCell::Quadrilateral => 2,
        ReferenceCell::Tetrahedron => 3,
        ReferenceCell::Hexahedron => 3,
        ReferenceCell::Prism => 3,
        ReferenceCell::Pyramid => 3,
    };

    if let Some((order, points, weights)) = SIMPLEX_RULE_DEFINITIONS
        .get(&cell_type)
        .unwrap()
        .get(&npoints)
    {
        Ok(NumericalQuadratureDefinition {
            dim,
            order: *order,
            npoints,
            weights: weights.to_vec(),
            points: points.to_vec(),
        })
    } else {
        Err(QuadratureError::RuleNotFound)
    }
}

/// For a given cell type return a vector with the numbers of points for which simplex rules are available.
pub fn available_rules(cell_type: ReferenceCell) -> Vec<usize> {
    SIMPLEX_RULE_DEFINITIONS
        .get(&cell_type)
        .unwrap()
        .iter()
        .map(|(npoints, _)| *npoints)
        .collect()
}

#[cfg(test)]
mod test {

    use super::*;
    use paste::paste;

    use approx::*;

    fn get_volume(cell_type: ReferenceCell) -> f64 {
        match cell_type {
            ReferenceCell::Point => 0.0,
            ReferenceCell::Interval => 1.0,
            ReferenceCell::Triangle => 0.5,
            ReferenceCell::Quadrilateral => 1.0,
            ReferenceCell::Tetrahedron => 1.0 / 6.0,
            ReferenceCell::Hexahedron => 1.0,
            ReferenceCell::Prism => 0.5,
            ReferenceCell::Pyramid => 1.0 / 3.0,
        }
    }

    macro_rules! test_cell {

        ($($cell:ident),+) => {

        $(
            paste! {

                #[test]
                fn [<test_volume_ $cell:lower>]() {
                    let cell_type = ReferenceCellType::[<$cell>];
                    let rules = available_rules(cell_type);
                    for npoints in rules {
                        let rule = simplex_rule(cell_type, npoints).unwrap();
                        let volume_actual: f64 = rule.weights.iter().sum();
                        let volume_expected = get_volume(cell_type);
                        assert_relative_eq!(volume_actual, volume_expected, max_relative=1E-14);
                        }

                }

            }
        )*
        };
    }

    test_cell!(
        Interval,
        Triangle,
        Quadrilateral,
        Tetrahedron,
        Hexahedron,
        Prism,
        Pyramid
    );
}
