//! Get rules on simplices.

use crate::quadrature::simplex_rule_definitions::SIMPLEX_RULE_DEFINITIONS;
use crate::quadrature::types::NumericalQuadratureDefinition;
use crate::quadrature::types::QuadratureError;
use ndelement::types::ReferenceCellType;

/// Return a simplex rule for a given number of points.
///
/// If the rule does not exist `Err(())` is returned.
pub fn simplex_rule(
    cell_type: ReferenceCellType,
    npoints: usize,
) -> Result<NumericalQuadratureDefinition, QuadratureError> {
    let dim: usize = match cell_type {
        ReferenceCellType::Point => 0,
        ReferenceCellType::Interval => 1,
        ReferenceCellType::Triangle => 2,
        ReferenceCellType::Quadrilateral => 2,
        ReferenceCellType::Tetrahedron => 3,
        ReferenceCellType::Hexahedron => 3,
        ReferenceCellType::Prism => 3,
        ReferenceCellType::Pyramid => 3,
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
pub fn available_rules(cell_type: ReferenceCellType) -> Vec<usize> {
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

    fn get_volume(cell_type: ReferenceCellType) -> f64 {
        match cell_type {
            ReferenceCellType::Point => 0.0,
            ReferenceCellType::Interval => 1.0,
            ReferenceCellType::Triangle => 0.5,
            ReferenceCellType::Quadrilateral => 1.0,
            ReferenceCellType::Tetrahedron => 1.0 / 6.0,
            ReferenceCellType::Hexahedron => 1.0,
            ReferenceCellType::Prism => 0.5,
            ReferenceCellType::Pyramid => 1.0 / 3.0,
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
