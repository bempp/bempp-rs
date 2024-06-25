//! Duffy rules for quadrilaterals.
use crate::quadrature::{
    duffy::common::{create_quadrilateral_mapper, next_quadrilateral_vertex, transform_coords},
    simplex_rules::simplex_rule,
    types::{
        CellToCellConnectivity, NumericalQuadratureDefinition, QuadratureError,
        TestTrialNumericalQuadratureDefinition,
    },
};
use ndelement::types::ReferenceCellType;

fn identical_quadrilaterals(
    interval_rule: &NumericalQuadratureDefinition,
) -> TestTrialNumericalQuadratureDefinition {
    let NumericalQuadratureDefinition {
        dim,
        order,
        npoints,
        weights,
        points,
    } = interval_rule;

    let n_output_points = 8 * npoints * npoints * npoints * npoints;

    let mut test_output_points = Vec::<f64>::with_capacity(2 * n_output_points);
    let mut trial_output_points = Vec::<f64>::with_capacity(2 * n_output_points);
    let mut output_weights = Vec::<f64>::with_capacity(n_output_points);

    for index1 in 0..*npoints {
        for index2 in 0..*npoints {
            for index3 in 0..*npoints {
                for index4 in 0..*npoints {
                    let eta1 = points[index1];
                    let eta2 = points[index2];
                    let eta3 = points[index3];
                    let xi = points[index4];

                    // First part

                    let weight = weights[index1]
                        * weights[index2]
                        * weights[index3]
                        * weights[index4]
                        * xi
                        * (1.0 - xi)
                        * (1.0 - xi * eta1);

                    test_output_points.push((1.0 - xi) * eta3);
                    test_output_points.push((1.0 - xi * eta1) * eta2);
                    trial_output_points.push(xi + (1.0 - xi) * eta3);
                    trial_output_points.push(xi * eta1 + (1.0 - xi * eta1) * eta2);
                    output_weights.push(weight);

                    // Second part

                    test_output_points.push((1.0 - xi * eta1) * eta2);
                    test_output_points.push((1.0 - xi) * eta3);
                    trial_output_points.push(xi * eta1 + (1.0 - xi * eta1) * eta2);
                    trial_output_points.push(xi + (1.0 - xi) * eta3);
                    output_weights.push(weight);

                    // Third part

                    test_output_points.push((1.0 - xi) * eta3);
                    test_output_points.push(xi * eta1 + (1.0 - xi * eta1) * eta2);
                    trial_output_points.push(xi + (1.0 - xi) * eta3);
                    trial_output_points.push((1.0 - xi * eta1) * eta2);
                    output_weights.push(weight);

                    // Fourth part

                    test_output_points.push((1.0 - xi * eta1) * eta2);
                    test_output_points.push(xi + (1.0 - xi) * eta3);
                    trial_output_points.push(xi * eta1 + (1.0 - xi * eta1) * eta2);
                    trial_output_points.push((1.0 - xi) * eta3);
                    output_weights.push(weight);

                    // Fifth part

                    test_output_points.push(xi + (1.0 - xi) * eta3);
                    test_output_points.push((1.0 - xi * eta1) * eta2);
                    trial_output_points.push((1.0 - xi) * eta3);
                    trial_output_points.push(xi * eta1 + (1.0 - xi * eta1) * eta2);
                    output_weights.push(weight);

                    // Sixth part

                    test_output_points.push(xi * eta1 + (1.0 - xi * eta1) * eta2);
                    test_output_points.push((1.0 - xi) * eta3);
                    trial_output_points.push((1.0 - xi * eta1) * eta2);
                    trial_output_points.push(xi + (1.0 - xi) * eta3);
                    output_weights.push(weight);

                    // Seventh part

                    test_output_points.push(xi + (1.0 - xi) * eta3);
                    test_output_points.push(xi * eta1 + (1.0 - xi * eta1) * eta2);
                    trial_output_points.push((1.0 - xi) * eta3);
                    trial_output_points.push((1.0 - xi * eta1) * eta2);
                    output_weights.push(weight);

                    // Eigth part

                    test_output_points.push(xi * eta1 + (1.0 - xi * eta1) * eta2);
                    test_output_points.push(xi + (1.0 - xi) * eta3);
                    trial_output_points.push((1.0 - xi * eta1) * eta2);
                    trial_output_points.push((1.0 - xi) * eta3);
                    output_weights.push(weight);
                }
            }
        }
    }

    TestTrialNumericalQuadratureDefinition {
        dim: *dim,
        order: *order,
        npoints: n_output_points,
        weights: output_weights,
        test_points: test_output_points,
        trial_points: trial_output_points,
    }
}

fn edge_adjacent_quadrilaterals(
    interval_rule: &NumericalQuadratureDefinition,
    test_singular_edge_vertices: (usize, usize),
    trial_singular_edge_vertices: (usize, usize),
) -> TestTrialNumericalQuadratureDefinition {
    let NumericalQuadratureDefinition {
        dim,
        order,
        npoints,
        weights,
        points,
    } = interval_rule;

    let n_output_points = 6 * npoints * npoints * npoints * npoints;

    let mut test_output_points = Vec::<f64>::with_capacity(2 * n_output_points);
    let mut trial_output_points = Vec::<f64>::with_capacity(2 * n_output_points);
    let mut output_weights = Vec::<f64>::with_capacity(n_output_points);

    for index1 in 0..*npoints {
        for index2 in 0..*npoints {
            for index3 in 0..*npoints {
                for index4 in 0..*npoints {
                    let eta1 = points[index1];
                    let eta2 = points[index2];
                    let eta3 = points[index3];
                    let xi = points[index4];

                    // First part

                    let weight =
                        weights[index1] * weights[index2] * weights[index3] * weights[index4];

                    test_output_points.push((1.0 - xi) * eta3 + xi);
                    test_output_points.push(xi * eta2);
                    trial_output_points.push((1.0 - xi) * eta3);
                    trial_output_points.push(xi * eta1);
                    output_weights.push(weight * xi * xi * (1.0 - xi));

                    // Second part

                    test_output_points.push((1.0 - xi) * eta3);
                    test_output_points.push(xi * eta2);
                    trial_output_points.push(xi + (1.0 - xi) * eta3);
                    trial_output_points.push(xi * eta1);
                    output_weights.push(weight * xi * xi * (1.0 - xi));

                    // Third part

                    test_output_points.push((1.0 - xi * eta1) * eta3 + xi * eta1);
                    test_output_points.push(xi * eta2);
                    trial_output_points.push((1.0 - xi * eta1) * eta3);
                    trial_output_points.push(xi);
                    output_weights.push(weight * xi * xi * (1.0 - xi * eta1));

                    // Fourth part

                    test_output_points.push((1.0 - xi * eta1) * eta3 + xi * eta1);
                    test_output_points.push(xi);
                    trial_output_points.push((1.0 - xi * eta1) * eta3);
                    trial_output_points.push(xi * eta2);
                    output_weights.push(weight * xi * xi * (1.0 - xi * eta1));

                    // Fifth part

                    test_output_points.push((1.0 - xi * eta1) * eta3);
                    test_output_points.push(xi * eta2);
                    trial_output_points.push((1.0 - xi * eta1) * eta3 + xi * eta1);
                    trial_output_points.push(xi);
                    output_weights.push(weight * xi * xi * (1.0 - xi * eta1));

                    // Sixth part

                    test_output_points.push((1.0 - xi * eta1) * eta3);
                    test_output_points.push(xi);
                    trial_output_points.push((1.0 - xi * eta1) * eta3 + xi * eta1);
                    trial_output_points.push(xi * eta2);
                    output_weights.push(weight * xi * xi * (1.0 - xi * eta1));
                }
            }
        }
    }

    transform_coords(
        &mut test_output_points,
        &create_quadrilateral_mapper(test_singular_edge_vertices.0, test_singular_edge_vertices.1),
    );
    transform_coords(
        &mut trial_output_points,
        &create_quadrilateral_mapper(
            trial_singular_edge_vertices.0,
            trial_singular_edge_vertices.1,
        ),
    );

    TestTrialNumericalQuadratureDefinition {
        dim: *dim,
        order: *order,
        npoints: n_output_points,
        weights: output_weights,
        test_points: test_output_points,
        trial_points: trial_output_points,
    }
}

fn vertex_adjacent_quadrilaterals(
    interval_rule: &NumericalQuadratureDefinition,
    test_singular_vertex: usize,
    trial_singular_vertex: usize,
) -> TestTrialNumericalQuadratureDefinition {
    let NumericalQuadratureDefinition {
        dim,
        order,
        npoints,
        weights,
        points,
    } = interval_rule;

    let n_output_points = 4 * npoints * npoints * npoints * npoints;

    let mut test_output_points = Vec::<f64>::with_capacity(2 * n_output_points);
    let mut trial_output_points = Vec::<f64>::with_capacity(2 * n_output_points);
    let mut output_weights = Vec::<f64>::with_capacity(n_output_points);

    for index1 in 0..*npoints {
        for index2 in 0..*npoints {
            for index3 in 0..*npoints {
                for index4 in 0..*npoints {
                    let eta1 = points[index1];
                    let eta2 = points[index2];
                    let eta3 = points[index3];
                    let xi = points[index4];

                    // First part

                    let weight = weights[index1]
                        * weights[index2]
                        * weights[index3]
                        * weights[index4]
                        * xi
                        * xi
                        * xi;

                    test_output_points.push(xi);
                    test_output_points.push(xi * eta1);
                    trial_output_points.push(xi * eta2);
                    trial_output_points.push(xi * eta3);
                    output_weights.push(weight);

                    // Second part

                    test_output_points.push(xi * eta1);
                    test_output_points.push(xi);
                    trial_output_points.push(xi * eta2);
                    trial_output_points.push(xi * eta3);
                    output_weights.push(weight);

                    // Third part

                    test_output_points.push(xi * eta1);
                    test_output_points.push(xi * eta2);
                    trial_output_points.push(xi);
                    trial_output_points.push(xi * eta3);
                    output_weights.push(weight);

                    // Fourth part

                    test_output_points.push(xi * eta1);
                    test_output_points.push(xi * eta2);
                    trial_output_points.push(xi * eta3);
                    trial_output_points.push(xi);
                    output_weights.push(weight);
                }
            }
        }
    }

    transform_coords(
        &mut test_output_points,
        &create_quadrilateral_mapper(
            test_singular_vertex,
            next_quadrilateral_vertex(test_singular_vertex),
        ),
    );
    transform_coords(
        &mut trial_output_points,
        &create_quadrilateral_mapper(
            trial_singular_vertex,
            next_quadrilateral_vertex(trial_singular_vertex),
        ),
    );

    TestTrialNumericalQuadratureDefinition {
        dim: *dim,
        order: *order,
        npoints: n_output_points,
        weights: output_weights,
        test_points: test_output_points,
        trial_points: trial_output_points,
    }
}

/// Create a Duffy rule on a quadrilateral
pub fn quadrilateral_duffy(
    connectivity: &CellToCellConnectivity,
    npoints: usize,
) -> Result<TestTrialNumericalQuadratureDefinition, QuadratureError> {
    let rule = simplex_rule(ReferenceCellType::Interval, npoints)?;

    match connectivity.connectivity_dimension {
        // Identical triangles
        2 => Ok(identical_quadrilaterals(&rule)),
        0 => {
            // Quadrilaterals have adjacent vertex
            if connectivity.local_indices.len() != 1 {
                Err(QuadratureError::ConnectivityError)
            } else {
                let (test_singular_vertex, trial_singular_vertex) = connectivity.local_indices[0];
                Ok(vertex_adjacent_quadrilaterals(
                    &rule,
                    test_singular_vertex,
                    trial_singular_vertex,
                ))
            }
        }
        1 => {
            // Quadrilaterals have adjacent edge
            if connectivity.local_indices.len() != 2 {
                Err(QuadratureError::ConnectivityError)
            } else {
                let first_pair = connectivity.local_indices[0];
                let second_pair = connectivity.local_indices[1];

                let test_singular_edge = (first_pair.0, second_pair.0);
                let trial_singular_edge = (first_pair.1, second_pair.1);
                Ok(edge_adjacent_quadrilaterals(
                    &rule,
                    test_singular_edge,
                    trial_singular_edge,
                ))
            }
        }
        _ => Err(QuadratureError::ConnectivityError),
    }
}

#[cfg(test)]
mod test {

    use approx::assert_relative_eq;

    use super::*;

    fn laplace_green(x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
        let inv_dist = 1.0 / f64::sqrt((x1 - y1) * (x1 - y1) + (x2 - y2) * (x2 - y2));

        0.25 * std::f64::consts::FRAC_1_PI * inv_dist
    }

    #[test]
    fn test_identical_quadrilaterals() {
        let compute_integral = |npoints: usize| -> f64 {
            let connectivity = CellToCellConnectivity {
                connectivity_dimension: 2,
                local_indices: Vec::new(),
            };

            let singular_rule = quadrilateral_duffy(&connectivity, npoints).unwrap();

            let mut sum = 0.0;

            for index in 0..singular_rule.npoints {
                let (x1, x2) = (
                    singular_rule.test_points[2 * index],
                    singular_rule.test_points[2 * index + 1],
                );

                let (y1, y2) = (
                    singular_rule.trial_points[2 * index],
                    singular_rule.trial_points[2 * index + 1],
                );

                let weight = singular_rule.weights[index];

                sum += laplace_green(x1, x2, y1, y2) * weight;
            }

            sum
        };

        assert_relative_eq!(
            compute_integral(10),
            0.23660050220466244,
            epsilon = 0.0,
            max_relative = 1E-10
        );
    }

    #[test]
    fn test_edge_adjacent_quadrilaterals() {
        let compute_integral = |npoints: usize| -> f64 {
            let connectivity = CellToCellConnectivity {
                connectivity_dimension: 1,
                local_indices: vec![(1, 0), (3, 2)],
            };

            let singular_rule = quadrilateral_duffy(&connectivity, npoints).unwrap();

            let mut sum = 0.0;

            for index in 0..singular_rule.npoints {
                let (x1, x2) = (
                    singular_rule.test_points[2 * index],
                    singular_rule.test_points[2 * index + 1],
                );

                let (y1, y2) = (
                    singular_rule.trial_points[2 * index],
                    singular_rule.trial_points[2 * index + 1],
                );

                let weight = singular_rule.weights[index];

                sum += laplace_green(x1, x2, 1.0 + y1, y2) * weight;
            }

            sum
        };

        assert_relative_eq!(
            compute_integral(10),
            0.08850038917185951,
            epsilon = 0.0,
            max_relative = 1E-12
        );
    }

    #[test]
    fn test_vertex_adjacent_quadrilaterals() {
        let compute_integral = |npoints: usize| -> f64 {
            let connectivity = CellToCellConnectivity {
                connectivity_dimension: 0,
                local_indices: vec![(1, 2)],
            };

            let singular_rule = quadrilateral_duffy(&connectivity, npoints).unwrap();

            let mut sum = 0.0;

            for index in 0..singular_rule.npoints {
                let (x1, x2) = (
                    singular_rule.test_points[2 * index],
                    singular_rule.test_points[2 * index + 1],
                );

                let (y1, y2) = (
                    singular_rule.trial_points[2 * index],
                    singular_rule.trial_points[2 * index + 1],
                );

                let weight = singular_rule.weights[index];

                sum += laplace_green(x1, x2, 1.0 + y1, -1.0 + y2) * weight;
            }

            sum
        };

        assert_relative_eq!(
            compute_integral(10),
            0.05959972386088679,
            epsilon = 0.0,
            max_relative = 1E-13
        );
    }
}
