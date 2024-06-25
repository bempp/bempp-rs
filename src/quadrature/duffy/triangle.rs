//! Duffy rules for triangles.
use crate::quadrature::{
    duffy::common::{create_triangle_mapper, next_triangle_vertex, transform_coords},
    simplex_rules::simplex_rule,
    types::{
        CellToCellConnectivity, NumericalQuadratureDefinition, QuadratureError,
        TestTrialNumericalQuadratureDefinition,
    },
};
use ndelement::types::ReferenceCellType;

fn identical_triangles(
    interval_rule: &NumericalQuadratureDefinition,
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

                    let eta12 = eta1 * eta2;
                    let eta123 = eta12 * eta3;

                    // First part

                    let weight = weights[index1]
                        * weights[index2]
                        * weights[index3]
                        * weights[index4]
                        * xi
                        * xi
                        * xi
                        * eta1
                        * eta1
                        * eta2;

                    test_output_points.push(xi);
                    test_output_points.push(xi * (1.0 - eta1 + eta12));
                    trial_output_points.push(xi * (1.0 - eta123));
                    trial_output_points.push(xi * (1.0 - eta1));
                    output_weights.push(weight);

                    // Second part

                    test_output_points.push(xi * (1.0 - eta123));
                    test_output_points.push(xi * (1.0 - eta1));
                    trial_output_points.push(xi);
                    trial_output_points.push(xi * (1.0 - eta1 + eta12));
                    output_weights.push(weight);

                    // Third part

                    test_output_points.push(xi);
                    test_output_points.push(xi * (eta1 - eta12 + eta123));
                    trial_output_points.push(xi * (1.0 - eta12));
                    trial_output_points.push(xi * (eta1 - eta12));
                    output_weights.push(weight);

                    // Fourth part

                    test_output_points.push(xi * (1.0 - eta12));
                    test_output_points.push(xi * (eta1 - eta12));
                    trial_output_points.push(xi);
                    trial_output_points.push(xi * (eta1 - eta12 + eta123));
                    output_weights.push(weight);

                    // Fifth part

                    test_output_points.push(xi * (1.0 - eta123));
                    test_output_points.push(xi * (eta1 - eta123));
                    trial_output_points.push(xi);
                    trial_output_points.push(xi * (eta1 - eta12));
                    output_weights.push(weight);

                    // Sixth part

                    test_output_points.push(xi);
                    test_output_points.push(xi * (eta1 - eta12));
                    trial_output_points.push(xi * (1.0 - eta123));
                    trial_output_points.push(xi * (eta1 - eta123));
                    output_weights.push(weight);
                }
            }
        }
    }

    let mapper = create_triangle_mapper(0, 1);

    transform_coords(&mut test_output_points, &mapper);
    transform_coords(&mut trial_output_points, &mapper);

    TestTrialNumericalQuadratureDefinition {
        dim: *dim,
        order: *order,
        npoints: n_output_points,
        weights: output_weights,
        test_points: test_output_points,
        trial_points: trial_output_points,
    }
}

fn edge_adjacent_triangles(
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

    let n_output_points = 5 * npoints * npoints * npoints * npoints;

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

                    let eta12 = eta1 * eta2;
                    let eta123 = eta12 * eta3;

                    // First part

                    let weight = weights[index1]
                        * weights[index2]
                        * weights[index3]
                        * weights[index4]
                        * xi
                        * xi
                        * xi
                        * eta1
                        * eta1;

                    test_output_points.push(xi);
                    test_output_points.push(xi * eta1 * eta3);
                    trial_output_points.push(xi * (1.0 - eta12));
                    trial_output_points.push(xi * (eta1 - eta12));
                    output_weights.push(weight);

                    // Second part

                    test_output_points.push(xi);
                    test_output_points.push(xi * eta1);
                    trial_output_points.push(xi * (1.0 - eta123));
                    trial_output_points.push(xi * (eta12 - eta123));
                    output_weights.push(eta2 * weight);

                    // Third part

                    test_output_points.push(xi * (1.0 - eta12));
                    test_output_points.push(xi * (eta1 - eta12));
                    trial_output_points.push(xi);
                    trial_output_points.push(xi * eta123);
                    output_weights.push(eta2 * weight);

                    // Fourth part

                    test_output_points.push(xi * (1.0 - eta123));
                    test_output_points.push(xi * (eta12 - eta123));
                    trial_output_points.push(xi);
                    trial_output_points.push(xi * eta1);
                    output_weights.push(eta2 * weight);

                    // Fifth part

                    test_output_points.push(xi * (1.0 - eta123));
                    test_output_points.push(xi * (eta1 - eta123));
                    trial_output_points.push(xi);
                    trial_output_points.push(xi * eta12);
                    output_weights.push(eta2 * weight);
                }
            }
        }
    }

    transform_coords(
        &mut test_output_points,
        &create_triangle_mapper(test_singular_edge_vertices.0, test_singular_edge_vertices.1),
    );
    transform_coords(
        &mut trial_output_points,
        &create_triangle_mapper(
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

fn vertex_adjacent_triangles(
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

    let n_output_points = 2 * npoints * npoints * npoints * npoints;

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
                        * xi
                        * eta2;

                    test_output_points.push(xi);
                    test_output_points.push(xi * eta1);
                    trial_output_points.push(xi * eta2);
                    trial_output_points.push(xi * eta2 * eta3);
                    output_weights.push(weight);

                    // Second part

                    test_output_points.push(xi * eta2);
                    test_output_points.push(xi * eta2 * eta3);
                    trial_output_points.push(xi);
                    trial_output_points.push(xi * eta1);
                    output_weights.push(weight);
                }
            }
        }
    }

    transform_coords(
        &mut test_output_points,
        &create_triangle_mapper(
            test_singular_vertex,
            next_triangle_vertex(test_singular_vertex),
        ),
    );
    transform_coords(
        &mut trial_output_points,
        &create_triangle_mapper(
            trial_singular_vertex,
            next_triangle_vertex(trial_singular_vertex),
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

/// Create a Duffy rule on a triangle
pub fn triangle_duffy(
    connectivity: &CellToCellConnectivity,
    npoints: usize,
) -> Result<TestTrialNumericalQuadratureDefinition, QuadratureError> {
    let rule = simplex_rule(ReferenceCellType::Interval, npoints)?;

    match connectivity.connectivity_dimension {
        // Identical triangles
        2 => Ok(identical_triangles(&rule)),
        0 => {
            // Triangles have adjacent vertex
            if connectivity.local_indices.len() != 1 {
                Err(QuadratureError::ConnectivityError)
            } else {
                let (test_singular_vertex, trial_singular_vertex) = connectivity.local_indices[0];
                Ok(vertex_adjacent_triangles(
                    &rule,
                    test_singular_vertex,
                    trial_singular_vertex,
                ))
            }
        }
        1 => {
            // Triangles have adjacent edge
            if connectivity.local_indices.len() != 2 {
                Err(QuadratureError::ConnectivityError)
            } else {
                let first_pair = connectivity.local_indices[0];
                let second_pair = connectivity.local_indices[1];

                let test_singular_edge = (first_pair.0, second_pair.0);
                let trial_singular_edge = (first_pair.1, second_pair.1);
                Ok(edge_adjacent_triangles(
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
    use approx::{assert_relative_eq, assert_ulps_eq};

    use super::*;

    fn laplace_green(x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
        let inv_dist = 1.0 / f64::sqrt((x1 - y1) * (x1 - y1) + (x2 - y2) * (x2 - y2));

        0.25 * std::f64::consts::FRAC_1_PI * inv_dist
    }

    #[test]
    fn test_edge_transformer() {
        let tuple_compare = |tuple1: (f64, f64), tuple2: (f64, f64)| {
            assert_ulps_eq!(tuple1.0, tuple2.0);
            assert_ulps_eq!(tuple1.1, tuple2.1);
        };

        let transformer = create_triangle_mapper(1, 2);

        tuple_compare(transformer((0.0, 0.0)), (1.0, 0.0));
        tuple_compare(transformer((1.0, 0.0)), (0.0, 1.0));
        tuple_compare(transformer((1.0, 1.0)), (0.0, 0.0));

        let transformer = create_triangle_mapper(2, 0);
        tuple_compare(transformer((0.0, 0.0)), (0.0, 1.0));
        tuple_compare(transformer((1.0, 0.0)), (0.0, 0.0));
        tuple_compare(transformer((1.0, 1.0)), (1.0, 0.0));
    }

    #[test]
    fn test_identical_triangles() {
        let compute_integral = |npoints: usize| -> f64 {
            let connectivity = CellToCellConnectivity {
                connectivity_dimension: 2,
                local_indices: Vec::new(),
            };

            let singular_rule = triangle_duffy(&connectivity, npoints).unwrap();

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

        // The comparison values have been created with Bempp-cl
        // For the first 4 digits 0.0798 we also have independent
        // confirmation.
        // Comparisons were also performed with legacy Bempp.
        // The corresponding results are (order parameter refers to
        // the corresponding orders in legacy Bempp)
        // Order 2: 0.079267768872634842
        // Order 6: 0.07980853550151136
        // Order 14: 0.079821438597427713

        assert_relative_eq!(
            compute_integral(2),
            0.07926776887263483,
            epsilon = 0.0,
            max_relative = 1E-13
        );
        assert_relative_eq!(
            compute_integral(4),
            0.07980853550151085,
            epsilon = 0.0,
            max_relative = 1E-13
        );
        assert_relative_eq!(
            compute_integral(8),
            0.07982143859742521,
            epsilon = 0.0,
            max_relative = 1E-13
        );
    }

    #[test]
    fn test_edge_adjacent_triangles() {
        // We create two triangles, the reference triangle
        // (0, 0), (1,0). (0, 1)
        // and the second triangle with coordinates
        // (1,0), (1, 1), (0, 1)
        // We integrate the Green's function against those two triangles.

        // First we need to create the reference map to the second triangle.

        let reference_map =
            |point: (f64, f64)| -> (f64, f64) { (1.0 - point.1, point.0 + point.1) };

        let compute_integral = |npoints: usize| -> f64 {
            let connectivity = CellToCellConnectivity {
                connectivity_dimension: 1,
                local_indices: vec![(1, 0), (2, 2)],
            };

            let singular_rule = triangle_duffy(&connectivity, npoints).unwrap();

            let mut sum = 0.0;

            for index in 0..singular_rule.npoints {
                let (x1, x2) = (
                    singular_rule.test_points[2 * index],
                    singular_rule.test_points[2 * index + 1],
                );

                let (y1, y2) = reference_map((
                    singular_rule.trial_points[2 * index],
                    singular_rule.trial_points[2 * index + 1],
                ));

                let weight = singular_rule.weights[index];
                sum += laplace_green(x1, x2, y1, y2) * weight;
            }
            sum
        };

        // The comparison values are obtained from computations
        // with Bempp-cl
        // Comparisons were also performed with legacy Bempp.
        // Similar results are (order refers to order parameter in
        // legacy Bempp):
        // Order 2: 0.03835750527929082
        // Order 4: 0.038477651910551768
        // Order 8: 0.038478803829805695

        assert_relative_eq!(
            compute_integral(2),
            0.03835750527929083,
            epsilon = 0.0,
            max_relative = 1E-13
        );
        assert_relative_eq!(
            compute_integral(4),
            0.03847765191055147,
            epsilon = 0.0,
            max_relative = 1E-13
        );
        assert_relative_eq!(
            compute_integral(8),
            0.038478803829804516,
            epsilon = 0.0,
            max_relative = 1E-13
        );
    }

    #[test]
    fn test_edge_adjacent_triangles_rotations() {
        let compute_integral = |npoints: usize,
                                cell0_indices: [usize; 3],
                                cell1_indices: [usize; 3]|
         -> f64 {
            let mut local_indices = vec![];
            for (i0, c0) in cell0_indices.iter().enumerate() {
                for (i1, c1) in cell1_indices.iter().enumerate() {
                    if c0 == c1 {
                        local_indices.push((i0, i1));
                    }
                }
            }
            let connectivity = CellToCellConnectivity {
                connectivity_dimension: 1,
                local_indices,
            };

            let singular_rule = triangle_duffy(&connectivity, npoints).unwrap();

            let points = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

            let mut sum = 0.0;

            for index in 0..singular_rule.npoints {
                let (x1, x2) = (
                    points[2 * cell0_indices[0]]
                        + singular_rule.test_points[2 * index]
                            * (points[2 * cell0_indices[1]] - points[2 * cell0_indices[0]])
                        + singular_rule.test_points[2 * index + 1]
                            * (points[2 * cell0_indices[2]] - points[2 * cell0_indices[0]]),
                    points[2 * cell0_indices[0] + 1]
                        + singular_rule.test_points[2 * index]
                            * (points[2 * cell0_indices[1] + 1] - points[2 * cell0_indices[0] + 1])
                        + singular_rule.test_points[2 * index + 1]
                            * (points[2 * cell0_indices[2] + 1] - points[2 * cell0_indices[0] + 1]),
                );

                let (y1, y2) = (
                    points[2 * cell1_indices[0]]
                        + singular_rule.trial_points[2 * index]
                            * (points[2 * cell1_indices[1]] - points[2 * cell1_indices[0]])
                        + singular_rule.trial_points[2 * index + 1]
                            * (points[2 * cell1_indices[2]] - points[2 * cell1_indices[0]]),
                    points[2 * cell1_indices[0] + 1]
                        + singular_rule.trial_points[2 * index]
                            * (points[2 * cell1_indices[1] + 1] - points[2 * cell1_indices[0] + 1])
                        + singular_rule.trial_points[2 * index + 1]
                            * (points[2 * cell1_indices[2] + 1] - points[2 * cell1_indices[0] + 1]),
                );

                let weight = singular_rule.weights[index];
                sum += laplace_green(x1, x2, y1, y2) * weight;
            }
            sum
        };
        for cell0_indices in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            for cell1_indices in [
                [3, 1, 2],
                [3, 2, 1],
                [1, 3, 2],
                [1, 2, 3],
                [2, 3, 1],
                [2, 1, 3],
            ] {
                for npts in [2, 4, 8] {
                    assert_relative_eq!(
                        compute_integral(npts, [0, 1, 2], [3, 1, 2]),
                        compute_integral(npts, cell0_indices, cell1_indices),
                        epsilon = 0.0,
                        max_relative = 1E-13
                    );
                }
            }
        }
    }

    #[test]
    fn test_vertex_adjacent_triangles() {
        // We create two triangles, the reference triangle
        // (0, 0), (1,0). (0, 1)
        // and the second triangle with coordinates
        // (2, 0), (1, 1), (1, 0)
        // We integrate the Green's function against those two triangles.

        // First we need to create the reference map to the second triangle.

        let reference_map =
            |point: (f64, f64)| -> (f64, f64) { (2.0 - point.0 - point.1, point.0) };

        let compute_integral = |npoints: usize| -> f64 {
            let connectivity = CellToCellConnectivity {
                connectivity_dimension: 0,
                local_indices: vec![(1, 2)],
            };

            let singular_rule = triangle_duffy(&connectivity, npoints).unwrap();

            // let rule = simplex_rule(ReferenceCellType::Interval, npoints).unwrap();

            // let singular_rule = vertex_adjacent_triangles(&rule, 1, 2);

            let mut sum = 0.0;

            for index in 0..singular_rule.npoints {
                let (x1, x2) = (
                    singular_rule.test_points[2 * index],
                    singular_rule.test_points[2 * index + 1],
                );

                let (y1, y2) = reference_map((
                    singular_rule.trial_points[2 * index],
                    singular_rule.trial_points[2 * index + 1],
                ));

                let weight = singular_rule.weights[index];
                sum += laplace_green(x1, x2, y1, y2) * weight;
            }
            sum
        };

        // The comparison values are obtained from computations
        // with Bempp-cl
        // Comparisons were also performed with legacy Bempp.
        // Similar results are (order refers to order parameter in
        // legacy Bempp):
        // Order 2: 0.021000886281551452
        // Order 6: 0.021034083055467126
        // Order 14: 0.021034026656569074

        assert_relative_eq!(
            compute_integral(2),
            0.021000886281551452,
            epsilon = 0.0,
            max_relative = 1E-13
        );
        assert_relative_eq!(
            compute_integral(4),
            0.02103408305546695,
            epsilon = 0.0,
            max_relative = 1E-13
        );
        assert_relative_eq!(
            compute_integral(8),
            0.02103402665656858,
            epsilon = 0.0,
            max_relative = 1E-13
        );
    }
}
