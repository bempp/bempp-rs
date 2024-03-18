//! Cell definitions

use bempp_traits::types::ReferenceCellType;
use rlst::RlstScalar;

/// The topological dimension of the cell
pub fn dim(cell: ReferenceCellType) -> usize {
    match cell {
        ReferenceCellType::Point => 0,
        ReferenceCellType::Interval => 1,
        ReferenceCellType::Triangle => 2,
        ReferenceCellType::Quadrilateral => 2,
        ReferenceCellType::Tetrahedron => 3,
        ReferenceCellType::Hexahedron => 3,
        ReferenceCellType::Prism => 3,
        ReferenceCellType::Pyramid => 3,
    }
}
/// Is the cell a simplex?
pub fn is_simplex(cell: ReferenceCellType) -> bool {
    match cell {
        ReferenceCellType::Point => true,
        ReferenceCellType::Interval => true,
        ReferenceCellType::Triangle => true,
        ReferenceCellType::Quadrilateral => false,
        ReferenceCellType::Tetrahedron => true,
        ReferenceCellType::Hexahedron => false,
        ReferenceCellType::Prism => false,
        ReferenceCellType::Pyramid => false,
    }
}

/// The vertices of the reference cell
pub fn vertices<T: RlstScalar<Real = T>>(cell: ReferenceCellType) -> Vec<Vec<T>> {
    let zero = T::from(0.0).unwrap();
    let one = T::from(1.0).unwrap();
    match cell {
        ReferenceCellType::Point => vec![],
        ReferenceCellType::Interval => vec![vec![zero], vec![one]],
        ReferenceCellType::Triangle => vec![vec![zero, zero], vec![one, zero], vec![zero, one]],
        ReferenceCellType::Quadrilateral => vec![
            vec![zero, zero],
            vec![one, zero],
            vec![zero, one],
            vec![one, one],
        ],
        ReferenceCellType::Tetrahedron => vec![
            vec![zero, zero, zero],
            vec![one, zero, zero],
            vec![zero, one, zero],
            vec![zero, zero, one],
        ],
        ReferenceCellType::Hexahedron => vec![
            vec![zero, zero, zero],
            vec![one, zero, zero],
            vec![zero, one, zero],
            vec![one, one, zero],
            vec![zero, zero, one],
            vec![one, zero, one],
            vec![zero, one, one],
            vec![one, one, one],
        ],
        ReferenceCellType::Prism => vec![
            vec![zero, zero, zero],
            vec![one, zero, zero],
            vec![zero, one, zero],
            vec![zero, zero, one],
            vec![one, zero, one],
            vec![zero, one, one],
        ],
        ReferenceCellType::Pyramid => vec![
            vec![zero, zero, zero],
            vec![one, zero, zero],
            vec![zero, one, zero],
            vec![one, one, zero],
            vec![zero, zero, one],
        ],
    }
}

/// The midpoint of the cell
pub fn midpoint<T: RlstScalar<Real = T>>(cell: ReferenceCellType) -> Vec<T> {
    let half = T::from(0.5).unwrap();
    let third = T::from(1.0).unwrap() / T::from(3.0).unwrap();
    match cell {
        ReferenceCellType::Point => vec![],
        ReferenceCellType::Interval => vec![half],
        ReferenceCellType::Triangle => vec![third; 2],
        ReferenceCellType::Quadrilateral => vec![half; 2],
        ReferenceCellType::Tetrahedron => vec![T::from(1.0).unwrap() / T::from(6.0).unwrap(); 3],
        ReferenceCellType::Hexahedron => vec![half; 3],
        ReferenceCellType::Prism => vec![third, third, half],
        ReferenceCellType::Pyramid => vec![
            T::from(0.4).unwrap(),
            T::from(0.4).unwrap(),
            T::from(0.2).unwrap(),
        ],
    }
}

/// The edges of the reference cell
pub fn edges(cell: ReferenceCellType) -> Vec<Vec<usize>> {
    match cell {
        ReferenceCellType::Point => vec![],
        ReferenceCellType::Interval => vec![vec![0, 1]],
        ReferenceCellType::Triangle => vec![vec![1, 2], vec![0, 2], vec![0, 1]],
        ReferenceCellType::Quadrilateral => vec![vec![0, 1], vec![0, 2], vec![1, 3], vec![2, 3]],
        ReferenceCellType::Tetrahedron => vec![
            vec![2, 3],
            vec![1, 3],
            vec![1, 2],
            vec![0, 3],
            vec![0, 2],
            vec![0, 1],
        ],
        ReferenceCellType::Hexahedron => vec![
            vec![0, 1],
            vec![0, 2],
            vec![0, 4],
            vec![1, 3],
            vec![1, 5],
            vec![2, 3],
            vec![2, 6],
            vec![3, 7],
            vec![4, 5],
            vec![4, 6],
            vec![5, 7],
            vec![6, 7],
        ],
        ReferenceCellType::Prism => vec![
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 2],
            vec![1, 4],
            vec![2, 5],
            vec![3, 4],
            vec![3, 5],
            vec![4, 5],
        ],
        ReferenceCellType::Pyramid => vec![
            vec![0, 1],
            vec![0, 2],
            vec![0, 4],
            vec![1, 3],
            vec![1, 4],
            vec![2, 3],
            vec![2, 4],
            vec![3, 4],
        ],
    }
}

/// The faces of the reference cell
pub fn faces(cell: ReferenceCellType) -> Vec<Vec<usize>> {
    match cell {
        ReferenceCellType::Point => vec![],
        ReferenceCellType::Interval => vec![],
        ReferenceCellType::Triangle => vec![vec![0, 1, 2]],
        ReferenceCellType::Quadrilateral => vec![vec![0, 1, 2, 3]],
        ReferenceCellType::Tetrahedron => {
            vec![vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2]]
        }
        ReferenceCellType::Hexahedron => vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 4, 5],
            vec![0, 2, 4, 6],
            vec![1, 3, 5, 7],
            vec![2, 3, 6, 7],
            vec![4, 5, 6, 7],
        ],
        ReferenceCellType::Prism => vec![
            vec![0, 1, 2],
            vec![0, 1, 3, 4],
            vec![0, 2, 3, 5],
            vec![1, 2, 4, 5],
            vec![3, 4, 5],
        ],
        ReferenceCellType::Pyramid => vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 4],
            vec![0, 2, 4],
            vec![1, 3, 4],
            vec![2, 3, 4],
        ],
    }
}

/// The types of the subentities of the reference cell
pub fn entity_types(cell: ReferenceCellType) -> Vec<Vec<ReferenceCellType>> {
    match cell {
        ReferenceCellType::Point => vec![vec![ReferenceCellType::Point], vec![], vec![], vec![]],
        ReferenceCellType::Interval => vec![
            vec![ReferenceCellType::Point; 2],
            vec![ReferenceCellType::Interval],
            vec![],
            vec![],
        ],
        ReferenceCellType::Triangle => vec![
            vec![ReferenceCellType::Point; 3],
            vec![ReferenceCellType::Interval; 3],
            vec![ReferenceCellType::Triangle],
            vec![],
        ],
        ReferenceCellType::Quadrilateral => vec![
            vec![ReferenceCellType::Point; 4],
            vec![ReferenceCellType::Interval; 4],
            vec![ReferenceCellType::Quadrilateral],
            vec![],
        ],
        ReferenceCellType::Tetrahedron => vec![
            vec![ReferenceCellType::Point; 4],
            vec![ReferenceCellType::Interval; 6],
            vec![ReferenceCellType::Triangle; 4],
            vec![ReferenceCellType::Tetrahedron],
        ],
        ReferenceCellType::Hexahedron => vec![
            vec![ReferenceCellType::Point; 8],
            vec![ReferenceCellType::Interval; 12],
            vec![ReferenceCellType::Quadrilateral; 6],
            vec![ReferenceCellType::Hexahedron],
        ],
        ReferenceCellType::Prism => vec![
            vec![ReferenceCellType::Point; 6],
            vec![ReferenceCellType::Interval; 9],
            vec![
                ReferenceCellType::Triangle,
                ReferenceCellType::Quadrilateral,
                ReferenceCellType::Quadrilateral,
                ReferenceCellType::Quadrilateral,
                ReferenceCellType::Triangle,
            ],
            vec![ReferenceCellType::Prism],
        ],
        ReferenceCellType::Pyramid => vec![
            vec![ReferenceCellType::Point; 5],
            vec![ReferenceCellType::Interval; 8],
            vec![
                ReferenceCellType::Quadrilateral,
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
            ],
            vec![ReferenceCellType::Pyramid],
        ],
    }
}

/// The number of subentities of each dimension
pub fn entity_counts(cell: ReferenceCellType) -> Vec<usize> {
    match cell {
        ReferenceCellType::Point => vec![1, 0, 0, 0],
        ReferenceCellType::Interval => vec![2, 1, 0, 0],
        ReferenceCellType::Triangle => vec![3, 3, 1, 0],
        ReferenceCellType::Quadrilateral => vec![4, 4, 1, 0],
        ReferenceCellType::Tetrahedron => vec![4, 6, 4, 1],
        ReferenceCellType::Hexahedron => vec![8, 12, 6, 1],
        ReferenceCellType::Prism => vec![6, 9, 5, 1],
        ReferenceCellType::Pyramid => vec![5, 8, 5, 1],
    }
}

/// The connectivity of the reference cell
///
/// The indices of the result are \[i\]\[j\]\[k\]\[l\]
pub fn connectivity(cell: ReferenceCellType) -> Vec<Vec<Vec<Vec<usize>>>> {
    match cell {
        ReferenceCellType::Point => vec![vec![vec![vec![0]]]],
        ReferenceCellType::Interval => vec![
            vec![vec![vec![0], vec![0]], vec![vec![1], vec![0]]],
            vec![vec![vec![0, 1], vec![0]]],
        ],
        ReferenceCellType::Triangle => vec![
            vec![
                vec![vec![0], vec![1, 2], vec![0]],
                vec![vec![1], vec![0, 2], vec![0]],
                vec![vec![2], vec![0, 1], vec![0]],
            ],
            vec![
                vec![vec![1, 2], vec![0], vec![0]],
                vec![vec![0, 2], vec![1], vec![0]],
                vec![vec![0, 1], vec![2], vec![0]],
            ],
            vec![vec![vec![0, 1, 2], vec![0, 1, 2], vec![0]]],
        ],
        ReferenceCellType::Quadrilateral => vec![
            vec![
                vec![vec![0], vec![0, 1], vec![0]],
                vec![vec![1], vec![0, 2], vec![0]],
                vec![vec![2], vec![1, 3], vec![0]],
                vec![vec![3], vec![2, 3], vec![0]],
            ],
            vec![
                vec![vec![0, 1], vec![0], vec![0]],
                vec![vec![0, 2], vec![1], vec![0]],
                vec![vec![1, 3], vec![2], vec![0]],
                vec![vec![2, 3], vec![3], vec![0]],
            ],
            vec![vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3], vec![0]]],
        ],
        ReferenceCellType::Tetrahedron => vec![
            vec![
                vec![vec![0], vec![3, 4, 5], vec![1, 2, 3], vec![0]],
                vec![vec![1], vec![1, 2, 5], vec![0, 2, 3], vec![0]],
                vec![vec![2], vec![0, 2, 4], vec![0, 1, 3], vec![0]],
                vec![vec![3], vec![0, 1, 3], vec![0, 1, 2], vec![0]],
            ],
            vec![
                vec![vec![2, 3], vec![0], vec![0, 1], vec![0]],
                vec![vec![1, 3], vec![1], vec![0, 2], vec![0]],
                vec![vec![1, 2], vec![2], vec![0, 3], vec![0]],
                vec![vec![0, 3], vec![3], vec![1, 2], vec![0]],
                vec![vec![0, 2], vec![4], vec![1, 3], vec![0]],
                vec![vec![0, 1], vec![5], vec![2, 3], vec![0]],
            ],
            vec![
                vec![vec![1, 2, 3], vec![0, 1, 2], vec![0], vec![0]],
                vec![vec![0, 2, 3], vec![0, 3, 4], vec![1], vec![0]],
                vec![vec![0, 1, 3], vec![1, 3, 5], vec![2], vec![0]],
                vec![vec![0, 1, 2], vec![2, 4, 5], vec![3], vec![0]],
            ],
            vec![vec![
                vec![0, 1, 2, 3],
                vec![0, 1, 2, 3, 4, 5],
                vec![0, 1, 2, 3],
                vec![0],
            ]],
        ],
        ReferenceCellType::Hexahedron => vec![
            vec![
                vec![vec![0], vec![0, 1, 2], vec![0, 1, 2], vec![0]],
                vec![vec![1], vec![0, 3, 4], vec![0, 1, 3], vec![0]],
                vec![vec![2], vec![1, 5, 6], vec![0, 2, 4], vec![0]],
                vec![vec![3], vec![3, 5, 7], vec![0, 3, 4], vec![0]],
                vec![vec![4], vec![2, 8, 9], vec![1, 2, 5], vec![0]],
                vec![vec![5], vec![4, 8, 10], vec![1, 3, 5], vec![0]],
                vec![vec![6], vec![6, 9, 11], vec![2, 4, 5], vec![0]],
                vec![vec![7], vec![7, 10, 11], vec![3, 4, 5], vec![0]],
            ],
            vec![
                vec![vec![0, 1], vec![0], vec![0, 1], vec![0]],
                vec![vec![0, 2], vec![1], vec![0, 2], vec![0]],
                vec![vec![0, 4], vec![2], vec![1, 2], vec![0]],
                vec![vec![1, 3], vec![3], vec![0, 3], vec![0]],
                vec![vec![1, 5], vec![4], vec![1, 3], vec![0]],
                vec![vec![2, 3], vec![5], vec![0, 4], vec![0]],
                vec![vec![2, 6], vec![6], vec![2, 4], vec![0]],
                vec![vec![3, 7], vec![7], vec![3, 4], vec![0]],
                vec![vec![4, 5], vec![8], vec![1, 5], vec![0]],
                vec![vec![4, 6], vec![9], vec![2, 5], vec![0]],
                vec![vec![5, 7], vec![10], vec![3, 5], vec![0]],
                vec![vec![6, 7], vec![11], vec![4, 5], vec![0]],
            ],
            vec![
                vec![vec![0, 1, 2, 3], vec![0, 1, 3, 5], vec![0], vec![0]],
                vec![vec![0, 1, 4, 5], vec![0, 2, 4, 8], vec![1], vec![0]],
                vec![vec![0, 2, 4, 6], vec![1, 2, 6, 9], vec![2], vec![0]],
                vec![vec![1, 3, 5, 7], vec![3, 4, 7, 10], vec![3], vec![0]],
                vec![vec![2, 3, 6, 7], vec![5, 6, 7, 11], vec![4], vec![0]],
                vec![vec![4, 5, 6, 7], vec![8, 9, 10, 11], vec![5], vec![0]],
            ],
            vec![vec![
                vec![0, 1, 2, 3, 4, 5, 6, 7],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                vec![0, 1, 2, 3, 4, 5],
                vec![0],
            ]],
        ],
        ReferenceCellType::Prism => vec![
            vec![
                vec![vec![0], vec![0, 1, 2], vec![0, 1, 2], vec![0]],
                vec![vec![1], vec![0, 3, 4], vec![0, 1, 3], vec![0]],
                vec![vec![2], vec![1, 3, 5], vec![0, 2, 3], vec![0]],
                vec![vec![3], vec![2, 6, 7], vec![1, 2, 4], vec![0]],
                vec![vec![4], vec![4, 6, 8], vec![1, 3, 4], vec![0]],
                vec![vec![5], vec![5, 7, 8], vec![2, 3, 4], vec![0]],
            ],
            vec![
                vec![vec![0, 1], vec![0], vec![0, 1], vec![0]],
                vec![vec![0, 2], vec![1], vec![0, 2], vec![0]],
                vec![vec![0, 3], vec![2], vec![1, 2], vec![0]],
                vec![vec![1, 2], vec![3], vec![0, 3], vec![0]],
                vec![vec![1, 4], vec![4], vec![1, 3], vec![0]],
                vec![vec![2, 5], vec![5], vec![2, 3], vec![0]],
                vec![vec![3, 4], vec![6], vec![1, 4], vec![0]],
                vec![vec![3, 5], vec![7], vec![2, 4], vec![0]],
                vec![vec![4, 5], vec![8], vec![3, 4], vec![0]],
            ],
            vec![
                vec![vec![0, 1, 2], vec![0, 1, 3], vec![0], vec![0]],
                vec![vec![0, 1, 3, 4], vec![0, 2, 4, 6], vec![1], vec![0]],
                vec![vec![0, 2, 3, 5], vec![1, 2, 5, 7], vec![2], vec![0]],
                vec![vec![1, 2, 4, 5], vec![3, 4, 5, 8], vec![3], vec![0]],
                vec![vec![3, 4, 5], vec![6, 7, 8], vec![4], vec![0]],
            ],
            vec![vec![
                vec![0, 1, 2, 3, 4, 5],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
                vec![0, 1, 2, 3, 4],
                vec![0],
            ]],
        ],
        ReferenceCellType::Pyramid => vec![
            vec![
                vec![vec![0], vec![0, 1, 2], vec![0, 1, 2], vec![0]],
                vec![vec![1], vec![0, 3, 4], vec![0, 1, 3], vec![0]],
                vec![vec![2], vec![1, 5, 6], vec![0, 2, 4], vec![0]],
                vec![vec![3], vec![3, 5, 7], vec![0, 3, 4], vec![0]],
                vec![vec![4], vec![2, 4, 6, 7], vec![1, 2, 3, 4], vec![0]],
            ],
            vec![
                vec![vec![0, 1], vec![0], vec![0, 1], vec![0]],
                vec![vec![0, 2], vec![1], vec![0, 2], vec![0]],
                vec![vec![0, 4], vec![2], vec![1, 2], vec![0]],
                vec![vec![1, 3], vec![3], vec![0, 3], vec![0]],
                vec![vec![1, 4], vec![4], vec![1, 3], vec![0]],
                vec![vec![2, 3], vec![5], vec![0, 4], vec![0]],
                vec![vec![2, 4], vec![6], vec![2, 4], vec![0]],
                vec![vec![3, 4], vec![7], vec![3, 4], vec![0]],
            ],
            vec![
                vec![vec![0, 1, 2, 3], vec![0, 1, 3, 5], vec![0], vec![0]],
                vec![vec![0, 1, 4], vec![0, 2, 4], vec![1], vec![0]],
                vec![vec![0, 2, 4], vec![1, 2, 6], vec![2], vec![0]],
                vec![vec![1, 3, 4], vec![3, 4, 7], vec![3], vec![0]],
                vec![vec![2, 3, 4], vec![5, 6, 7], vec![4], vec![0]],
            ],
            vec![vec![
                vec![0, 1, 2, 3, 4],
                vec![0, 1, 2, 3, 4, 5, 6, 7],
                vec![0, 1, 2, 3, 4],
                vec![0],
            ]],
        ],
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use paste::paste;

    macro_rules! test_cell {

        ($($cell:ident),+) => {

        $(
            paste! {

                #[test]
                fn [<test_ $cell:lower>]() {
                    let v = vertices::<f64>(ReferenceCellType::[<$cell>]);
                    let d = dim(ReferenceCellType::[<$cell>]);
                    let ec = entity_counts(ReferenceCellType::[<$cell>]);
                    let et = entity_types(ReferenceCellType::[<$cell>]);
                    let conn = connectivity(ReferenceCellType::[<$cell>]);
                    for i in 0..d+1 {
                        assert_eq!(ec[i], et[i].len());
                        assert_eq!(ec[i], conn[i].len());
                    }
                    assert_eq!(ec[0], v.len());
                    for i in v {
                        assert_eq!(i.len(), d);
                    }

                    for v_n in 0..ec[0] {
                        let v = &conn[0][v_n][0];
                        assert_eq!(v, &[v_n]);
                    }
                    for e_n in 0..ec[1] {
                        let vs = &conn[1][e_n][0];
                        assert_eq!(vs, &edges(ReferenceCellType::[<$cell>])[e_n]);
                    }
                    for f_n in 0..ec[2] {
                        let vs = &conn[2][f_n][0];
                        assert_eq!(vs, &faces(ReferenceCellType::[<$cell>])[f_n]);
                    }

                    for e_dim in 0..d {
                        for e_n in 0..ec[e_dim] {
                            let e_vertices = &conn[e_dim][e_n][0];
                            for c_dim in 0..d + 1 {
                                let connectivity = &conn[e_dim][e_n][c_dim];
                                if e_dim == c_dim {
                                    assert_eq!(connectivity.len(), 1);
                                    assert_eq!(connectivity[0], e_n)
                                } else {
                                    for c_n in connectivity {
                                        let c_vertices = &conn[c_dim][*c_n][0];
                                        if e_dim < c_dim {
                                            for i in e_vertices {
                                                assert!(c_vertices.contains(&i));
                                            }
                                        } else {
                                            for i in c_vertices {
                                                assert!(e_vertices.contains(&i));
                                            }
                                        }
                                        assert!(connectivity.contains(&c_n));
                                    }
                                }
                            }
                        }
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
