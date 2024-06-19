//! Cell definitions

use crate::traits::types::ReferenceCell;
use rlst::RlstScalar;

/// The topological dimension of the cell
pub fn dim(cell: ReferenceCell) -> usize {
    match cell {
        ReferenceCell::Point => 0,
        ReferenceCell::Interval => 1,
        ReferenceCell::Triangle => 2,
        ReferenceCell::Quadrilateral => 2,
        ReferenceCell::Tetrahedron => 3,
        ReferenceCell::Hexahedron => 3,
        ReferenceCell::Prism => 3,
        ReferenceCell::Pyramid => 3,
    }
}
/// Is the cell a simplex?
pub fn is_simplex(cell: ReferenceCell) -> bool {
    match cell {
        ReferenceCell::Point => true,
        ReferenceCell::Interval => true,
        ReferenceCell::Triangle => true,
        ReferenceCell::Quadrilateral => false,
        ReferenceCell::Tetrahedron => true,
        ReferenceCell::Hexahedron => false,
        ReferenceCell::Prism => false,
        ReferenceCell::Pyramid => false,
    }
}

/// The vertices of the reference cell
pub fn vertices<T: RlstScalar<Real = T>>(cell: ReferenceCell) -> Vec<Vec<T>> {
    let zero = T::from(0.0).unwrap();
    let one = T::from(1.0).unwrap();
    match cell {
        ReferenceCell::Point => vec![],
        ReferenceCell::Interval => vec![vec![zero], vec![one]],
        ReferenceCell::Triangle => vec![vec![zero, zero], vec![one, zero], vec![zero, one]],
        ReferenceCell::Quadrilateral => vec![
            vec![zero, zero],
            vec![one, zero],
            vec![zero, one],
            vec![one, one],
        ],
        ReferenceCell::Tetrahedron => vec![
            vec![zero, zero, zero],
            vec![one, zero, zero],
            vec![zero, one, zero],
            vec![zero, zero, one],
        ],
        ReferenceCell::Hexahedron => vec![
            vec![zero, zero, zero],
            vec![one, zero, zero],
            vec![zero, one, zero],
            vec![one, one, zero],
            vec![zero, zero, one],
            vec![one, zero, one],
            vec![zero, one, one],
            vec![one, one, one],
        ],
        ReferenceCell::Prism => vec![
            vec![zero, zero, zero],
            vec![one, zero, zero],
            vec![zero, one, zero],
            vec![zero, zero, one],
            vec![one, zero, one],
            vec![zero, one, one],
        ],
        ReferenceCell::Pyramid => vec![
            vec![zero, zero, zero],
            vec![one, zero, zero],
            vec![zero, one, zero],
            vec![one, one, zero],
            vec![zero, zero, one],
        ],
    }
}

/// The midpoint of the cell
pub fn midpoint<T: RlstScalar<Real = T>>(cell: ReferenceCell) -> Vec<T> {
    let half = T::from(0.5).unwrap();
    let third = T::from(1.0).unwrap() / T::from(3.0).unwrap();
    match cell {
        ReferenceCell::Point => vec![],
        ReferenceCell::Interval => vec![half],
        ReferenceCell::Triangle => vec![third; 2],
        ReferenceCell::Quadrilateral => vec![half; 2],
        ReferenceCell::Tetrahedron => vec![T::from(1.0).unwrap() / T::from(6.0).unwrap(); 3],
        ReferenceCell::Hexahedron => vec![half; 3],
        ReferenceCell::Prism => vec![third, third, half],
        ReferenceCell::Pyramid => vec![
            T::from(0.4).unwrap(),
            T::from(0.4).unwrap(),
            T::from(0.2).unwrap(),
        ],
    }
}

/// The edges of the reference cell
pub fn edges(cell: ReferenceCell) -> Vec<Vec<usize>> {
    match cell {
        ReferenceCell::Point => vec![],
        ReferenceCell::Interval => vec![vec![0, 1]],
        ReferenceCell::Triangle => vec![vec![1, 2], vec![0, 2], vec![0, 1]],
        ReferenceCell::Quadrilateral => vec![vec![0, 1], vec![0, 2], vec![1, 3], vec![2, 3]],
        ReferenceCell::Tetrahedron => vec![
            vec![2, 3],
            vec![1, 3],
            vec![1, 2],
            vec![0, 3],
            vec![0, 2],
            vec![0, 1],
        ],
        ReferenceCell::Hexahedron => vec![
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
        ReferenceCell::Prism => vec![
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
        ReferenceCell::Pyramid => vec![
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
pub fn faces(cell: ReferenceCell) -> Vec<Vec<usize>> {
    match cell {
        ReferenceCell::Point => vec![],
        ReferenceCell::Interval => vec![],
        ReferenceCell::Triangle => vec![vec![0, 1, 2]],
        ReferenceCell::Quadrilateral => vec![vec![0, 1, 2, 3]],
        ReferenceCell::Tetrahedron => {
            vec![vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2]]
        }
        ReferenceCell::Hexahedron => vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 4, 5],
            vec![0, 2, 4, 6],
            vec![1, 3, 5, 7],
            vec![2, 3, 6, 7],
            vec![4, 5, 6, 7],
        ],
        ReferenceCell::Prism => vec![
            vec![0, 1, 2],
            vec![0, 1, 3, 4],
            vec![0, 2, 3, 5],
            vec![1, 2, 4, 5],
            vec![3, 4, 5],
        ],
        ReferenceCell::Pyramid => vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 4],
            vec![0, 2, 4],
            vec![1, 3, 4],
            vec![2, 3, 4],
        ],
    }
}

/// The types of the subentities of the reference cell
pub fn entity_types(cell: ReferenceCell) -> Vec<Vec<ReferenceCell>> {
    match cell {
        ReferenceCell::Point => vec![vec![ReferenceCell::Point], vec![], vec![], vec![]],
        ReferenceCell::Interval => vec![
            vec![ReferenceCell::Point; 2],
            vec![ReferenceCell::Interval],
            vec![],
            vec![],
        ],
        ReferenceCell::Triangle => vec![
            vec![ReferenceCell::Point; 3],
            vec![ReferenceCell::Interval; 3],
            vec![ReferenceCell::Triangle],
            vec![],
        ],
        ReferenceCell::Quadrilateral => vec![
            vec![ReferenceCell::Point; 4],
            vec![ReferenceCell::Interval; 4],
            vec![ReferenceCell::Quadrilateral],
            vec![],
        ],
        ReferenceCell::Tetrahedron => vec![
            vec![ReferenceCell::Point; 4],
            vec![ReferenceCell::Interval; 6],
            vec![ReferenceCell::Triangle; 4],
            vec![ReferenceCell::Tetrahedron],
        ],
        ReferenceCell::Hexahedron => vec![
            vec![ReferenceCell::Point; 8],
            vec![ReferenceCell::Interval; 12],
            vec![ReferenceCell::Quadrilateral; 6],
            vec![ReferenceCell::Hexahedron],
        ],
        ReferenceCell::Prism => vec![
            vec![ReferenceCell::Point; 6],
            vec![ReferenceCell::Interval; 9],
            vec![
                ReferenceCell::Triangle,
                ReferenceCell::Quadrilateral,
                ReferenceCell::Quadrilateral,
                ReferenceCell::Quadrilateral,
                ReferenceCell::Triangle,
            ],
            vec![ReferenceCell::Prism],
        ],
        ReferenceCell::Pyramid => vec![
            vec![ReferenceCell::Point; 5],
            vec![ReferenceCell::Interval; 8],
            vec![
                ReferenceCell::Quadrilateral,
                ReferenceCell::Triangle,
                ReferenceCell::Triangle,
                ReferenceCell::Triangle,
                ReferenceCell::Triangle,
            ],
            vec![ReferenceCell::Pyramid],
        ],
    }
}

/// The number of subentities of each dimension
pub fn entity_counts(cell: ReferenceCell) -> Vec<usize> {
    match cell {
        ReferenceCell::Point => vec![1, 0, 0, 0],
        ReferenceCell::Interval => vec![2, 1, 0, 0],
        ReferenceCell::Triangle => vec![3, 3, 1, 0],
        ReferenceCell::Quadrilateral => vec![4, 4, 1, 0],
        ReferenceCell::Tetrahedron => vec![4, 6, 4, 1],
        ReferenceCell::Hexahedron => vec![8, 12, 6, 1],
        ReferenceCell::Prism => vec![6, 9, 5, 1],
        ReferenceCell::Pyramid => vec![5, 8, 5, 1],
    }
}

/// The connectivity of the reference cell
///
/// The indices of the result are \[i\]\[j\]\[k\]\[l\]
pub fn connectivity(cell: ReferenceCell) -> Vec<Vec<Vec<Vec<usize>>>> {
    match cell {
        ReferenceCell::Point => vec![vec![vec![vec![0]]]],
        ReferenceCell::Interval => vec![
            vec![vec![vec![0], vec![0]], vec![vec![1], vec![0]]],
            vec![vec![vec![0, 1], vec![0]]],
        ],
        ReferenceCell::Triangle => vec![
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
        ReferenceCell::Quadrilateral => vec![
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
        ReferenceCell::Tetrahedron => vec![
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
        ReferenceCell::Hexahedron => vec![
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
        ReferenceCell::Prism => vec![
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
        ReferenceCell::Pyramid => vec![
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
