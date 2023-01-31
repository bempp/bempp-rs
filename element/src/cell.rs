//! Cell definitions

pub mod cells_0d;
pub use cells_0d::*;
pub mod cells_1d;
pub use cells_1d::*;
pub mod cells_2d;
pub use cells_2d::*;
pub mod cells_3d;
pub use cells_3d::*;
pub use solvers_traits::cell::PhysicalCell;
pub use solvers_traits::cell::ReferenceCell;
pub use solvers_traits::cell::ReferenceCellType;

#[cfg(test)]
mod test {
    use crate::cell::*;
    use paste::paste;

    macro_rules! test_cell {

        ($($cell:ident),+) => {

        $(
            paste! {

                #[test]
                fn [<test_ $cell:lower>]() {
                    let c = [<$cell>] {};
                    assert_eq!(c.cell_type(), ReferenceCellType::[<$cell>]);
                    assert_eq!(c.label(), stringify!([<$cell:lower>]));
                    assert_eq!(c.vertex_count() as usize, c.vertices().len() / (c.dim() as usize));
                    assert_eq!(c.edge_count() as usize, c.edges().len() / 2);
                    assert_eq!(c.face_count() as usize, c.faces_nvertices().len());

                    for v_n in 0..c.vertex_count() {
                        let v = c.connectivity(0, v_n, 0).unwrap();
                        assert_eq!(v[0], v_n);
                    }
                    for e_n in 0..c.edge_count() {
                        let v = c.connectivity(1, e_n, 0).unwrap();
                        let edge = &c.edges()[2 * (e_n as usize)..2 * ((e_n as usize) + 1)];
                        assert_eq!(v, edge);
                    }
                    let mut start = 0;
                    for f_n in 0..c.face_count() {
                        let v = c.connectivity(2, f_n, 0).unwrap();
                        let face = &c.faces()[start..start + (c.faces_nvertices()[f_n as usize] as usize)];
                        assert_eq!(v, face);
                        start += (c.faces_nvertices()[f_n as usize] as usize);
                    }

                    for e_dim in 0..c.dim() + 1 {
                        for e_n in 0..c.entity_count(e_dim).unwrap() {
                            let e_vertices = c.connectivity(e_dim, e_n, 0).unwrap();
                            for c_dim in 0..c.dim() + 1 {
                                let connectivity = c.connectivity(e_dim, e_n, c_dim).unwrap();
                                if e_dim == c_dim {
                                    assert_eq!(connectivity.len(), 1);
                                    assert_eq!(connectivity[0], e_n)
                                } else {
                                    for c_n in &connectivity {
                                        let c_vertices = c.connectivity(c_dim, *c_n, 0).unwrap();
                                        println!("{} {} {} {}", e_dim, e_n, c_dim, c_n);
                                        if e_dim < c_dim {
                                            for i in &e_vertices {
                                                println!(" c {}", i);
                                                assert!(c_vertices.contains(&i));
                                            }
                                        } else {
                                            for i in &c_vertices {
                                                println!(" e {}", i);
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
