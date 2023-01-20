use crate::grid::SerialTriangle3DGrid;
pub use solvers_traits::grid::Geometry;
pub use solvers_traits::grid::Grid;
pub use solvers_traits::grid::Topology;
use std::fs;

// pub fn export_as_gmsh(grid: impl Grid, fname: String) {
pub fn export_as_gmsh(grid: SerialTriangle3DGrid, fname: String) {
    let mut gmsh_s = String::from("");
    gmsh_s.push_str("$MeshFormat\n");
    gmsh_s.push_str("2.2 0 8\n");
    gmsh_s.push_str("$EndMeshFormat\n");
    gmsh_s.push_str("$Nodes\n");
    let node_count = grid.geometry().point_count();
    gmsh_s.push_str(&format!("{node_count}\n"));
    for i in 0..node_count {
        gmsh_s.push_str(&format!("{i}"));
        for j in 0..grid.geometry().dim() {
            let coord = grid.geometry().point(i).unwrap()[j];
            gmsh_s.push_str(&format!(" {coord}"));
        }
        gmsh_s.push_str("\n");
    }
    gmsh_s.push_str("$EndNodes\n");
    gmsh_s.push_str("$Elements\n");
    let cell_count = grid.topology().entity_count(grid.topology().dim());
    gmsh_s.push_str(&format!("{cell_count}\n"));
    for i in 0..cell_count {
        gmsh_s.push_str(&format!("{i} 2 2 0 0")); // TODO: 2 is hardcoded triangle
        for j in 0..3 {
            // currently assumes that Geometry and Topology use the same order
            let vertex = grid.topology().cell(i)[j];
            gmsh_s.push_str(&format!(" {vertex}"))
        }
        gmsh_s.push_str("\n");
    }
    gmsh_s.push_str("$EndElements\n");

    fs::write(fname, gmsh_s).expect("Unable to write file");
}

#[cfg(test)]
mod test {
    use crate::io::*;
    pub use crate::shapes::regular_sphere;

    #[test]
    fn test_gmsh_output() {
        let g = regular_sphere(2);
        export_as_gmsh(g, String::from("test_io_sphere.msh"));
    }
}
