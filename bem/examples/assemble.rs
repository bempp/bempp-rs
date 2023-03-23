use bempp_bem::dofmap::SerialDofMap;
use bempp_element::element::LagrangeElementTriangleDegree1;
use bempp_grid::shapes::regular_sphere;
use bempp_traits::bem::DofMap;
use bempp_traits::grid::{Geometry, Grid, Topology};

fn main() {
    let grid = regular_sphere(0);
    let element = LagrangeElementTriangleDegree1 {};
    let dofmap = SerialDofMap::new(&grid, &element);

    for cell0 in 0..grid.geometry().cell_count() {
        println!("DIAGONAL: {} {}", cell0, cell0);
        println!("  {:?}", dofmap.cell_dofs(cell0).unwrap());
        for cell1 in grid.topology().adjacent_cells(cell0).iter() {
            println!(
                "OFF DIAGONAL: {} {} (connected by {} vertex/vertices)",
                cell0, cell1.0, cell1.1
            );
            println!(
                "  {:?} and {:?}",
                dofmap.cell_dofs(cell0).unwrap(),
                dofmap.cell_dofs(cell1.0).unwrap()
            );
        }
    }
}
