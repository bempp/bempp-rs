use bempp_bem::dofmap::SerialDofMap;
use bempp_element::element::LagrangeElementTriangleDegree0;
use bempp_grid::shapes::regular_sphere;
use bempp_traits::bem::DofMap;
use bempp_traits::grid::{Geometry, Grid, Topology};
use bempp_quadrature::*;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::types::CellToCellConnectivity;

fn laplace_green(x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    let inv_dist = 1.0 / f64::sqrt((x1 - y1) * (x1 - y1) + (x2 - y2) * (x2 - y2));

    0.25 * std::f64::consts::FRAC_1_PI * inv_dist
}

fn main() {
    let grid = regular_sphere(0);
    let element = LagrangeElementTriangleDegree0 {};
    let dofmap = SerialDofMap::new(&grid, &element);

    for cell0 in 0..grid.geometry().cell_count() {
        //println!("DIAGONAL: {} {}", cell0, cell0);
        //println!("  {:?}", dofmap.cell_dofs(cell0).unwrap());
        println!("{:?}", grid.geometry().cell_vertices(cell0).unwrap());
        println!("{:?}", grid.geometry().point(grid.geometry().cell_vertices(cell0).unwrap()[0]).unwrap());

        grid.geometry().integration_element();

        let npoints = 3;

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
            
            println!("{}", sum);
        //for cell1 in grid.topology().adjacent_cells(cell0).iter() {
            //println!(
            //    "OFF DIAGONAL: {} {} (connected by {} vertex/vertices)",
            //    cell0, cell1.0, cell1.1
            //);
            //println!(
            //    "  {:?} and {:?}",
            //    dofmap.cell_dofs(cell0).unwrap(),
            //    dofmap.cell_dofs(cell1.0).unwrap()
            //);
        //}
    }
}
