use bempp_bem::dofmap::SerialDofMap;
use bempp_element::element::LagrangeElementTriangleDegree0;
use bempp_grid::shapes::regular_sphere;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::types::CellToCellConnectivity;
use bempp_tools::arrays::Array2D;
use bempp_traits::bem::DofMap;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};

fn laplace_green(x1: f64, x2: f64, x3: f64, y1: f64, y2: f64, y3: f64) -> f64 {
    let inv_dist =
        1.0 / f64::sqrt((x1 - y1) * (x1 - y1) + (x2 - y2) * (x2 - y2) + (x3 - y3) * (x3 - y3));

    0.25 * std::f64::consts::FRAC_1_PI * inv_dist
}

fn main() {
    let grid = regular_sphere(0);
    let element = LagrangeElementTriangleDegree0 {};
    let dofmap = SerialDofMap::new(&grid, &element);

    let npoints = 4;
    let same_triangle_rule = triangle_duffy(
        &CellToCellConnectivity {
            connectivity_dimension: 2,
            local_indices: Vec::new(),
        },
        npoints,
    )
    .unwrap();

    let test_points = Array2D::from_data(
        same_triangle_rule.test_points,
        (same_triangle_rule.npoints, 2),
    );
    let trial_points = Array2D::from_data(
        same_triangle_rule.trial_points,
        (same_triangle_rule.npoints, 2),
    );
    let mut test_table = element.create_tabulate_array(0, same_triangle_rule.npoints);
    let mut trial_table = element.create_tabulate_array(0, same_triangle_rule.npoints);

    element.tabulate(&test_points, 0, &mut test_table);
    element.tabulate(&trial_points, 0, &mut trial_table);

    // Assign working memory
    let mut pts = Array2D::<f64>::new((2, 2));
    let mut mapped_pts = Array2D::<f64>::new((2, 3));
    let mut test_jdet = vec![0.0; same_triangle_rule.npoints];
    let mut trial_jdet = vec![0.0; same_triangle_rule.npoints];

    let mut matrix = Array2D::<f64>::new((dofmap.global_size(), dofmap.global_size()));

    for cell0 in 0..grid.geometry().cell_count() {
        grid.geometry()
            .compute_jacobian_determinants(&test_points, cell0, &mut test_jdet);
        grid.geometry()
            .compute_jacobian_determinants(&trial_points, cell0, &mut trial_jdet);

        for (test_i, test_dof) in dofmap.cell_dofs(cell0).unwrap().iter().enumerate() {
            for (trial_i, trial_dof) in dofmap.cell_dofs(cell0).unwrap().iter().enumerate() {
                let mut sum = 0.0;

                for index in 0..same_triangle_rule.npoints {
                    unsafe {
                        *pts.get_unchecked_mut(0, 0) = *test_points.get_unchecked(index, 0);
                        *pts.get_unchecked_mut(0, 1) = *test_points.get_unchecked(index, 1);
                        *pts.get_unchecked_mut(1, 0) = *trial_points.get_unchecked(index, 0);
                        *pts.get_unchecked_mut(1, 1) = *trial_points.get_unchecked(index, 1);
                    }
                    grid.geometry().compute_points(&pts, cell0, &mut mapped_pts);
                    let weight = same_triangle_rule.weights[index];

                    sum += laplace_green(
                        unsafe { *mapped_pts.get_unchecked(0, 0) },
                        unsafe { *mapped_pts.get_unchecked(0, 1) },
                        unsafe { *mapped_pts.get_unchecked(0, 2) },
                        unsafe { *mapped_pts.get_unchecked(1, 0) },
                        unsafe { *mapped_pts.get_unchecked(1, 1) },
                        unsafe { *mapped_pts.get_unchecked(1, 2) },
                    ) * weight
                        * unsafe { test_table.get_unchecked(0, index, test_i, 0) }
                        * test_jdet[index]
                        * unsafe { trial_table.get_unchecked(0, index, trial_i, 0) }
                        * trial_jdet[index];
                }
                *matrix.get_mut(*test_dof, *trial_dof).unwrap() = sum;
            }
        }
        for cell1 in grid.topology().adjacent_cells(cell0).iter() {
            if cell1.1 == 2 {
                let test_edges = grid.topology().connectivity(2, 1).row(cell0).unwrap();
                let trial_edges = grid.topology().connectivity(2, 1).row(cell1.0).unwrap();
                
                println!(
                    "OFF DIAGONAL: {} {} (connected by {} vertex/vertices)",
                    cell0, cell1.0, cell1.1
                );
                println!("{:?} {:?}", grid.geometry().cell_vertices(cell0).unwrap(), grid.geometry().cell_vertices(cell1.0).unwrap());
                println!("{:?} {:?}", grid.topology().connectivity(2, 1).row(cell0).unwrap(), grid.topology().connectivity(2, 1).row(cell1.0).unwrap());
            }
        }
    }
    println!("Laplace single layer matrix");
    for i in 0..dofmap.global_size() {
        println!("{:?}", matrix.row(i).unwrap());
    }
    // Diagonal value computed using Bempp-cl is 0.1854538822982487
    println!(
        "Relative error of diagonal entry compared to Bempp-cl: {}",
        (matrix.get(0, 0).unwrap() - 0.1854538822982487).abs() / 0.1854538822982487
    );
}
