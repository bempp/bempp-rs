// Laplace kernel
use std::collections::HashSet;
use std::vec;

use ndarray::*;
use ndarray_linalg::*;

use solvers_traits::fmm::KiFmmNode;
use solvers_traits::types::{Error, EvalType};
use solvers_traits::{
    fmm::{Fmm, FmmTree, Translation},
    kernel::Kernel,
};
use solvers_tree::constants::ROOT;
use solvers_tree::types::data::NodeData;

use solvers_tree::types::{
    domain::Domain,
    morton::{MortonKey, MortonKeys},
    point::{Point, Points},
};

use crate::linalg::pinv;

// TODO: Create from FMM Factory pattern, specialised for Rust in some way
pub struct KiFmm {
    pub kernel: Box<dyn Kernel<Data = Vec<f64>>>,
    pub tree: Box<
        dyn FmmTree<
            NodeIndex = MortonKey,
            NodeIndices = MortonKeys,
            FmmNodeDataType = NodeData,
            NodeDataType = NodeData,
            Domain = Domain,
            Point = Point,
            Points = Points,
            NodeIndicesSet = HashSet<MortonKey>,
            NodeDataContainer = Vec<f64>,
        >,
    >,
    pub order_equivalent: usize,
    pub order_check: usize,
    pub alpha_inner: f64,
    pub alpha_outer: f64,
    pub uc2e_inv: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),
    pub dc2e_inv: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),
}

impl KiFmm {
    fn ncoeffs(order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }

    fn new(
        order_equivalent: usize,
        order_check: usize,
        alpha_inner: f64,
        alpha_outer: f64,
        tree: Box<
            dyn FmmTree<
                Domain = Domain,
                Point = Point,
                Points = Points,
                NodeIndex = MortonKey,
                NodeIndices = MortonKeys,
                NodeIndicesSet = HashSet<MortonKey>,
                NodeDataType = NodeData,
                FmmNodeDataType = NodeData,
                NodeDataContainer = Vec<f64>,
            >,
        >,
        kernel: Box<dyn Kernel<Data = Vec<f64>>>,
    ) -> KiFmm {
        let upward_equivalent_surface =
            ROOT.compute_surface(order_equivalent, alpha_inner, tree.get_domain());

        let upward_check_surface =
            ROOT.compute_surface(order_check, alpha_outer, tree.get_domain());

        let downward_equivalent_surface =
            ROOT.compute_surface(order_equivalent, alpha_outer, tree.get_domain());

        let downward_check_surface =
            ROOT.compute_surface(order_check, alpha_inner, tree.get_domain());

        let uc2e = kernel
            .gram(&upward_equivalent_surface, &upward_check_surface)
            .unwrap();
        let dc2e = kernel
            .gram(&downward_equivalent_surface, &downward_check_surface)
            .unwrap();

        let nrows = KiFmm::ncoeffs(order_check);
        let ncols = KiFmm::ncoeffs(order_equivalent);

        let uc2e = Array1::from(uc2e)
            .to_shape((nrows, ncols))
            .unwrap()
            .to_owned();
        let dc2e = Array1::from(dc2e)
            .to_shape((nrows, ncols))
            .unwrap()
            .to_owned();

        let (a, b, c) = pinv(&uc2e);

        println!(
            "a r={:?} c={:?} b r={:?} c={:?} c r={:?} c={:?}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols(),
            c.nrows(),
            c.ncols()
        );

        let uc2e_inv = (a.to_owned(), b.dot(&c).to_owned());

        let (a, b, c) = pinv(&dc2e);
        let dc2e_inv = (a.to_owned(), b.dot(&c).to_owned());

        KiFmm {
            kernel,
            tree,
            order_equivalent,
            order_check,
            alpha_inner,
            alpha_outer,
            uc2e_inv,
            dc2e_inv,
        }
    }
}

pub struct LaplaceKernel {
    pub dim: usize,
    pub is_singular: bool,
    pub value_dimension: usize,
}

impl Kernel for LaplaceKernel {
    type Data = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn is_singular(&self) -> bool {
        self.is_singular
    }

    fn value_dimension(&self) -> usize {
        self.value_dimension
    }

    fn evaluate(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
        eval_type: &solvers_traits::types::EvalType,
    ) -> solvers_traits::types::Result<Self::Data> {
        let mut result: Vec<f64> = vec![0.; targets.len()];

        match eval_type {
            EvalType::Value => {
                for (i, target) in targets.iter().enumerate() {
                    let mut potential = 0.0;

                    for (source, charge) in sources.iter().zip(charges) {
                        let mut tmp = source
                            .iter()
                            .zip(target.iter())
                            .map(|(s, t)| (s - t).powf(2.0))
                            .sum::<f64>()
                            .powf(0.5)
                            * std::f64::consts::PI
                            * 4.0;

                        tmp = tmp.recip();

                        if tmp.is_infinite() {
                            continue;
                        } else {
                            potential += tmp * charge;
                        }
                    }
                    result[i] = potential
                }
                solvers_traits::types::Result::Ok(result)
            }
            _ => solvers_traits::types::Result::Err(Error::Generic("foo".to_string())),
        }
    }

    fn gram(
        &self,
        sources: &[[f64; 3]],
        targets: &[[f64; 3]],
    ) -> solvers_traits::types::Result<Self::Data> {
        let mut result: Vec<f64> = Vec::new();

        for target in targets.iter() {
            let mut row: Vec<f64> = Vec::new();
            for source in sources.iter() {
                let mut tmp = source
                    .iter()
                    .zip(target.iter())
                    .map(|(s, t)| (s - t).powf(2.0))
                    .sum::<f64>()
                    .powf(0.5)
                    * std::f64::consts::PI
                    * 4.0;

                tmp = tmp.recip();

                if tmp.is_infinite() {
                    row.push(0.0);
                } else {
                    row.push(tmp);
                }
            }
            result.append(&mut row);
        }

        Result::Ok(result)
    }

    fn scale(&self, level: u64) -> f64 {
        1. / (2f64.powf(level as f64))
    }
}

impl Translation for KiFmm {
    type NodeIndex = MortonKey;

    fn p2m(&mut self, leaf: &Self::NodeIndex) {
        let upward_check_surface =
            leaf.compute_surface(self.order_check, self.alpha_outer, self.tree.get_domain());

        let sources = self.tree.get_points(leaf).unwrap();
        let charges: Vec<f64> = sources.iter().map(|s| s.data).collect();
        let sources: Vec<[f64; 3]> = sources.iter().map(|s| s.coordinate).collect();

        // Check potential
        let check_potential = self
            .kernel
            .evaluate(
                &sources[..],
                &charges[..],
                &upward_check_surface[..],
                &EvalType::Value,
            )
            .unwrap();

        let check_potential = Array1::from_vec(check_potential);

        let tmp = &self.uc2e_inv.1.dot(&check_potential);
        let multipole_expansion = self.kernel.scale(leaf.level()) * self.uc2e_inv.0.dot(tmp);

        let multipole_expansion = multipole_expansion.to_vec();

        self.tree
            .set_multipole_expansion(&leaf, &multipole_expansion, self.order_equivalent);
    }

    fn m2m(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {}

    fn l2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {}

    fn m2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {}

    fn l2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {}

    fn m2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {}
}

impl Fmm for KiFmm {
    fn upward_pass(&mut self) {
        println!("Running upward pass");
    }
    fn downward_pass(&mut self) {
        println!("Running downward pass");
    }
    fn run(&mut self, expansion_order: usize) {
        println!("Running FMM");
        self.upward_pass();
        self.downward_pass();
    }
}

mod test {
    use std::vec;

    use solvers_traits::fmm::KiFmmNode;
    use solvers_traits::fmm::Translation;
    use solvers_traits::types::EvalType;
    use solvers_tree::types::morton::MortonKey;
    use solvers_tree::types::point::PointType;
    use solvers_tree::types::single_node::SingleNodeTree;

    use super::{KiFmm, LaplaceKernel};
    use rand::prelude::*;
    use rand::SeedableRng;

    use float_cmp::assert_approx_eq;

    macro_rules! assert_delta {
        ($x:expr, $y:expr, $d:expr) => {
            if !(($x - $y) / ($x) < $d || ($y - $x) / ($x) < $d) {
                panic!();
            }
        };
    }

    pub fn points_fixture(npoints: usize) -> Vec<[f64; 3]> {
        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0..1.0);
        let mut points: Vec<[PointType; 3]> = Vec::new();

        for _ in 0..npoints {
            points.push([
                between.sample(&mut range),
                between.sample(&mut range),
                between.sample(&mut range),
            ])
        }

        points
    }

    #[test]
    fn test_p2m() {
        // Create kernel
        let kernel = Box::new(LaplaceKernel {
            dim: 3,
            is_singular: true,
            value_dimension: 3,
        });

        // Create FmmTree
        let npoints: usize = 1000;
        let points = points_fixture(npoints);
        let point_data = vec![1.0; npoints];
        let depth = 1;
        let n_crit = 150;

        let tree = Box::new(SingleNodeTree::new(
            &points,
            &point_data,
            false,
            Some(n_crit),
            Some(depth),
        ));

        // New FMM
        let mut kifmm = KiFmm::new(6, 6, 1.05, 1.95, tree, kernel);

        let mut node = kifmm.tree.get_leaves()[0];

        for leaf in kifmm.tree.get_leaves().iter() {
            if kifmm.tree.get_points(leaf).iter().len() == 0 {
                continue;
            } else {
                node = *leaf;
                break;
            }
        }

        kifmm.p2m(&node);

        let multipole = kifmm.tree.get_multipole_expansion(&node).unwrap();
        let upward_equivalent_surface = node.compute_surface(
            kifmm.order_equivalent,
            kifmm.alpha_inner,
            kifmm.tree.get_domain(),
        );

        let distant_point = [[100.0, 0., 0.], [0., 0., 23.]];

        let node_points = kifmm.tree.get_points(&node).unwrap();
        let node_point_data: Vec<f64> = node_points.iter().map(|p| p.data).collect();
        let node_points: Vec<[f64; 3]> = node_points.iter().map(|p| p.coordinate).collect();

        let direct = kifmm
            .kernel
            .evaluate(
                &node_points,
                &node_point_data,
                &distant_point,
                &EvalType::Value,
            )
            .unwrap();

        let result = kifmm
            .kernel
            .evaluate(
                &upward_equivalent_surface,
                &multipole,
                &distant_point,
                &EvalType::Value,
            )
            .unwrap();

        // Test that correct digits match the expansion order
        for (a, b) in result.iter().zip(direct.iter()) {
            assert_approx_eq!(f64, *a, *b, epsilon = 1e-6);
        }
    }
}
