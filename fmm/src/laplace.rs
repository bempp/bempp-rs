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
    pub order: usize,
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
        order: usize,
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
        let upward_equivalent_surface = ROOT.compute_surface(order, alpha_inner, tree.get_domain());

        let upward_check_surface = ROOT.compute_surface(order, alpha_outer, tree.get_domain());

        let downward_equivalent_surface =
            ROOT.compute_surface(order, alpha_outer, tree.get_domain());

        let downward_check_surface = ROOT.compute_surface(order, alpha_inner, tree.get_domain());

        let uc2e = kernel
            .gram(&upward_equivalent_surface, &upward_check_surface)
            .unwrap();
        let dc2e = kernel
            .gram(&downward_equivalent_surface, &downward_check_surface)
            .unwrap();

        let nrows = KiFmm::ncoeffs(order);
        let ncols = nrows;

        let uc2e = Array1::from(uc2e)
            .to_shape((nrows, ncols))
            .unwrap()
            .to_owned();

        let dc2e = Array1::from(dc2e)
            .to_shape((nrows, ncols))
            .unwrap()
            .to_owned();

        let (a, b, c) = pinv(&uc2e);

        let uc2e_inv = (a.to_owned(), b.dot(&c).to_owned());

        let (a, b, c) = pinv(&dc2e);
        let dc2e_inv = (a.to_owned(), b.dot(&c).to_owned());

        KiFmm {
            kernel,
            tree,
            order,
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

impl LaplaceKernel {

    fn potential(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
    ) -> Vec<f64> {

        let mut potentials: Vec<f64> = vec![0.; targets.len()];

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
            potentials[i] = potential
        }
        potentials
    }

    fn gradient_kernel (
        &self,
        source: &[f64; 3],
        target: &[f64; 3],
        c: usize
    ) -> f64 {
        let num = source[c] - target[c];
        let invdiff: f64 = source
            .iter()
            .zip(target.iter())
            .map(|(s,t)| (s-t).powf(2.0))
            .sum::<f64>()
            .sqrt()
            * std::f64::consts::PI
            * 4.0;

        let mut tmp = invdiff.recip();
        tmp *= invdiff;
        tmp *= invdiff;
        tmp *= num;

        if tmp.is_finite() {
            tmp
        } else {
            0.
        }
            
    }

    fn gradient(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
    ) -> Vec<[f64; 3]> {

        let mut gradients: Vec<[f64; 3]> = vec![[0., 0., 0.]; targets.len()];

        for (i, target) in targets.iter().enumerate() {

            for (source, charge) in sources.iter().zip(charges) {
                gradients[i][0] -= charge*self.gradient_kernel(source, target, 0);
                gradients[i][1] -= charge*self.gradient_kernel(source, target, 1);
                gradients[i][2] -= charge*self.gradient_kernel(source, target, 2);
            }
        }

    gradients

    }
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

        match eval_type {
            EvalType::Value => {
                solvers_traits::types::Result::Ok(self.potential(sources, charges, targets))
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
            leaf.compute_surface(self.order, self.alpha_outer, self.tree.get_domain());

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

        let multipole_expansion = (self.kernel.scale(leaf.level())
            * self.uc2e_inv.0.dot(&self.uc2e_inv.1.dot(&check_potential)))
        .to_vec();

        self.tree
            .set_multipole_expansion(leaf, &multipole_expansion, self.order);
    }

    fn m2m(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {
        
        
    }

    fn l2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {

    }

    fn m2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {

    }

    fn l2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {

    }

    fn m2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {

    }
}

impl Fmm for KiFmm {
    fn upward_pass(&mut self) {
        println!("Running upward pass");

        // P2M over leaves
        // for leaf in self.tree.leaves {
        //     self.p2m(leaf)
        // }

        // M2M

            
    
    }
    fn downward_pass(&mut self) {
        println!("Running downward pass");
    }
    fn run(&mut self) {
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
        // Create Kernel
        let kernel = Box::new(LaplaceKernel {
            dim: 3,
            is_singular: true,
            value_dimension: 3,
        });

        // Create FmmTree
        let npoints: usize = 10000;
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
        let mut kifmm = KiFmm::new(6, 1.05, 1.95, tree, kernel);

        // Run P2M on some node containing points
        let mut node = kifmm.tree.get_leaves()[0];
        kifmm.p2m(&node);

        // Evaluate multipole expansion vs direct computation at some distant points
        let multipole = kifmm.tree.get_multipole_expansion(&node).unwrap();
        let upward_equivalent_surface =
            node.compute_surface(kifmm.order, kifmm.alpha_inner, kifmm.tree.get_domain());

        let distant_point = [[42.0, 0., 0.], [0., 0., 24.]];

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
