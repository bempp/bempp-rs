use bempp_traits::kernel::Kernel;
use bempp_traits::tree::Tree;
use bempp_traits::{
    fmm::{Fmm, SourceDataTree, SourceTranslation, TargetDataTree, TargetTranslation},
    tree::AttachedDataTree,
};
use bempp_tree::types::domain::Domain;
use bempp_tree::types::morton::MortonKey;
use bempp_tree::types::point::Point;
use bempp_tree::types::single_node::SingleNodeTree;
use itertools::Itertools;
use std::{collections::HashMap, hash::Hash};

use ndarray::*;

use bempp_tree::constants::ROOT;

use crate::linalg::pinv;

pub struct FmmDataTree {
    multipoles: HashMap<MortonKey, Vec<f64>>,
    locals: HashMap<MortonKey, Vec<f64>>,
    potentials: HashMap<MortonKey, Vec<f64>>,
    points: HashMap<MortonKey, Vec<Point>>,
}

pub struct KiFmmSingleNode {
    order: usize,

    uc2e_inv: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),

    alpha_inner: f64,
    alpha_outer: f64,
    kernel: Box<dyn Kernel<PotentialData = Vec<f64>>>,

    tree: SingleNodeTree,
}

impl KiFmmSingleNode {
    /// Number of coefficients related to a given expansion order.
    fn ncoeffs(order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }
}

impl FmmDataTree {
    fn new<'a>(tree: &SingleNodeTree) -> Self {
        let mut multipoles = HashMap::new();
        let mut locals = HashMap::new();
        let mut potentials = HashMap::new();
        let mut points = HashMap::new();

        for level in (0..=tree.get_depth()).rev() {
            if let Some(keys) = tree.get_keys(level) {
                for key in keys.iter() {
                    multipoles.insert(key.clone(), Vec::new());
                    locals.insert(key.clone(), Vec::new());
                    potentials.insert(key.clone(), Vec::new());
                    if let Some(point_data) = tree.get_points(key) {
                        points.insert(key.clone(), point_data.iter().cloned().collect_vec());
                    }
                }
            }
        }

        Self {
            multipoles,
            locals,
            potentials,
            points,
        }
    }
}

impl SourceDataTree for FmmDataTree {
    type Tree = SingleNodeTree;
    type Coefficient = f64;
    type Coefficients<'a> = &'a [f64];

    fn get_multipole_expansion<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Coefficients<'a>> {
        if let Some(multipole) = self.multipoles.get(key) {
            Some(multipole.as_slice())
        } else {
            None
        }
    }

    fn set_multipole_expansion<'a>(
        &'a mut self,
        key: &<Self::Tree as Tree>::NodeIndex,
        data: &Self::Coefficients<'a>,
    ) {
        if let Some(multipole) = self.multipoles.get_mut(key) {
            if !multipole.is_empty() {
                for (curr, &new) in multipole.iter_mut().zip(data.iter()) {
                    *curr += new;
                }
            } else {
                *multipole = data.clone().to_vec();
            }
        }
    }

    fn get_points<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::PointSlice<'a>> {
        if let Some(points) = self.points.get(key) {
            Some(points.as_slice())
        } else {
            None
        }
    }
}

impl SourceTranslation for FmmDataTree {
    type Fmm = KiFmmSingleNode;

    fn p2m(&mut self, fmm: &Self::Fmm) {
        for leaf in fmm.tree.get_leaves() {
            // Calculate check surface
            let upward_check_surface = leaf
                .compute_surface(fmm.tree.get_domain(), fmm.order(), fmm.alpha_outer)
                .into_iter()
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            if let Some(points) = fmm.tree.get_points(leaf) {
                // Lookup data
                let coordinates = points
                    .iter()
                    .map(|p| p.coordinate)
                    .flat_map(|[x, y, z]| vec![x, y, z])
                    .collect_vec();

                let charges = points.iter().map(|p| p.data[0]).collect_vec();

                // Calculate check potential
                let mut check_potential = vec![0.; upward_check_surface.len()/3];
                fmm.kernel.potential(
                    &coordinates[..],
                    &charges[..],
                    &upward_check_surface[..],
                    &mut check_potential[..],
                );
                let check_potential = Array1::from_vec(check_potential);

                // Calculate multipole expansion
                let multipole_expansion = fmm.kernel.scale(leaf.level())
                    * fmm.uc2e_inv.0.dot(&fmm.uc2e_inv.1.dot(&check_potential));
                let multipole_expansion = multipole_expansion.as_slice().unwrap();

                // Set multipole expansion at node
                self.set_multipole_expansion(leaf, &multipole_expansion);
            }
        }
    }

    fn m2m(&mut self, fmm: &Self::Fmm) {}
}

impl TargetTranslation for FmmDataTree {
    fn m2l(&self) {}

    fn l2l(&self) {}

    fn l2p(&self) {}

    fn p2l(&self) {}

    fn p2p(&self) {}
}

impl TargetDataTree for FmmDataTree {
    type Coefficient = f64;
    type Coefficients<'a> = &'a [f64];
    type Potential = f64;
    type Potentials<'a> = &'a [f64];
    type Tree = SingleNodeTree;

    fn get_local_expansion<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Coefficients<'a>> {
        if let Some(local) = self.locals.get(key) {
            Some(local.as_slice())
        } else {
            None
        }
    }

    fn set_local_expansion<'a>(
        &'a mut self,
        key: &<Self::Tree as Tree>::NodeIndex,
        data: &Self::Coefficients<'a>,
    ) {
        if let Some(locals) = self.locals.get_mut(key) {
            if !locals.is_empty() {
                for (curr, &new) in locals.iter_mut().zip(data.iter()) {
                    *curr += new;
                }
            } else {
                *locals = data.clone().to_vec();
            }
        }
    }

    fn get_potentials<'a>(
        &'a self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<Self::Potentials<'a>> {
        if let Some(potentials) = self.potentials.get(key) {
            Some(potentials.as_slice())
        } else {
            None
        }
    }

    fn set_potentials<'a>(
        &'a mut self,
        key: &<Self::Tree as Tree>::NodeIndex,
        data: &Self::Potentials<'a>,
    ) {
        if let Some(potentials) = self.potentials.get_mut(key) {
            if !potentials.is_empty() {
                for (curr, &new) in potentials.iter_mut().zip(data.iter()) {
                    *curr += new;
                }
            } else {
                *potentials = data.clone().to_vec();
            }
        }
    }
}

impl Fmm for KiFmmSingleNode {
    type Tree = SingleNodeTree;

    fn order(&self) -> usize {
        self.order
    }

    fn new<'a>(
        order: usize,
        alpha_inner: f64,
        alpha_outer: f64,
        kernel: Box<dyn Kernel<PotentialData = Vec<f64>>>,
        points: <Self::Tree as Tree>::PointSlice<'a>,
        point_data: <Self::Tree as Tree>::PointDataSlice<'a>,
        adaptive: bool,
        n_crit: Option<u64>,
        depth: Option<u64>,
    ) -> Self {
        let tree = SingleNodeTree::new(points, point_data, adaptive, n_crit, depth);
        let upward_equivalent_surface = ROOT
            .compute_surface(tree.get_domain(), order, alpha_inner)
            .into_iter()
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();

        let upward_check_surface = ROOT
            .compute_surface(tree.get_domain(), order, alpha_outer)
            .into_iter()
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();

        // Compute upward check to equivalent, and downward check to equivalent Gram matrices
        // as well as their inverses using DGESVD.
        let uc2e = kernel
            .gram(&upward_equivalent_surface, &upward_check_surface)
            .unwrap();

        let nrows = KiFmmSingleNode::ncoeffs(order);
        let ncols = nrows;

        let uc2e = Array1::from(uc2e)
            .to_shape((nrows, ncols))
            .unwrap()
            .to_owned();
        let (a, b, c) = pinv(&uc2e);
        let uc2e_inv = (a.to_owned(), b.dot(&c).to_owned());

        Self {
            order,
            uc2e_inv,
            alpha_inner,
            alpha_outer,
            kernel,
            tree,
        }
    }
}

mod test {

    use crate::laplace::LaplaceKernel;

    use super::*;

    use bempp_tree::types::point::{PointType, Points};
    use rand::prelude::*;
    use rand::SeedableRng;

    #[allow(dead_code)]
    fn points_fixture(npoints: usize) -> Vec<Point> {
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

        let points = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                global_idx: i,
                base_key: MortonKey::default(),
                encoded_key: MortonKey::default(),
                data: Vec::new(),
            })
            .collect_vec();
        points
    }

    #[test]
    fn test_p2m() {
        let npoints = 10000;
        let points = points_fixture(npoints);
        let point_data = vec![vec![1.0]; npoints];
        let depth = 3;
        let n_crit = 150;

        let order = 5;
        let alpha_inner = 1.05;
        let alpha_outer = 1.95;
        let adaptive = false;

        let kernel = Box::new(LaplaceKernel {
            dim: 3,
            is_singular: false,
            value_dimension: 3,
        });

        let fmm = KiFmmSingleNode::new(
            order,
            alpha_inner,
            alpha_outer,
            kernel,
            &points,
            &point_data,
            adaptive,
            Some(n_crit),
            Some(depth),
        );

        let mut datatree = FmmDataTree::new(&fmm.tree);

        datatree.p2m(&fmm);


    }
}
