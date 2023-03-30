use bempp_traits::fmm::FmmAlgorithm;
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
use std::ops::Deref;
use std::{collections::HashMap, hash::Hash};

use bempp_tree::constants::ROOT;
use ndarray::*;
use std::sync::{Arc, Mutex};

use crate::laplace::LaplaceKernel;
use crate::linalg::pinv;
use std::sync::MutexGuard;

use rayon::prelude::*;


pub struct FmmDataTree<T: Fmm> {
    fmm: Arc<T>,
    multipoles: HashMap<MortonKey, Arc<Mutex<Vec<f64>>>>,
    locals: HashMap<MortonKey, Vec<f64>>,
    potentials: HashMap<MortonKey, Vec<f64>>,
    points: HashMap<MortonKey, Vec<Point>>,
    charges: HashMap<MortonKey, Vec<f64>>
}

pub struct KiFmm<T: Tree, S: Kernel> {
    order: usize,

    uc2e_inv: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),

    alpha_inner: f64,
    alpha_outer: f64,

    m2m: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>>,

    tree: T,
    kernel: S,
}

impl KiFmm<SingleNodeTree, LaplaceKernel> {
    /// Number of coefficients related to a given expansion order.
    fn ncoeffs(order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }

    fn new<'a>(
        order: usize,
        alpha_inner: f64,
        alpha_outer: f64,
        kernel: LaplaceKernel,
        tree: SingleNodeTree,
    ) -> Self {
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

        let mut m2m: Vec<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = Vec::new();

        let nrows = KiFmm::ncoeffs(order);
        let ncols = nrows;

        let uc2e = Array1::from(uc2e)
            .to_shape((nrows, ncols))
            .unwrap()
            .to_owned();
        let (a, b, c) = pinv(&uc2e);
        let uc2e_inv = (a.to_owned(), b.dot(&c).to_owned());

        // Calculate M2M matrices
        let children = ROOT.children();

        for child in children.iter() {
            let child_upward_equivalent_surface =
                child.compute_surface(tree.get_domain(), order, alpha_inner)
                .into_iter()
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            let child_downward_check_surface =
                child.compute_surface(tree.get_domain(), order, alpha_inner)
                .into_iter()
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            let pc2ce = kernel
                .gram(&child_upward_equivalent_surface, &upward_check_surface)
                .unwrap();

            let pc2e = Array::from_shape_vec((nrows, ncols), pc2ce).unwrap();

            m2m.push(uc2e_inv.0.dot(&uc2e_inv.1.dot(&pc2e)));

            // let cc2pe = kernel
            //     .gram(&downward_equivalent_surface, &child_downward_check_surface)
            //     .unwrap();
            // let cc2pe = Array::from_shape_vec((ncols, nrows), cc2pe).unwrap();

            // l2l.push(kernel.scale(child.level()) * dc2e_inv.0.dot(&dc2e_inv.1.dot(&cc2pe)))
        }

        Self {
            order,
            uc2e_inv,
            alpha_inner,
            alpha_outer,
            m2m,
            kernel,
            tree,
        }
    }
}

impl FmmDataTree<KiFmm<SingleNodeTree, LaplaceKernel>> {
    fn new(fmm: KiFmm<SingleNodeTree, LaplaceKernel>) -> Self {
        let mut multipoles = HashMap::new();
        let mut locals = HashMap::new();
        let mut potentials = HashMap::new();
        let mut points = HashMap::new();
        let mut charges = HashMap::new();

        if let Some(keys) = fmm.tree().get_all_keys() {
            for key in keys.iter() {
                multipoles.insert(key.clone(), Arc::new(Mutex::new(Vec::new())));
                locals.insert(key.clone(), Vec::new());
                potentials.insert(key.clone(), Vec::new());
                if let Some(point_data) = fmm.tree().get_points(key) {
                    points.insert(key.clone(), point_data.iter().cloned().collect_vec());

                    // TODO: Replace with a global index lookup at some point
                    charges.insert(key.clone(), vec![1.0; point_data.len()]);
                }
            }  
        }

        let fmm = Arc::new(fmm);
        
        Self {
            fmm,
            multipoles,
            locals,
            potentials,
            points,
            charges,
        }
    }
}


impl SourceTranslation for FmmDataTree<KiFmm<SingleNodeTree, LaplaceKernel>> {
    fn p2m(&self) {

        let leaves = self.fmm.tree.get_leaves();
        
        leaves.par_iter().for_each(move |&leaf| {
            let multipoles = Arc::clone(&self.multipoles.get(&leaf).unwrap());
            let fmm = Arc::clone(&self.fmm);
            
            if let Some(points) = self.points.get(&leaf) {
                // Lookup data
                let coordinates = points
                    .iter()
                    .map(|p| p.coordinate)
                    .flat_map(|[x, y, z]| vec![x, y, z])
                    .collect_vec();

                let upward_check_surface = leaf
                    .compute_surface(&fmm.tree.domain, fmm.order, fmm.alpha_outer)
                    .into_iter()
                    .flat_map(|[x, y, z]| vec![x, y, z])
                    .collect_vec();

                let charges = self.charges.get(&leaf).unwrap(); 

                // Calculate check potential
                let mut check_potential = vec![0.; upward_check_surface.len() / self.fmm.kernel.dim()];
                
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

                let mut curr = multipoles.lock().unwrap();

                if !curr.is_empty() {
                    curr.iter_mut()
                        .zip(multipole_expansion.iter())
                        .for_each(|(c, m)| *c += *m);
                } else {
                    curr.extend(multipole_expansion);
                }
            }
        }); 
    }

    fn m2m(&self, level: u64) {

        // Parallelise over nodes at a given level
        if let Some(nodes) = self.fmm.tree.get_keys(level) {

            nodes.par_iter().for_each(move |&node| {
                let in_node = Arc::clone(&self.multipoles.get(&node).unwrap());
                let out_node = Arc::clone(&self.multipoles.get(&node.parent()).unwrap());
                let fmm = Arc::clone(&self.fmm);

                let operator_index = node 
                    .siblings()
                    .iter()
                    .position(|&x| x == node)
                    .unwrap();

                let in_multipole_lock = in_node.lock().unwrap();
                let in_multipole_ref = ArrayView::from(in_multipole_lock.deref());

                let out_expansion = fmm.m2m[operator_index].dot(&in_multipole_ref);
                let mut out_multipole = out_node.lock().unwrap();

                if !out_multipole.is_empty() {
                    out_multipole.iter_mut()
                        .zip(out_expansion.iter())
                        .for_each(|(c, m)| *c += *m);
                } else {
                    out_multipole.extend(out_expansion);
                }

            })
        }
    }
}


// impl TargetTranslation for FmmDataTree {
//     fn m2l(&self) {}

//     fn l2l(&self) {}

//     fn l2p(&self) {}

//     fn p2l(&self) {}

//     fn p2p(&self) {}
// }


impl<T, U> Fmm for KiFmm<T, U>
where
    T: Tree,
    U: Kernel,
{
    type Tree = T;
    type Kernel = U;

    fn order(&self) -> usize {
        self.order
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn tree(&self) -> &Self::Tree {
        &self.tree
    }
}

impl <T>FmmAlgorithm for FmmDataTree<T>
where 
    T: Fmm,
    FmmDataTree<T>: SourceTranslation
{

    fn upward_pass(&self) {

        // Particle to Multipole
        self.p2m();

        // Multipole to Multipole
        let depth = self.fmm.tree().get_depth();
        for level in (1..=depth).rev() {
            self.m2m(level)
        }
    }

    // fn downward_pass(&self) {        
    // }

    fn run(&self) {
        self.upward_pass();
        // self.downward_pass();
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
            })
            .collect_vec();
        points
    }

    #[test]
    fn test_upward_pass() {
        let npoints = 10000;
        let points = points_fixture(npoints);
        let depth = 3;
        let n_crit = 150;

        let order = 10;
        let alpha_inner = 1.05;
        let alpha_outer = 1.95;
        let adaptive = false;

        let kernel = LaplaceKernel {
            dim: 3,
            is_singular: false,
            value_dimension: 3,
        };

        let tree = SingleNodeTree::new(
            &points,
            adaptive,
            Some(n_crit),
            Some(depth)
        );

        let fmm = KiFmm::new(
            order,
            alpha_inner,
            alpha_outer,
            kernel,
            tree
        );

        let source_datatree = FmmDataTree::new(fmm);

        source_datatree.upward_pass();
 
        let distant_point = vec![1000., 0., 0.];
        let mut direct = vec![0.];
        let coordinates = points
            .iter()
            .map(|p| p.coordinate)
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec(); 
        let charges = vec![1.; coordinates.len()]; 
        source_datatree.fmm.kernel.potential(&coordinates[..], &charges[..], &distant_point[..], &mut direct);
        println!("Direct {:?}", direct);

        let mut estimate = vec![0.];

        let tree = SingleNodeTree::new(
            &points,
            adaptive,
            Some(n_crit),
            Some(depth)
        );

        let domain = tree.get_domain();
        
        let equivalent_surface = ROOT
            .compute_surface(&domain, order, alpha_inner)
            .into_iter()
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec(); 
        let expansion = source_datatree.multipoles.get(&ROOT).unwrap().lock().unwrap().deref().clone();

        source_datatree.fmm.kernel.potential(&equivalent_surface[..], &expansion[..], &distant_point[..], &mut estimate[..]);

        println!("FMM {:?}", estimate);
        assert!(false)
    }
}
