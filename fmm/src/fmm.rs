// TODO Should be generic over kernel float type parameter - this requires trees to be generic over float type 
// TODO should check what happens with rectangular distributions of points would be easier to do as a part of the above todo.
// TODO: charge input should be utilized NOW!

use itertools::Itertools;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use rlst::algorithms::linalg::LinAlg;
use rlst::algorithms::traits::pseudo_inverse::Pinv;
use rlst::common::traits::NewLikeSelf;
use rlst::common::traits::Eval;
use rlst::dense::{base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix};
use rlst::dense::{rlst_mat, rlst_pointer_mat, traits::*, Dot};
use rlst::{
    self,
    dense::rlst_col_vec,
};

use bempp_traits::{
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, FmmLoop,SourceTranslation, TargetTranslation},
    kernel::{EvalType, Kernel},
    tree::Tree,
};
use bempp_tree::{
    constants::ROOT,
    types::single_node::SingleNodeTree,
};

use crate::types::{Charges, C2EType, FmmData, KiFmm};


#[allow(dead_code)]
impl<T, U> KiFmm<SingleNodeTree, T, U>
where
    T: Kernel<T = f64>,
    U: FieldTranslationData<T>,
{
    pub fn new<'a>(
        order: usize,
        alpha_inner: f64,
        alpha_outer: f64,
        kernel: T,
        tree: SingleNodeTree,
        m2l: U,
    ) -> Self {
        let upward_equivalent_surface = ROOT.compute_surface(tree.get_domain(), order, alpha_inner);
        let upward_check_surface = ROOT.compute_surface(tree.get_domain(), order, alpha_outer);
        let downward_equivalent_surface =
            ROOT.compute_surface(tree.get_domain(), order, alpha_outer);
        let downward_check_surface = ROOT.compute_surface(tree.get_domain(), order, alpha_inner);

        let nequiv_surface = upward_equivalent_surface.len() / kernel.space_dimension();
        let ncheck_surface = upward_check_surface.len() / kernel.space_dimension();

        // Store in RLST matrices
        let upward_equivalent_surface = unsafe {
            rlst_pointer_mat!['a, f64, upward_equivalent_surface.as_ptr(), (nequiv_surface, kernel.space_dimension()), (1, nequiv_surface)]
        };
        let upward_check_surface = unsafe {
            rlst_pointer_mat!['a, f64, upward_check_surface.as_ptr(), (ncheck_surface, kernel.space_dimension()), (1, ncheck_surface)]
        };
        let downward_equivalent_surface = unsafe {
            rlst_pointer_mat!['a, f64, downward_equivalent_surface.as_ptr(), (nequiv_surface, kernel.space_dimension()), (1, nequiv_surface)]
        };
        let downward_check_surface = unsafe {
            rlst_pointer_mat!['a, f64, downward_check_surface.as_ptr(), (ncheck_surface, kernel.space_dimension()), (1, ncheck_surface)]
        };

        // Compute upward check to equivalent, and downward check to equivalent Gram matrices
        // as well as their inverses using DGESVD.
        let mut uc2e = rlst_mat![f64, (ncheck_surface, nequiv_surface)];
        kernel.gram(
            EvalType::Value,
            upward_equivalent_surface.data(),
            upward_check_surface.data(),
            uc2e.data_mut(),
        );

        let mut dc2e = rlst_mat![f64, (ncheck_surface, nequiv_surface)];
        kernel.gram(
            EvalType::Value,
            downward_equivalent_surface.data(),
            downward_check_surface.data(),
            dc2e.data_mut(),
        );

        let (s, ut, v) = uc2e.linalg().pinv(None).unwrap();
        let s = s.unwrap();
        let ut = ut.unwrap();
        let v = v.unwrap();
        let mut mat_s = rlst_mat![f64, (s.len(), s.len())];
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i];
        }
        let uc2e_inv = v.dot(&mat_s).dot(&ut);

        let (s, ut, v) = dc2e.linalg().pinv(None).unwrap();
        let s = s.unwrap();
        let ut = ut.unwrap();
        let v = v.unwrap();
        let mut mat_s = rlst_mat![f64, (s.len(), s.len())];
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i];
        }
        let dc2e_inv = v.dot(&mat_s).dot(&ut);

        // Calculate M2M/L2L matrices
        let children = ROOT.children();
        let mut m2m: Vec<C2EType> = Vec::new();
        let mut l2l: Vec<C2EType> = Vec::new();

        for child in children.iter() {
            let child_upward_equivalent_surface =
                child.compute_surface(tree.get_domain(), order, alpha_inner);
            let child_downward_check_surface =
                child.compute_surface(tree.get_domain(), order, alpha_inner);
            let child_upward_equivalent_surface = unsafe {
                rlst_pointer_mat!['a, f64, child_upward_equivalent_surface.as_ptr(), (nequiv_surface, kernel.space_dimension()), (1, nequiv_surface)]
            };
            let child_downward_check_surface = unsafe {
                rlst_pointer_mat!['a, f64, child_downward_check_surface.as_ptr(), (ncheck_surface, kernel.space_dimension()), (1, ncheck_surface)]
            };

            let mut pc2ce = rlst_mat![f64, (ncheck_surface, nequiv_surface)];

            kernel.gram(
                EvalType::Value,
                child_upward_equivalent_surface.data(),
                upward_check_surface.data(),
                pc2ce.data_mut(),
            );

            m2m.push(uc2e_inv.dot(&pc2ce).eval());

            let mut cc2pe = rlst_mat![f64, (ncheck_surface, nequiv_surface)];

            kernel.gram(
                EvalType::Value,
                downward_equivalent_surface.data(),
                &child_downward_check_surface.data(),
                cc2pe.data_mut(),
            );
            l2l.push((kernel.scale(child.level()) * dc2e_inv.dot(&cc2pe)).eval());
        }

        Self {
            order,
            uc2e_inv,
            dc2e_inv,
            alpha_inner,
            alpha_outer,
            m2m,
            l2l,
            kernel,
            tree,
            m2l,
        }
    }
}

#[allow(dead_code)]
impl<T, U> FmmData<KiFmm<SingleNodeTree, T, U>>
where
    T: Kernel,
    U: FieldTranslationData<T>,
{
    pub fn new(fmm: KiFmm<SingleNodeTree, T, U>, _charges: Charges) -> Self {
        let mut multipoles = HashMap::new();
        let mut locals = HashMap::new();
        let mut potentials = HashMap::new();
        let mut points = HashMap::new();
        let mut charges = HashMap::new();

        let ncoeffs = fmm.m2l.ncoeffs(fmm.order);

        let dummy = rlst_col_vec![f64, ncoeffs];

        if let Some(keys) = fmm.tree().get_all_keys() {
            for key in keys.iter() {
                multipoles.insert(*key, Arc::new(Mutex::new(dummy.new_like_self().eval())));
                locals.insert(*key, Arc::new(Mutex::new(dummy.new_like_self().eval())));
                if let Some(point_data) = fmm.tree().get_points(key) {
                    points.insert(*key, point_data.iter().cloned().collect_vec());

                    // TODO: Fragile
                    let npoints = point_data.len();
                    potentials.insert(*key, Arc::new(Mutex::new(rlst_col_vec![f64, npoints])));
                    // TODO: Replace with a global index lookup at some point
                    charges.insert(*key, Arc::new(vec![1.0; point_data.len()]));
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

impl<T, U, V> Fmm for KiFmm<T, U, V>
where
    T: Tree,
    U: Kernel,
    V: FieldTranslationData<U>,
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

impl<T> FmmLoop for FmmData<T>
where
    T: Fmm,
    FmmData<T>: SourceTranslation + FieldTranslation + TargetTranslation,
{
    fn upward_pass(&self) {
        // Particle to Multipole
        let start = Instant::now();
        self.p2m();
        println!("P2M = {:?}ms", start.elapsed().as_millis());

        // Multipole to Multipole
        let depth = self.fmm.tree().get_depth();
        let start = Instant::now();
        for level in (1..=depth).rev() {
            self.m2m(level)
        }
        println!("M2M = {:?}ms", start.elapsed().as_millis());
    }

    fn downward_pass(&self) {
        let depth = self.fmm.tree().get_depth();
        let mut l2l_time = 0;
        let mut m2l_time = 0;
        for level in 2..=depth {
            if level > 2 {
                let start = Instant::now();
                self.l2l(level);
                l2l_time += start.elapsed().as_millis();
            }

            let start = Instant::now();
            self.m2l(level);
            m2l_time += start.elapsed().as_millis();
        }
        println!("M2L = {:?}ms", m2l_time);
        println!("L2L = {:?}ms", l2l_time);

        let start = Instant::now();
        // Leaf level computations
        self.p2l();
        println!("P2L = {:?}ms", start.elapsed().as_millis());

        // // Sum all potential contributions
        let start = Instant::now();
        self.m2p();
        println!("M2P = {:?}ms", start.elapsed().as_millis());
        let start = Instant::now();
        self.p2p();
        println!("P2P = {:?}ms", start.elapsed().as_millis());
        let start = Instant::now();
        self.l2p();
        println!("L2P = {:?}ms", start.elapsed().as_millis());
    }

    fn run(&self) {
        self.upward_pass();
        self.downward_pass();
    }
}

#[allow(unused_imports)]
#[allow(warnings)]
mod test {
    use bempp_kernel::laplace_3d::evaluate_laplace_one_target;
    use bempp_field::types::SvdFieldTranslationKiFmm;
    // use approx::{assert_relative_eq, RelativeEq};
    use rand::prelude::*;
    use rand::SeedableRng;

    // use bempp_tree::types::point::PointType;
    // use rayon::ThreadPool;

    use bempp_kernel::laplace_3d::Laplace3dKernel;
    // // use crate::laplace::LaplaceKernel;

    use rlst::{common::traits::ColumnMajorIterator, dense::rlst_rand_mat};

    use super::*;

    // #[allow(dead_code)]
    // fn points_fixture(npoints: usize) -> Vec<Point> {
    //     let mut range = StdRng::seed_from_u64(0);
    //     let between = rand::distributions::Uniform::from(0.0..1.0);
    //     let mut points: Vec<[PointType; 3]> = Vec::new();

    //     for _ in 0..npoints {
    //         points.push([
    //             between.sample(&mut range),
    //             between.sample(&mut range),
    //             between.sample(&mut range),
    //         ])
    //     }

    //     let points = points
    //         .iter()
    //         .enumerate()
    //         .map(|(i, p)| Point {
    //             coordinate: *p,
    //             global_idx: i,
    //             base_key: MortonKey::default(),
    //             encoded_key: MortonKey::default(),
    //         })
    //         .collect_vec();
    //     points
    // }
    fn points_fixture(
        npoints: usize,
        min: Option<f64>,
        max: Option<f64>,
    ) -> Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>
    {
        // Generate a set of randomly distributed points
        let mut range = StdRng::seed_from_u64(0);

        let between;
        if let (Some(min), Some(max)) = (min, max) {
            between = rand::distributions::Uniform::from(min..max);
        } else {
            between = rand::distributions::Uniform::from(0.0_f64..1.0_f64);
        }

        let mut points = rlst_mat![f64, (npoints, 3)];

        for i in 0..npoints {
            points[[i, 0]] = between.sample(&mut range);
            points[[i, 1]] = between.sample(&mut range);
            points[[i, 2]] = between.sample(&mut range);
        }

        points
    }

    // #[test]
    // fn test_upward_pass() {
    //     let npoints = 1000;
    //     let points = points_fixture(npoints, None, None);

    //     let order = 5;
    //     let alpha_inner = 1.05;
    //     let alpha_outer = 2.9;
    //     let adaptive = false;
    //     let k = 50;
    //     let ncrit = 100;
    //     let depth = 2;
    //     let kernel = Laplace3dKernel::<f64>::default();

    //     let start = Instant::now();
    //     let tree = SingleNodeTree::new(points.data(), adaptive, Some(ncrit), Some(depth));
    //     println!("Tree = {:?}ms", start.elapsed().as_millis());

    //     let start = Instant::now();

    //     //     // let m2l_data_svd_naive = SvdFieldTranslationNaiveKiFmm::new(
    //     //     //     kernel.clone(),
    //     //     //     Some(k),
    //     //     //     order,
    //     //     //     tree.get_domain().clone(),
    //     //     //     alpha_inner,
    //     //     // );

    //     let m2l_data_svd = SvdFieldTranslationKiFmm::new(
    //         kernel.clone(),
    //         Some(k),
    //         order,
    //         tree.get_domain().clone(),
    //         alpha_inner,
    //     );
    //     println!("SVD operators = {:?}ms", start.elapsed().as_millis());

    //     //     let start = Instant::now();
    //     //     let m2l_data_fft = FftFieldTranslationNaiveKiFmm::new(
    //     //         kernel.clone(),
    //     //         order,
    //     //         tree.get_domain().clone(),
    //     //         alpha_inner,
    //     //     );
    //     //     println!("FFT operators = {:?}ms", start.elapsed().as_millis());

    //     let fmm = KiFmm::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data_svd);

    //     let charges = Charges::new();
    //     let datatree = FmmData::new(fmm, charges);
    //     datatree.upward_pass();

    //     // let e = e.unwrap().lock().unwrap();
    //     // let e= datatree.multipoles.get(&ROOT).unwrap().lock().unwrap().deref();

    //     let pt = vec![100., 0., 0.];
    //     let distant_point = unsafe { rlst_pointer_mat!['static, f64, pt.as_ptr(), (1, 3), (1, 1)] };

    //     let charges = vec![1.0; npoints];
    //     let charges =
    //         unsafe { rlst_pointer_mat!['static, f64, charges.as_ptr(), (1, npoints), (1, 1)] };
    //     let mut direct = rlst_col_vec![f64, 1];
    //     evaluate_laplace_one_target(
    //         EvalType::Value,
    //         distant_point.data(),
    //         points.data(),
    //         charges.data(),
    //         direct.data_mut(),
    //     );

    //     let mut result = rlst_col_vec![f64, 1];

    //     let upward_equivalent_surface = ROOT.compute_surface(
    //         datatree.fmm.tree().get_domain(),
    //         datatree.fmm.order,
    //         datatree.fmm.alpha_inner,
    //     );
    //     let binding = datatree.multipoles.get(&ROOT).unwrap().lock().unwrap();
    //     let multipole_expansion = binding.deref();

    //     evaluate_laplace_one_target(
    //         EvalType::Value,
    //         distant_point.data(),
    //         &upward_equivalent_surface[..],
    //         multipole_expansion.data(),
    //         result.data_mut(),
    //     );

    //     result.pretty_print();
    //     direct.pretty_print();
    //     // kernel.evaluate_st(EvalType::Value, points.data(), , charges, result)
    //     // println!("distant {:?}", distant_point)
    //     assert!(false)
    // }

    #[test]
    fn test_fmm<'a>() {
        let npoints = 1000;
        let points = points_fixture(npoints, None, None);

        let order = 6;
        let alpha_inner = 1.05;
        let alpha_outer = 2.9;
        let adaptive = false;
        // TODO: Have to pass this information to data tree creation!!!!
        let k = 1000;
        let ncrit = 100;
        let depth = 2;
        let kernel = Laplace3dKernel::<f64>::default();

        let start = Instant::now();
        let tree = SingleNodeTree::new(points.data(), adaptive, Some(ncrit), Some(depth));
        println!("Tree = {:?}ms", start.elapsed().as_millis());

        let start = Instant::now();

        let m2l_data_svd = SvdFieldTranslationKiFmm::new(
            kernel.clone(),
            Some(k),
            order,
            tree.get_domain().clone(),
            alpha_inner,
        );
        println!("SVD operators = {:?}ms", start.elapsed().as_millis());

        //     let start = Instant::now();
        //     let m2l_data_fft = FftFieldTranslationNaiveKiFmm::new(
        //         kernel.clone(),
        //         order,
        //         tree.get_domain().clone(),
        //         alpha_inner,
        //     );
        //     println!("FFT operators = {:?}ms", start.elapsed().as_millis());

        let fmm = KiFmm::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data_svd);

        let charges = Charges::new();
        let datatree = FmmData::new(fmm, charges);
        datatree.run();

        let leaf = &datatree.fmm.tree.get_leaves().unwrap()[0];

        let potentials = datatree.potentials.get(&leaf).unwrap().lock().unwrap();
        let pts = datatree.fmm.tree().get_points(&leaf).unwrap();

        let leaf_coordinates = pts
            .iter()
            .map(|p| p.coordinate)
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();

        let ntargets = leaf_coordinates.len() / datatree.fmm.kernel.space_dimension();

        // Get into row major order
        let leaf_coordinates = unsafe {
            rlst_pointer_mat!['a, f64, leaf_coordinates.as_ptr(), (ntargets, datatree.fmm.kernel.space_dimension()), (datatree.fmm.kernel.space_dimension(), 1)]
        }.eval();

        let mut direct = vec![0f64; pts.len()];
        let all_point_coordinates = points_fixture(npoints, None, None);

        let all_charges = vec![1f64; npoints];

        let kernel = Laplace3dKernel::<f64>::default();

        kernel.evaluate_st(
            EvalType::Value,
            all_point_coordinates.data(),
            leaf_coordinates.data(),
            &all_charges[..],
            &mut direct[..],
        );

        println!("potentials {:?}", potentials.data());
        println!("direct {:?}", direct);

        //     let abs_error: f64 = potentials
        //         .iter()
        //         .zip(direct.iter())
        //         .map(|(a, b)| (a - b).abs())
        //         .sum();
        //     let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

        //     println!("p={:?} rel_error={:?}\n", order, rel_error);
        assert!(false)
    }
}
