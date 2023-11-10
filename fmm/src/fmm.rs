//! Implementation of constructors for FMMs as well as the implementation of FmmData, Fmm traits.
use cauchy::Scalar;
use itertools::Itertools;
use num::Float;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use rlst::{
    common::traits::{Eval, NewLikeSelf, Transpose},
    dense::{rlst_col_vec, rlst_dynamic_mat, rlst_pointer_mat, traits::*, Dot},
};

use bempp_traits::{
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, FmmLoop, SourceTranslation, TargetTranslation, TimeDict},
    kernel::{Kernel, KernelScale},
    tree::Tree,
    types::EvalType,
};
use bempp_tree::{constants::ROOT, types::single_node::SingleNodeTree};

use crate::pinv::pinv;
use crate::types::{C2EType, ChargeDict, FmmData, KiFmm};

/// Implementation of constructor for single node KiFMM
impl<'a, K, R, F> KiFmm<SingleNodeTree<R>, K, F>
where
    K: Kernel<T = R> + KernelScale<T = R>,
    R: Scalar + Default + Float,
    F: FieldTranslationData<K>,
{
    /// Constructor for single node kernel independent FMM (KiFMM). This object contains all the precomputed operator matrices and metadata, as well as references to
    /// the associated single node octree, and the associated kernel function.
    ///
    /// # Arguments
    /// * `order` - The expansion order for the multipole and local expansions.
    /// * `alpha_inner` - The ratio of the inner check surface diamater in comparison to the surface discretising a box.
    /// * `alpha_order` - The ratio of the outer check surface diamater in comparison to the surface discretising a box.
    /// * `kernel` - The kernel function for this FMM.
    /// * `tree` - The type of tree associated with this FMM, can be single or multi node.
    /// * `m2l` - The M2L operator matrices, as well as metadata associated with this FMM.
    pub fn new(
        order: usize,
        alpha_inner: f64,
        alpha_outer: f64,
        kernel: K,
        tree: SingleNodeTree<R>,
        m2l: F,
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
        let mut uc2e = rlst_dynamic_mat![f64, (ncheck_surface, nequiv_surface)];
        kernel.assemble_st(
            EvalType::Value,
            upward_equivalent_surface.data(),
            upward_check_surface.data(),
            uc2e.data_mut(),
        );

        // Need to tranapose so that rows correspond to targets and columns to sources
        let uc2e = uc2e.transpose().eval();

        let mut dc2e = rlst_dynamic_mat![f64, (ncheck_surface, nequiv_surface)];
        kernel.assemble_st(
            EvalType::Value,
            downward_equivalent_surface.data(),
            downward_check_surface.data(),
            dc2e.data_mut(),
        );

        // Need to tranapose so that rows correspond to targets and columns to sources
        let dc2e = dc2e.transpose().eval();

        let (s, ut, v) = pinv::<f64>(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_mat![f64, (s.len(), s.len())];
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i];
        }
        let uc2e_inv_1 = v.dot(&mat_s);
        let uc2e_inv_2 = ut;

        let (s, ut, v) = pinv::<f64>(&dc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_mat![f64, (s.len(), s.len())];
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i];
        }
        let dc2e_inv_1 = v.dot(&mat_s);
        let dc2e_inv_2 = ut;

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

            let mut pc2ce = rlst_dynamic_mat![f64, (ncheck_surface, nequiv_surface)];

            kernel.assemble_st(
                EvalType::Value,
                child_upward_equivalent_surface.data(),
                upward_check_surface.data(),
                pc2ce.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let pc2ce = pc2ce.transpose().eval();

            m2m.push(uc2e_inv_1.dot(&uc2e_inv_2.dot(&pc2ce)).eval());

            let mut cc2pe = rlst_dynamic_mat![f64, (ncheck_surface, nequiv_surface)];

            kernel.assemble_st(
                EvalType::Value,
                downward_equivalent_surface.data(),
                child_downward_check_surface.data(),
                cc2pe.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let cc2pe = cc2pe.transpose().eval();
            l2l.push(
                (kernel.scale(child.level()) * dc2e_inv_1.dot(&dc2e_inv_2.dot(&cc2pe))).eval(),
            );
        }

        Self {
            order,
            uc2e_inv_1,
            uc2e_inv_2,
            dc2e_inv_1,
            dc2e_inv_2,
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

/// Implementation of the data structure to store the data for the single node KiFMM.
impl<K, F, R> FmmData<KiFmm<SingleNodeTree<R>, K, F>>
where
    K: Kernel,
    F: FieldTranslationData<K>,
    R: Float + Scalar + Default,
{
    /// Constructor fo the KiFMM's associated FmmData on a single node.
    ///
    /// # Arguments
    /// `fmm` - A single node KiFMM object.
    /// `global_charges` - The charge data associated to the point data via unique global indices.
    pub fn new(fmm: KiFmm<SingleNodeTree<R>, K, F>, global_charges: &ChargeDict) -> Self {
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

                    let npoints = point_data.len();
                    potentials.insert(*key, Arc::new(Mutex::new(rlst_col_vec![f64, npoints])));

                    // Lookup indices and store with charges
                    let mut tmp_idx = Vec::new();
                    for point in point_data.iter() {
                        tmp_idx.push(point.global_idx)
                    }
                    let mut tmp_charges = vec![0.; point_data.len()];
                    for i in 0..tmp_idx.len() {
                        tmp_charges[i] = *global_charges.get(&tmp_idx[i]).unwrap();
                    }

                    charges.insert(*key, Arc::new(tmp_charges));
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
    fn upward_pass(&self, time: Option<bool>) -> Option<TimeDict> {
        match time {
            Some(true) => {
                let mut times = TimeDict::default();
                // Particle to Multipole
                let start = Instant::now();
                self.p2m();
                times.insert("p2m".to_string(), start.elapsed().as_millis());

                // Multipole to Multipole
                let depth = self.fmm.tree().get_depth();
                let start = Instant::now();
                for level in (1..=depth).rev() {
                    self.m2m(level)
                }
                times.insert("m2m".to_string(), start.elapsed().as_millis());
                Some(times)
            }
            Some(false) | None => {
                // Particle to Multipole
                self.p2m();

                // Multipole to Multipole
                let depth = self.fmm.tree().get_depth();
                for level in (1..=depth).rev() {
                    self.m2m(level)
                }
                None
            }
        }
    }

    fn downward_pass(&self, time: Option<bool>) -> Option<TimeDict> {
        let depth = self.fmm.tree().get_depth();

        match time {
            Some(true) => {
                let mut times = TimeDict::default();
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

                times.insert("l2l".to_string(), l2l_time);
                times.insert("m2l".to_string(), m2l_time);

                // Leaf level computations
                let start = Instant::now();
                self.p2l();
                times.insert("p2l".to_string(), start.elapsed().as_millis());

                // Sum all potential contributions
                let start = Instant::now();
                self.m2p();
                times.insert("m2p".to_string(), start.elapsed().as_millis());

                let start = Instant::now();
                self.p2p();
                times.insert("p2p".to_string(), start.elapsed().as_millis());

                let start = Instant::now();
                self.l2p();
                times.insert("l2p".to_string(), start.elapsed().as_millis());

                Some(times)
            }
            Some(false) | None => {
                for level in 2..=depth {
                    if level > 2 {
                        self.l2l(level);
                    }
                    self.m2l(level);
                }
                // Leaf level computations
                self.p2l();

                // Sum all potential contributions
                self.m2p();
                self.p2p();
                self.l2p();

                None
            }
        }
    }

    fn run(&self, time: Option<bool>) -> Option<TimeDict> {
        let t1 = self.upward_pass(time);
        let t2 = self.downward_pass(time);

        if let (Some(mut t1), Some(t2)) = (t1, t2) {
            t1.extend(t2);
            Some(t1)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use std::env;

    use bempp_field::types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm};
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::implementations::helpers::points_fixture;

    use crate::charge::build_charge_dict;

    #[test]
    fn test_fmm_svd() {
        // Set OMP threads to avoid thrashing during matrix-matrix products in M2L.
        env::set_var("OMP_NUM_THREADS", "1");

        // Generate a set of point data
        let npoints = 10000;
        let points = points_fixture(npoints, None, None);
        let global_idxs = (0..npoints).collect_vec();
        let charges = vec![1.0; npoints];

        // Setup a FMM experiment
        let order = 6;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let adaptive = false;
        let k = 1000;
        let ncrit = 150;
        let depth = 3;

        // Create a kernel
        let kernel = Laplace3dKernel::<f64>::default();

        // Create a tree
        let tree = SingleNodeTree::new(
            points.data(),
            adaptive,
            Some(ncrit),
            Some(depth),
            &global_idxs[..],
        );

        // Precompute the M2L data
        let m2l_data_svd = SvdFieldTranslationKiFmm::new(
            kernel.clone(),
            Some(k),
            order,
            *tree.get_domain(),
            alpha_inner,
        );

        // Create an FMM
        let fmm = KiFmm::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data_svd);

        // Form charge dict, matching charges with their associated global indices
        let charge_dict = build_charge_dict(&global_idxs[..], &charges[..]);

        // Associate data with the FMM
        let datatree = FmmData::new(fmm, &charge_dict);

        // Run the experiment
        datatree.run(Some(true));

        // Test that direct computation is close to the FMM.
        let leaf = &datatree.fmm.tree.get_leaves().unwrap()[0];

        let potentials = datatree.potentials.get(leaf).unwrap().lock().unwrap();
        let pts = datatree.fmm.tree().get_points(leaf).unwrap();

        let leaf_coordinates = pts
            .iter()
            .map(|p| p.coordinate)
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();

        let ntargets = leaf_coordinates.len() / datatree.fmm.kernel.space_dimension();

        let leaf_coordinates = unsafe {
            rlst_pointer_mat!['static, f64, leaf_coordinates.as_ptr(), (ntargets, datatree.fmm.kernel.space_dimension()), (datatree.fmm.kernel.space_dimension(), 1)]
        }.eval();

        let mut direct = vec![0f64; pts.len()];
        let all_point_coordinates = points_fixture(npoints, None, None);

        let all_charges = charge_dict.into_values().collect_vec();

        let kernel = Laplace3dKernel::<f64>::default();

        kernel.evaluate_st(
            EvalType::Value,
            all_point_coordinates.data(),
            leaf_coordinates.data(),
            &all_charges[..],
            &mut direct[..],
        );

        let abs_error: f64 = potentials
            .data()
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

        assert!(rel_error <= 1e-6);
    }

    #[test]
    fn test_fmm_fft() {
        let npoints = 10000;
        let points = points_fixture(npoints, None, None);
        let global_idxs = (0..npoints).collect_vec();
        let charges = vec![1.0; npoints];

        let order = 6;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let adaptive = false;
        let ncrit = 150;
        let depth = 3;
        let kernel = Laplace3dKernel::<f64>::default();

        let tree = SingleNodeTree::new(
            points.data(),
            adaptive,
            Some(ncrit),
            Some(depth),
            &global_idxs[..],
        );

        let m2l_data_fft =
            FftFieldTranslationKiFmm::new(kernel.clone(), order, *tree.get_domain(), alpha_inner);

        let fmm = KiFmm::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data_fft);

        // Form charge dict, matching charges with their associated global indices
        let charge_dict = build_charge_dict(&global_idxs[..], &charges[..]);

        let datatree = FmmData::new(fmm, &charge_dict);

        datatree.run(Some(true));

        let leaf = &datatree.fmm.tree.get_leaves().unwrap()[0];

        let potentials = datatree.potentials.get(leaf).unwrap().lock().unwrap();
        let pts = datatree.fmm.tree().get_points(leaf).unwrap();

        let leaf_coordinates = pts
            .iter()
            .map(|p| p.coordinate)
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();

        let ntargets = leaf_coordinates.len() / datatree.fmm.kernel.space_dimension();

        let leaf_coordinates = unsafe {
            rlst_pointer_mat!['static, f64, leaf_coordinates.as_ptr(), (ntargets, datatree.fmm.kernel.space_dimension()), (datatree.fmm.kernel.space_dimension(), 1)]
        }.eval();

        let mut direct = vec![0f64; pts.len()];
        let all_point_coordinates = points_fixture(npoints, None, None);

        let all_charges = charge_dict.into_values().collect_vec();

        let kernel = Laplace3dKernel::<f64>::default();

        kernel.evaluate_st(
            EvalType::Value,
            all_point_coordinates.data(),
            leaf_coordinates.data(),
            &all_charges[..],
            &mut direct[..],
        );

        let abs_error: f64 = potentials
            .data()
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());
        assert!(rel_error <= 1e-6);
    }
}
