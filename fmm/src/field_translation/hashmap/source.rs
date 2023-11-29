//! Implementation of Source and Target translations, as well as Source to Target translation.
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use itertools::Itertools;
use num::Float;
use rayon::prelude::*;

use bempp_traits::{
    field::FieldTranslationData,
    fmm::{Fmm, SourceTranslation},
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
    types::EvalType,
};
use bempp_tree::types::single_node::SingleNodeTree;

use rlst::{
    common::traits::*,
    dense::{rlst_col_vec, rlst_pointer_mat, traits::*, Dot, MultiplyAdd, VectorContainer},
};

use crate::types::{FmmData, KiFmmHashMap};

impl<T, U, V> SourceTranslation for FmmData<KiFmmHashMap<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V> + std::marker::Send + std::marker::Sync,
    U: FieldTranslationData<T> + std::marker::Sync + std::marker::Send,
    V: Scalar<Real = V> + Float + Default + std::marker::Sync + std::marker::Send,
    V: MultiplyAdd<
        V,
        VectorContainer<V>,
        VectorContainer<V>,
        VectorContainer<V>,
        Dynamic,
        Dynamic,
        Dynamic,
    >,
{
    /// Point to multipole evaluations, multithreaded over each leaf box.
    fn p2m<'a>(&self) {
        if let Some(leaves) = self.fmm.tree().get_all_leaves() {
            leaves.par_iter().enumerate().for_each(move |(i, &leaf)| {

                let leaf_multipole_arc = Arc::clone(self.multipoles.get(&leaf).unwrap());

                if let Some(leaf_points) = self.points.get(&leaf) {
                    let leaf_charges_arc = Arc::clone(self.charges.get(&leaf).unwrap());

                    // Lookup data
                    let leaf_coordinates = leaf_points
                        .iter()
                        .map(|p| p.coordinate)
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();

                    let nsources = leaf_coordinates.len() / self.fmm.kernel.space_dimension();

                    let leaf_coordinates = unsafe {
                        rlst_pointer_mat!['a, V, leaf_coordinates.as_ptr(), (nsources, self.fmm.kernel.space_dimension()), (self.fmm.kernel.space_dimension(), 1)]
                    }.eval();

                    let upward_check_surface = leaf.compute_surface(
                        &self.fmm.tree().domain,
                        self.fmm.order,
                        self.fmm.alpha_outer,
                    );
                    let ntargets = upward_check_surface.len() / self.fmm.kernel.space_dimension();

                    let leaf_charges = leaf_charges_arc.deref();

                    // Calculate check potential
                    let mut check_potential = rlst_col_vec![V, ntargets];

                    self.fmm.kernel.evaluate_st(
                        EvalType::Value,
                        leaf_coordinates.data(),
                        &upward_check_surface[..],
                        &leaf_charges[..],
                        check_potential.data_mut(),
                    );

                    let mut tmp = self.fmm.uc2e_inv_1.dot(&self.fmm.uc2e_inv_2.dot(&check_potential)).eval();
                    tmp.data_mut().iter_mut().for_each(|d| *d  *= self.fmm.kernel.scale(leaf.level()));
                    let leaf_multipole_owned = tmp;
                    let mut leaf_multipole_lock = leaf_multipole_arc.lock().unwrap();
                    *leaf_multipole_lock.deref_mut() = (leaf_multipole_lock.deref() + leaf_multipole_owned).eval();
                }
            });
        }
    }

    /// Multipole to multipole translations, multithreaded over all boxes at a given level.
    fn m2m<'a>(&self, level: u64) {
        // Parallelise over nodes at a given level
        if let Some(sources) = self.fmm.tree().get_keys(level) {
            sources.par_iter().for_each(move |&source| {
                let operator_index = source.siblings().iter().position(|&x| x == source).unwrap();
                let source_multipole_arc = Arc::clone(self.multipoles.get(&source).unwrap());
                let target_multipole_arc =
                    Arc::clone(self.multipoles.get(&source.parent()).unwrap());

                let source_multipole_lock = source_multipole_arc.lock().unwrap();

                let target_multipole_owned =
                    self.fmm.m2m[operator_index].dot(&source_multipole_lock);

                let mut target_multipole_lock = target_multipole_arc.lock().unwrap();

                *target_multipole_lock.deref_mut() =
                    (target_multipole_lock.deref() + target_multipole_owned).eval();
            })
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::charge::build_charge_dict;
    use bempp_field::types::SvdFieldTranslationKiFmm;
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::{constants::ROOT, implementations::helpers::points_fixture};

    #[test]
    fn test_upward_pass() {
        let npoints = 10000;
        let points = points_fixture(npoints, None, None);
        let global_idxs = (0..npoints).collect_vec();
        let charges = vec![1.0; npoints];

        let kernel = Laplace3dKernel::<f64>::default();
        let order = 6;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let adaptive = false;
        let k = 1000;
        let ncrit = 150;
        let depth = 3;

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
        let fmm = KiFmmHashMap::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data_svd);

        // Form charge dict, matching charges with their associated global indices
        let charge_dict = build_charge_dict(&global_idxs[..], &charges[..]);

        // Associate data with the FMM
        let datatree = FmmData::new(fmm, &charge_dict);

        // Upward pass
        {
            datatree.p2m();

            for level in (1..=depth).rev() {
                datatree.m2m(level);
            }
        }

        let multipole = datatree.multipoles.get(&ROOT).unwrap();

        let surface =
            ROOT.compute_surface(&datatree.fmm.tree().domain, order, datatree.fmm.alpha_inner);

        let test_point = vec![100000., 0., 0.];

        let mut expected = vec![0.];
        let mut found = vec![0.];

        let kernel = Laplace3dKernel::<f64>::default();
        kernel.evaluate_st(
            EvalType::Value,
            points.data(),
            &test_point,
            &charges,
            &mut expected,
        );

        kernel.evaluate_st(
            EvalType::Value,
            &surface,
            &test_point,
            &multipole.lock().unwrap().data(),
            &mut found,
        );

        let abs_error = (expected[0] - found[0]).abs();
        let rel_error = abs_error / expected[0];
        assert!(rel_error <= 1e-5);
    }
}
