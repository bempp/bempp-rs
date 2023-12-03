//! kiFMM based on simple linear data structures that minimises memory allocations, maximises cache re-use.
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

use crate::{
    constants::P2M_MAX_CHUNK_SIZE,
    types::{FmmDataLinear, KiFmmLinear},
};
use rlst::{
    common::traits::*,
    dense::{rlst_col_vec, rlst_pointer_mat, traits::*, Dot, MultiplyAdd, VectorContainer},
};

/// Euclidean algorithm to find greatest common divisor less than max
fn find_chunk_size(n: usize, max_chunk_size: usize) -> usize {
    let max_divisor = max_chunk_size;
    for divisor in (1..=max_divisor).rev() {
        if n % divisor == 0 {
            return divisor;
        }
    }
    1 // If no divisor is found greater than 1, return 1 as the GCD
}

impl<T, U, V> SourceTranslation for FmmDataLinear<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
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
        // Iterate over sibling sets
        if let Some(leaves) = self.fmm.tree().get_all_leaves() {
            let nleaves = leaves.len();
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let surface_size = ncoeffs * self.fmm.kernel.space_dimension();

            let mut check_potentials = rlst_col_vec![V, nleaves * ncoeffs];
            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
            let dim = self.fmm.kernel.space_dimension();

            check_potentials
                .data_mut()
                .par_chunks_exact_mut(ncoeffs)
                // .enumerate()
                .zip(self.leaf_upward_surfaces.par_chunks_exact(surface_size))
                .zip(&self.charge_index_pointer)
                .for_each(
                    |((check_potential, upward_check_surface), charge_index_pointer)| {
                        let charges = &self.charges[charge_index_pointer.0..charge_index_pointer.1];
                        let coordinates = &coordinates
                            [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];

                        let nsources = coordinates.len() / dim;

                        let coordinates = unsafe {
                            rlst_pointer_mat!['a, V, coordinates.as_ptr(), (nsources, dim), (dim, 1)]
                        }.eval();

                        if nsources > 0 {
                            self.fmm.kernel.evaluate_st(
                                EvalType::Value,
                                coordinates.data(),
                                upward_check_surface,
                                charges,
                                check_potential,
                            );
                        }
                    },
                );

            // Now compute the multipole expansions, with each of chunk_size boxes at a time.
            let chunk_size = find_chunk_size(nleaves, P2M_MAX_CHUNK_SIZE);

            check_potentials
                .data()
                .par_chunks_exact(ncoeffs*chunk_size)
                .zip(self.leaf_multipoles.par_chunks_exact(chunk_size))
                .zip(self.scales.par_chunks_exact(ncoeffs*chunk_size))
                .for_each(|((check_potential, multipole_ptrs), scale)| {

                    let check_potential = unsafe { rlst_pointer_mat!['a, V, check_potential.as_ptr(), (ncoeffs, chunk_size), (1, ncoeffs)] };
                    let scale = unsafe {rlst_pointer_mat!['a, V, scale.as_ptr(), (ncoeffs, chunk_size), (1, ncoeffs)]}.eval();

                    let tmp = (self.fmm.uc2e_inv_1.dot(&self.fmm.uc2e_inv_2.dot(&check_potential.cmp_wise_product(&scale)))).eval();

                    unsafe {
                        for (i, multipole_ptr) in multipole_ptrs.iter().enumerate().take(chunk_size) {
                            let mut ptr = multipole_ptr.raw;
                            for j in 0..ncoeffs {
                                *ptr += tmp.data()[i*ncoeffs+j];
                                ptr = ptr.add(1);
                            }
                        }
                    }
                })
        }
    }

    /// Multipole to multipole translations, multithreaded over all boxes at a given level.
    fn m2m<'a>(&self, level: u64) {
        if let Some(sources) = self.fmm.tree().get_keys(level) {
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let nsources = sources.len();
            let min = &sources[0];
            let max = &sources[nsources - 1];
            let min_idx = self.fmm.tree().key_to_index.get(min).unwrap();
            let max_idx = self.fmm.tree().key_to_index.get(max).unwrap();

            let multipoles = &self.multipoles[min_idx * ncoeffs..(max_idx + 1) * ncoeffs];

            let nsiblings = 8;
            let mut max_chunk_size = 8_i32.pow((level - 1).try_into().unwrap()) as usize;

            if max_chunk_size > P2M_MAX_CHUNK_SIZE {
                max_chunk_size = P2M_MAX_CHUNK_SIZE;
            }
            let chunk_size = find_chunk_size(nsources, max_chunk_size);

            multipoles
                .par_chunks_exact(nsiblings * ncoeffs*chunk_size)
                .zip(self.level_multipoles[(level - 1) as usize].par_chunks_exact(chunk_size))
                .for_each(|(multipole_chunk, parent)| {

                    unsafe {
                        let tmp = rlst_pointer_mat!['a, V, multipole_chunk.as_ptr(), (ncoeffs*nsiblings, chunk_size), (1, ncoeffs*nsiblings)];
                        let tmp = self.fmm.m2m.dot(&tmp).eval();

                        for (i, par) in parent.iter().enumerate().take(chunk_size) {
                            let mut ptr = par.raw;
                            for j in 0..ncoeffs {
                                *ptr += tmp.data()[(i*ncoeffs)+j];
                                ptr = ptr.add(1)
                            }
                        }
                    }
                })
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use itertools::Itertools;

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
        let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data_svd);

        // Form charge dict, matching charges with their associated global indices
        let charge_dict = build_charge_dict(&global_idxs[..], &charges[..]);

        // Associate data with the FMM
        let datatree = FmmDataLinear::new(fmm, &charge_dict).unwrap();

        // Upward pass
        {
            datatree.p2m();

            for level in (1..=depth).rev() {
                datatree.m2m(level);
            }
        }

        let midx = datatree.fmm.tree().key_to_index.get(&ROOT).unwrap();
        let ncoeffs = datatree.fmm.m2l.ncoeffs(datatree.fmm.order);
        let multipole = &datatree.multipoles[midx * ncoeffs..(midx + 1) * ncoeffs];

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
            multipole,
            &mut found,
        );

        let abs_error = (expected[0] - found[0]).abs();
        let rel_error = abs_error / expected[0];
        assert!(rel_error <= 1e-5);
    }
}
