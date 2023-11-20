use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, RwLock},
};

use bempp_tools::Array3D;
use itertools::Itertools;
use num::{Complex, Float};
use rayon::prelude::*;

use bempp_field::{
    array::pad3,
    fft::Fft,
    types::{FftFieldTranslationKiFmm, FftMatrix, SvdFieldTranslationKiFmm},
};

use bempp_traits::{
    arrays::Array3DAccess,
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, InteractionLists, SourceTranslation, TargetTranslation},
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
    types::EvalType,
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use crate::types::{FmmData, FmmDataLinear, KiFmm, KiFmmLinear, SendPtr, SendPtrMut};
use rlst::{
    algorithms::{linalg::DenseMatrixLinAlgBuilder, traits::svd::Svd},
    common::traits::*,
    dense::{
        base_matrix::BaseMatrix, rlst_col_vec, rlst_dynamic_mat, rlst_pointer_mat, traits::*, Dot,
        Matrix, MultiplyAdd, Shape, VectorContainer,
    },
};
const P2M_MAX_CHUNK_SIZE: usize = 256;

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
                .enumerate()
                .zip(self.leaf_upward_surfaces.par_chunks_exact(surface_size))
                .zip(&self.charge_index_pointer)
                .for_each(
                    |(((i, check_potential), upward_check_surface), charge_index_pointer)| {
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
                        for i in 0..chunk_size {
                            let mut ptr = multipole_ptrs[i].raw;
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

                        for i in 0..chunk_size {
                            let mut ptr = parent[i].raw;
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

// Try this two different ways, ignoring w,x lists and also including them
impl<T, U, V> TargetTranslation for FmmDataLinear<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
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
    fn l2l(&self, level: u64) {}

    fn m2p<'a>(&self) {}

    fn l2p<'a>(&self) {}

    fn p2l<'a>(&self) {}

    fn p2p<'a>(&self) {}
}

/// Implement the multipole to local translation operator for an FFT accelerated KiFMM on a single node.
impl<T, U> FieldTranslation<U>
    for FmmDataLinear<KiFmmLinear<SingleNodeTree<U>, T, FftFieldTranslationKiFmm<U, T>, U>, U>
where
    T: Kernel<T = U>
        + ScaleInvariantKernel<T = U>
        + std::marker::Send
        + std::marker::Sync
        + Default,
    U: Scalar<Real = U>
        + Float
        + Default
        + std::marker::Send
        + std::marker::Sync
        + Fft<FftMatrix<U>, FftMatrix<Complex<U>>>,
    Complex<U>: Scalar,
    U: MultiplyAdd<
        U,
        VectorContainer<U>,
        VectorContainer<U>,
        VectorContainer<U>,
        Dynamic,
        Dynamic,
        Dynamic,
    >,
{
    fn m2l<'a>(&self, level: u64) {
        let Some(targets) = self.fmm.tree().get_keys(level) else {
            return;
        };
    }

    fn m2l_scale(&self, level: u64) -> U {
        if level < 2 {
            panic!("M2L only perfomed on level 2 and below")
        }

        if level == 2 {
            U::from(1. / 2.).unwrap()
        } else {
            let two = U::from(2.0).unwrap();
            Scalar::powf(two, U::from(level - 3).unwrap())
        }
    }
}

/// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
impl<T, U> FieldTranslation<U>
    for FmmDataLinear<KiFmmLinear<SingleNodeTree<U>, T, SvdFieldTranslationKiFmm<U, T>, U>, U>
where
    T: Kernel<T = U>
        + ScaleInvariantKernel<T = U>
        + std::marker::Send
        + std::marker::Sync
        + Default,
    DenseMatrixLinAlgBuilder<U>: Svd,
    U: Scalar<Real = U>,
    U: Float
        + Default
        + MultiplyAdd<
            U,
            VectorContainer<U>,
            VectorContainer<U>,
            VectorContainer<U>,
            Dynamic,
            Dynamic,
            Dynamic,
        >,
    U: std::marker::Send + std::marker::Sync + Default,
{
    fn m2l<'a>(&self, level: u64) {
        let Some(targets) = self.fmm.tree().get_keys(level) else {
            return;
        };
    }

    fn m2l_scale(&self, level: u64) -> U {
        if level < 2 {
            panic!("M2L only perfomed on level 2 and below")
        }

        if level == 2 {
            U::from(1. / 2.).unwrap()
        } else {
            let two = U::from(2.0).unwrap();
            Scalar::powf(two, U::from(level - 3).unwrap())
        }
    }
}
