//! Multipole to Local field translations for uniform and adaptive Kernel Indepenent FMMs
use itertools::Itertools;
use num::Float;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use bempp_traits::{
    field::{FieldTranslation, FieldTranslationData},
    fmm::Fmm,
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
    types::Scalar,
};
use bempp_tree::types::single_node::SingleNodeTree;

use crate::types::{FmmDataUniform, KiFmmLinear};

use rlst_dense::{
    array::{empty_array, Array},
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_array_from_slice2, rlst_dynamic_array2,
    traits::{MatrixSvd, MultIntoResize, RawAccess, RawAccessMut},
};

use rlst_dense::traits::Shape;



/// Field translations for uniformly refined trees that take matrix input for charges.
pub mod matrix {
    use bempp_field::types::SvdFieldTranslationKiFmmRcmp;

    use crate::types::{FmmDataUniformMatrix, KiFmmLinearMatrix, SendPtrMut};

    use super::*;

    impl<T, U>
        FmmDataUniformMatrix<
            KiFmmLinearMatrix<SingleNodeTree<U>, T, SvdFieldTranslationKiFmmRcmp<U, T>, U>,
            U,
        >
    where
        T: Kernel<T = U>
            + ScaleInvariantKernel<T = U>
            + std::marker::Send
            + std::marker::Sync
            + Default,
        U: Scalar<Real = U> + rlst_blis::interface::gemm::Gemm,
        U: Float + Default,
        U: std::marker::Send + std::marker::Sync + Default,
        Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    {
        fn displacements(&self, level: u64) -> Vec<Mutex<Vec<i64>>> {
            let sources = self.fmm.tree().get_keys(level).unwrap();
            let nsources = sources.len();

            let all_displacements = vec![vec![-1i64; nsources]; 316];
            let all_displacements = all_displacements.into_iter().map(Mutex::new).collect_vec();

            sources.into_par_iter().enumerate().for_each(|(j, source)| {
                let v_list = source
                    .parent()
                    .neighbors()
                    .iter()
                    .flat_map(|pn| pn.children())
                    .filter(|pnc| {
                        !source.is_adjacent(pnc) && self.fmm.tree().get_all_keys_set().contains(pnc)
                    })
                    .collect_vec();

                let transfer_vectors = v_list
                    .iter()
                    .map(|target| target.find_transfer_vector(source))
                    .collect_vec();

                let mut transfer_vectors_map = HashMap::new();
                for (i, v) in transfer_vectors.iter().enumerate() {
                    transfer_vectors_map.insert(v, i);
                }

                let transfer_vectors_set: HashSet<_> = transfer_vectors.iter().cloned().collect();

                for (i, tv) in self.fmm.m2l.transfer_vectors.iter().enumerate() {
                    let mut all_displacements_lock = all_displacements[i].lock().unwrap();

                    if transfer_vectors_set.contains(&tv.hash) {
                        let target = &v_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                        let target_index = self.level_index_pointer[level as usize]
                            .get(target)
                            .unwrap();
                        all_displacements_lock[j] = *target_index as i64;
                    }
                }
            });

            all_displacements
        }
    }

    /// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
    impl<T, U> FieldTranslation<U>
        for FmmDataUniformMatrix<
            KiFmmLinearMatrix<SingleNodeTree<U>, T, SvdFieldTranslationKiFmmRcmp<U, T>, U>,
            U,
        >
    where
        T: Kernel<T = U>
            + ScaleInvariantKernel<T = U>
            + std::marker::Send
            + std::marker::Sync
            + Default,
        U: Scalar<Real = U> + rlst_blis::interface::gemm::Gemm,
        U: Float + Default,
        U: std::marker::Send + std::marker::Sync + Default,
        Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    {
        fn p2l(&self, _level: u64) {}

        fn m2l<'a>(&self, level: u64) {
            let Some(sources) = self.fmm.tree().get_keys(level) else {
                return;
            };

            let nsources = sources.len();

            let all_displacements = self.displacements(level);

            // Interpret multipoles as a matrix
            let multipoles = rlst_array_from_slice2!(
                U,
                unsafe {
                    std::slice::from_raw_parts(
                        self.level_multipoles[level as usize][0][0].raw,
                        self.ncoeffs * nsources * self.ncharge_vectors,
                    )
                },
                [self.ncoeffs, nsources * self.ncharge_vectors]
            );

            rlst_blis::interface::threading::enable_threading();
            let mut compressed_multipoles = empty_array::<U, 2>()
                .simple_mult_into_resize(self.fmm.m2l.operator_data.st_block.view(), multipoles);
            rlst_blis::interface::threading::disable_threading();

            compressed_multipoles
                .data_mut()
                .iter_mut()
                .for_each(|d| *d *= self.fmm.kernel.scale(level) * self.m2l_scale(level));

            let compressed_locals_ =
                vec![U::zero(); nsources * self.ncharge_vectors * self.fmm.m2l.k];
            let compressed_locals = rlst_array_from_slice2!(
                U,
                compressed_locals_.as_slice(),
                [self.fmm.m2l.k, nsources * self.ncharge_vectors]
            );

            let mut compressed_locals_ptrs = Vec::new();

            for (i, _target) in sources.iter().enumerate() {
                let key_displacement = i * self.fmm.m2l.k * self.ncharge_vectors;
                let mut tmp = Vec::new();
                for charge_vec_idx in 0..self.ncharge_vectors {
                    let charge_vec_displacement = charge_vec_idx * self.fmm.m2l.k;

                    let raw = unsafe {
                        compressed_locals
                            .data()
                            .as_ptr()
                            .add(key_displacement + charge_vec_displacement)
                            as *mut U
                    };
                    let send_ptr = SendPtrMut { raw };
                    tmp.push(send_ptr)
                }
                compressed_locals_ptrs.push(tmp);
            }

            let compressed_level_locals =
                compressed_locals_ptrs.iter().map(Mutex::new).collect_vec();

            let multipole_idxs = all_displacements
                .iter()
                .map(|displacements| {
                    displacements
                        .lock()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .filter(|(_, &d)| d != -1)
                        .map(|(i, _)| i)
                        .collect_vec()
                })
                .collect_vec();

            let local_idxs = all_displacements
                .iter()
                .map(|displacements| {
                    displacements
                        .lock()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .filter(|(_, &d)| d != -1)
                        .map(|(_, &j)| j as usize)
                        .collect_vec()
                })
                .collect_vec();

            (0..316)
                .into_par_iter()
                .zip(multipole_idxs)
                .zip(local_idxs)
                .for_each(|((c_idx, multipole_idxs), local_idxs)| {
                    let c_u_sub = &self.fmm.m2l.operator_data.c_u[c_idx];
                    let c_vt_sub = &self.fmm.m2l.operator_data.c_vt[c_idx];

                    let mut compressed_multipoles_subset = rlst_dynamic_array2!(
                        U,
                        [self.fmm.m2l.k, multipole_idxs.len() * self.ncharge_vectors]
                    );

                    for (local_multipole_idx, &global_multipole_idx) in
                        multipole_idxs.iter().enumerate()
                    {
                        let key_displacement_global =
                            global_multipole_idx * self.fmm.m2l.k * self.ncharge_vectors;

                        let key_displacement_local =
                            local_multipole_idx * self.fmm.m2l.k * self.ncharge_vectors;

                        for charge_vec_idx in 0..self.ncharge_vectors {
                            let charge_vec_displacement = charge_vec_idx * self.fmm.m2l.k;

                            compressed_multipoles_subset.data_mut()[key_displacement_local
                                + charge_vec_displacement
                                ..key_displacement_local
                                    + charge_vec_displacement
                                    + self.fmm.m2l.k]
                                .copy_from_slice(
                                    &compressed_multipoles.data()[key_displacement_global
                                        + charge_vec_displacement
                                        ..key_displacement_global
                                            + charge_vec_displacement
                                            + self.fmm.m2l.k],
                                );
                        }
                    }

                    let locals = empty_array::<U, 2>().simple_mult_into_resize(
                        c_u_sub.view(),
                        empty_array::<U, 2>().simple_mult_into_resize(
                            c_vt_sub.view(),
                            compressed_multipoles_subset.view(),
                        ),
                    );

                    for (local_multipole_idx, &global_local_idx) in local_idxs.iter().enumerate() {
                        let local_lock = compressed_level_locals[global_local_idx].lock().unwrap();

                        for charge_vec_idx in 0..self.ncharge_vectors {
                            let local_send_ptr = local_lock[charge_vec_idx];
                            let local_ptr = local_send_ptr.raw;
                            let local = unsafe {
                                std::slice::from_raw_parts_mut(local_ptr, self.fmm.m2l.k)
                            };

                            let key_displacement =
                                local_multipole_idx * self.fmm.m2l.k * self.ncharge_vectors;
                            let charge_vec_displacement = charge_vec_idx * self.fmm.m2l.k;

                            let result = &locals.data()[key_displacement + charge_vec_displacement
                                ..key_displacement + charge_vec_displacement + self.fmm.m2l.k];
                            local.iter_mut().zip(result).for_each(|(l, r)| *l += *r);
                        }
                    }
                });

            rlst_blis::interface::threading::enable_threading();
            let result = empty_array::<U, 2>().simple_mult_into_resize(
                self.fmm.dc2e_inv_1.view(),
                empty_array::<U, 2>().simple_mult_into_resize(
                    self.fmm.dc2e_inv_2.view(),
                    empty_array::<U, 2>().simple_mult_into_resize(
                        self.fmm.m2l.operator_data.u.view(),
                        compressed_locals,
                    ),
                ),
            );
            rlst_blis::interface::threading::disable_threading();
            let ptr = self.level_locals[level as usize][0][0].raw;
            let all_locals = unsafe {
                std::slice::from_raw_parts_mut(ptr, nsources * self.ncoeffs * self.ncharge_vectors)
            };
            all_locals
                .iter_mut()
                .zip(result.data().iter())
                .for_each(|(l, r)| *l += *r);
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
}


use std::time::Instant;

pub mod uniform {
    use std::{ops::DerefMut};

    use bempp_field::types::SvdFieldTranslationKiFmmRcmp;

    use crate::types::SendPtrMut;

    use super::*;

    impl<T, U>
        FmmDataUniform<KiFmmLinear<SingleNodeTree<U>, T, SvdFieldTranslationKiFmmRcmp<U, T>, U>, U>
    where
        T: Kernel<T = U>
            + ScaleInvariantKernel<T = U>
            + std::marker::Send
            + std::marker::Sync
            + Default,
        U: Scalar<Real = U> + rlst_blis::interface::gemm::Gemm,
        U: Float + Default,
        U: std::marker::Send + std::marker::Sync + Default,
        Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    {
        fn displacements(&self, level: u64) -> Vec<Mutex<Vec<i64>>> {
            let sources = self.fmm.tree().get_keys(level).unwrap();
            let nsources = sources.len();

            let all_displacements = vec![vec![-1i64; nsources]; 316];
            let all_displacements = all_displacements.into_iter().map(Mutex::new).collect_vec();

            sources.into_par_iter().enumerate().for_each(|(j, source)| {
                let v_list = source
                    .parent()
                    .neighbors()
                    .iter()
                    .flat_map(|pn| pn.children())
                    .filter(|pnc| {
                        !source.is_adjacent(pnc) && self.fmm.tree().get_all_keys_set().contains(pnc)
                    })
                    .collect_vec();

                let transfer_vectors = v_list
                    .iter()
                    .map(|target| target.find_transfer_vector(source))
                    .collect_vec();

                let mut transfer_vectors_map = HashMap::new();
                for (i, v) in transfer_vectors.iter().enumerate() {
                    transfer_vectors_map.insert(v, i);
                }

                let transfer_vectors_set: HashSet<_> = transfer_vectors.iter().cloned().collect();

                for (i, tv) in self.fmm.m2l.transfer_vectors.iter().enumerate() {
                    let mut all_displacements_lock = all_displacements[i].lock().unwrap();

                    if transfer_vectors_set.contains(&tv.hash) {
                        let target = &v_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                        let target_index = self.level_index_pointer[level as usize]
                            .get(target)
                            .unwrap();
                        all_displacements_lock[j] = *target_index as i64;
                    }
                }
            });

            all_displacements
        }
    }

    /// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
    impl<T, U> FieldTranslation<U>
        for FmmDataUniform<
            KiFmmLinear<SingleNodeTree<U>, T, SvdFieldTranslationKiFmmRcmp<U, T>, U>,
            U,
        >
    where
        T: Kernel<T = U>
            + ScaleInvariantKernel<T = U>
            + std::marker::Send
            + std::marker::Sync
            + Default,
        U: Scalar<Real = U> + rlst_blis::interface::gemm::Gemm,
        U: Float + Default,
        U: std::marker::Send + std::marker::Sync + Default,
        Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    {
        fn p2l(&self, _level: u64) {}

        fn m2l<'a>(&self, level: u64) {
            let Some(sources) = self.fmm.tree().get_keys(level) else {
                return;
            };

            let mut s = Instant::now();
            let nsources = sources.len();

            /////
            let mut flops: usize = 0;
            let mut accesses = 0;
            let mut bytes = 0;
            let profile = true;
            let size_of_U = std::mem::size_of::<U>();
            let flops_mutex = Mutex::new(flops);
            let bytes_mutex = Mutex::new(bytes);
            ///////


            let all_displacements = self.displacements(level);

            // Interpret multipoles as a matrix
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let multipoles = rlst_array_from_slice2!(
                U,
                unsafe {
                    std::slice::from_raw_parts(
                        self.level_multipoles[level as usize][0].raw,
                        ncoeffs * nsources,
                    )
                },
                [ncoeffs, nsources]
            );

            *bytes_mutex.lock().unwrap() += ncoeffs * nsources * size_of_U;

            rlst_blis::interface::threading::enable_threading();
            let mut compressed_multipoles = empty_array::<U, 2>()
                .simple_mult_into_resize(self.fmm.m2l.operator_data.st_block.view(), multipoles);
            rlst_blis::interface::threading::disable_threading();

            // Computing compressed multipoles
            let [k, _] = self.fmm.m2l.operator_data.st_block.shape();
            *flops_mutex.lock().unwrap() += k * (2*ncoeffs-1) * nsources; // matmul
            *bytes_mutex.lock().unwrap() += (k*ncoeffs + ncoeffs * nsources) * size_of_U; // reads
            *bytes_mutex.lock().unwrap() += (k * nsources) * size_of_U; // writes

            compressed_multipoles
                .data_mut()
                .iter_mut()
                .for_each(|d| *d *= self.fmm.kernel.scale(level) * self.m2l_scale(level));

            // Scaling compressed multipoles
            *flops_mutex.lock().unwrap() += compressed_multipoles.data().len(); // pointwise mult
            *bytes_mutex.lock().unwrap() += compressed_multipoles.data().len() * size_of_U; // reads and writes in place

            let multipole_idxs = all_displacements
                .iter()
                .map(|displacements| {
                    displacements
                        .lock()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .filter(|(_, &d)| d != -1)
                        .map(|(i, _)| i)
                        .collect_vec()
                })
                .collect_vec();

            {
                let size_of_displacement_ptr = nsources; // Replace with actual type
                let size_of_displacement_element = std::mem::size_of::<i64>(); // Replace with actual type
                let size_of_index = std::mem::size_of::<usize>(); // Assuming usize for indices

                // For iterating over all_displacements
                // bytes += all_displacements.len() * size_of_displacement_ptr;

                // For iterating over elements within each displacement
                // for displacement in all_displacements.iter() {
                //     bytes += displacement.lock().unwrap().len() * size_of_displacement_element;
                // }
                // // For the map operation + inner collection
                // for idx in multipole_idxs.iter() {
                //     bytes +=  idx.len() * size_of_index;
                // }
                // For the outer collection of multipole idxs
                // bytes += all_displacements.len() * size_of_index;
            }

            let local_idxs = all_displacements
                .iter()
                .map(|displacements| {
                    displacements
                        .lock()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .filter(|(_, &d)| d != -1)
                        .map(|(_, &j)| j as usize)
                        .collect_vec()
                })
                .collect_vec();

            // {
            //     let size_of_displacement_ptr = nsources; // Replace with actual type
            //     let size_of_displacement_element = std::mem::size_of::<i64>(); // Replace with actual type
            //     let size_of_index = std::mem::size_of::<usize>(); // Assuming usize for indices
            //     // For iterating over all_displacements
            //     // bytes += all_displacements.len() * size_of_displacement_ptr;

            //     // For iterating over elements within each displacement
            //     for displacement in all_displacements.iter() {
            //         bytes += displacement.lock().unwrap().len() * size_of_displacement_element;
            //     }
            //     // For the map operation + inner collection
            //     for idx in multipole_idxs.iter() {
            //         bytes +=  idx.len() * size_of_index;
            //     }
            //     // For the outer collection of multipole idxs
            //     // bytes += all_displacements.len() * size_of_index;
            // }


            let compressed_locals_ = vec![U::zero(); nsources * self.fmm.m2l.k];
            let compressed_locals = rlst_array_from_slice2!(
                U,
                compressed_locals_.as_slice(),
                [self.fmm.m2l.k, nsources]
            );
            let mut compressed_locals_ptrs = Vec::new();

            for (i, _target) in sources.iter().enumerate() {
                let raw =
                    unsafe { compressed_locals.data().as_ptr().add(i * self.fmm.m2l.k) as *mut U };
                let send_ptr = SendPtrMut { raw };
                compressed_locals_ptrs.push(send_ptr);
            }

            let compressed_level_locals =
                compressed_locals_ptrs.iter().map(Mutex::new).collect_vec();

            // {
            //     let size_of_send_ptr_mut = std::mem::size_of::<SendPtrMut<U>>();
            //     let size_of_mutex_snd_ptr_mut = std::mem::size_of::<Mutex<SendPtrMut<U>>>();

            //     // For creating compressed_locals from a slice
            //     bytes += nsources * self.fmm.m2l.k * size_of_U;

            //     // For creating send pointers
            //     bytes += nsources * size_of_send_ptr_mut;

            //     // For creating mutexes
            //     bytes += nsources * size_of_mutex_snd_ptr_mut;
            // }




            (0..316)
                .into_par_iter()
                .zip(multipole_idxs)
                .zip(local_idxs)
                .for_each(|((c_idx, multipole_idxs), local_idxs)| {
                    let c_u_sub = &self.fmm.m2l.operator_data.c_u[c_idx];
                    let c_vt_sub = &self.fmm.m2l.operator_data.c_vt[c_idx];

                    let mut compressed_multipoles_subset =
                        rlst_dynamic_array2!(U, [self.fmm.m2l.k, multipole_idxs.len()]);

                    for (i, &multipole_idx) in multipole_idxs.iter().enumerate() {
                        compressed_multipoles_subset.data_mut()
                            [i * self.fmm.m2l.k..(i + 1) * self.fmm.m2l.k]
                            .copy_from_slice(
                                &compressed_multipoles.data()[(multipole_idx) * self.fmm.m2l.k
                                    ..(multipole_idx + 1) * self.fmm.m2l.k],
                            );
                    }

                    // For loading multipole data
                    {
                        let mut bytes_lock = bytes_mutex.lock().unwrap();
                        *bytes_lock.deref_mut() += multipole_idxs.len() * self.fmm.m2l.k * size_of_U;
                    }


                    let locals = empty_array::<U, 2>().simple_mult_into_resize(
                        c_u_sub.view(),
                        empty_array::<U, 2>().simple_mult_into_resize(
                            c_vt_sub.view(),
                            compressed_multipoles_subset.view(),
                        ),
                    );

                    {
                        // For computing compressed check potential
                        let mut flops_lock = flops_mutex.lock().unwrap();
                        let [a, b] = c_u_sub.shape();
                        let [_, c] = c_vt_sub.shape();
                        let [_, d] = compressed_multipoles_subset.shape();
                        *flops_lock += b * d * (2*c-1)+a*d*(2*b-1);

                        // For saving compressed check potential,  accessing C_t, reading multipole data
                        let mut bytes_lock = bytes_mutex.lock().unwrap();
                        *bytes_lock.deref_mut() += (c_u_sub.data().len() + c_vt_sub.data().len()  + compressed_multipoles_subset.data().len() +  locals.data().len() )* size_of_U;
                    }

                    for (multipole_idx, &local_idx) in local_idxs.iter().enumerate() {
                        let local_lock = compressed_level_locals[local_idx].lock().unwrap();
                        let local_ptr = local_lock.raw;
                        let local =
                            unsafe { std::slice::from_raw_parts_mut(local_ptr, self.fmm.m2l.k) };
                        let res = &locals.data()
                            [multipole_idx * self.fmm.m2l.k..(multipole_idx + 1) * self.fmm.m2l.k];
                        local.iter_mut().zip(res).for_each(|(l, r)| *l += *r);

                        {
                            // For saving compressed check potential
                            let mut bytes_lock = bytes_mutex.lock().unwrap();
                            *bytes_lock.deref_mut() += 2 * self.fmm.m2l.k * size_of_U;
                        }
                    }
                });

            let elapsed = s.elapsed().as_millis() as usize;

            let [a, b] = self.fmm.m2l.operator_data.u.shape();
            *flops_mutex.lock().unwrap() += a * (2*b-1) * nsources; // Computing locals
            *bytes_mutex.lock().unwrap() += (a * b + compressed_locals.data().len()) * size_of_U; // accessing post proc matrix and locals


            // Post process compressed locals
            rlst_blis::interface::threading::enable_threading();
            let result = empty_array::<U, 2>().simple_mult_into_resize(
                self.fmm.dc2e_inv_1.view(),
                empty_array::<U, 2>().simple_mult_into_resize(
                    self.fmm.dc2e_inv_2.view(),
                    empty_array::<U, 2>().simple_mult_into_resize(
                        self.fmm.m2l.operator_data.u.view(),
                        compressed_locals,
                    ),
                ),
            );
            rlst_blis::interface::threading::disable_threading();

            // Saving local data
            *flops_mutex.lock().unwrap() += result.data().len();
            *bytes_mutex.lock().unwrap() += result.data().len() * size_of_U;
            if level > 2 {
                println!("level {:?} flops {:?} bytes {:?} time {:?}", level, *flops_mutex.lock().unwrap(), *bytes_mutex.lock().unwrap(), elapsed);
            }


            let ptr = self.level_locals[level as usize][0].raw;
            let all_locals = unsafe { std::slice::from_raw_parts_mut(ptr, nsources * ncoeffs) };
            all_locals
                .iter_mut()
                .zip(result.data().iter())
                .for_each(|(l, r)| *l += *r);
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
}
