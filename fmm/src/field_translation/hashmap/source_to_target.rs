//! Implementation of Source and Target translations, as well as Source to Target translation.
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
    fmm::{Fmm, InteractionLists},
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use rlst::{
    algorithms::{linalg::DenseMatrixLinAlgBuilder, traits::svd::Svd},
    common::traits::*,
    dense::{
        rlst_col_vec, rlst_dynamic_mat, rlst_pointer_mat, traits::*, Dot, MultiplyAdd, Shape,
        VectorContainer,
    },
};

use crate::types::{FmmDataHashmap, KiFmmHashMap, SendPtr, SendPtrMut};

/// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
impl<T, U> FieldTranslation<U>
    for FmmDataHashmap<KiFmmHashMap<SingleNodeTree<U>, T, SvdFieldTranslationKiFmm<U, T>, U>, U>
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
        let mut transfer_vector_to_m2l =
            HashMap::<usize, Arc<Mutex<Vec<(MortonKey, MortonKey)>>>>::new();

        for tv in self.fmm.m2l.transfer_vectors.iter() {
            transfer_vector_to_m2l.insert(tv.hash, Arc::new(Mutex::new(Vec::new())));
        }

        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

        targets.par_iter().enumerate().for_each(|(_i, &target)| {
            if let Some(v_list) = self.fmm.get_v_list(&target) {
                let calculated_transfer_vectors = v_list
                    .iter()
                    .map(|source| target.find_transfer_vector(source))
                    .collect::<Vec<usize>>();
                for (transfer_vector, &source) in
                    calculated_transfer_vectors.iter().zip(v_list.iter())
                {
                    let m2l_arc = Arc::clone(transfer_vector_to_m2l.get(transfer_vector).unwrap());
                    let mut m2l_lock = m2l_arc.lock().unwrap();
                    m2l_lock.push((source, target));
                }
            }
        });

        let mut transfer_vector_to_m2l_rw_lock =
            HashMap::<usize, Arc<RwLock<Vec<(MortonKey, MortonKey)>>>>::new();

        // Find all multipole expansions and allocate
        for (&transfer_vector, m2l_arc) in transfer_vector_to_m2l.iter() {
            transfer_vector_to_m2l_rw_lock.insert(
                transfer_vector,
                Arc::new(RwLock::new(m2l_arc.lock().unwrap().clone())),
            );
        }

        transfer_vector_to_m2l_rw_lock
            .par_iter()
            .for_each(|(transfer_vector, m2l_arc)| {
                let c_idx = self
                    .fmm
                    .m2l
                    .transfer_vectors
                    .iter()
                    .position(|x| x.hash == *transfer_vector)
                    .unwrap();

                let (nrows, _) = self.fmm.m2l.operator_data.c.shape();
                let top_left = (0, c_idx * self.fmm.m2l.k);
                let dim = (nrows, self.fmm.m2l.k);

                let c_sub = self.fmm.m2l.operator_data.c.block(top_left, dim);

                let m2l_rw = m2l_arc.read().unwrap();
                let mut multipoles = rlst_dynamic_mat![U, (self.fmm.m2l.k, m2l_rw.len())];

                for (i, (source, _)) in m2l_rw.iter().enumerate() {
                    let source_multipole_arc = Arc::clone(self.multipoles.get(source).unwrap());
                    let source_multipole_lock = source_multipole_arc.lock().unwrap();

                    // Compressed multipole
                    let compressed_source_multipole_owned = self
                        .fmm
                        .m2l
                        .operator_data
                        .st_block
                        .dot(&source_multipole_lock)
                        .eval();

                    let first = i * self.fmm.m2l.k;
                    let last = first + self.fmm.m2l.k;

                    let multipole_slice = multipoles.get_slice_mut(first, last);
                    multipole_slice.copy_from_slice(compressed_source_multipole_owned.data());
                }

                // Compute convolution
                let compressed_check_potential_owned = c_sub.dot(&multipoles);

                // Post process to find check potential
                let check_potential_owned = self
                    .fmm
                    .m2l
                    .operator_data
                    .u
                    .dot(&compressed_check_potential_owned)
                    .eval();

                let mut tmp = self
                    .fmm
                    .dc2e_inv_1
                    .dot(&self.fmm.dc2e_inv_2.dot(&check_potential_owned));
                tmp.data_mut()
                    .iter_mut()
                    .for_each(|d| *d *= self.fmm.kernel.scale(level) * self.m2l_scale(level));
                let locals_owned = tmp;

                // Assign locals
                for (i, (_, target)) in m2l_rw.iter().enumerate() {
                    let target_local_arc = Arc::clone(self.locals.get(target).unwrap());
                    let mut target_local_lock = target_local_arc.lock().unwrap();

                    let top_left = (0, i);
                    let dim = (ncoeffs, 1);
                    let target_local_owned = locals_owned.block(top_left, dim);

                    *target_local_lock.deref_mut() =
                        (target_local_lock.deref() + target_local_owned).eval();
                }
            });
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

/// Implement the multipole to local translation operator for an FFT accelerated KiFMM on a single node.
impl<T, U> FieldTranslation<U>
    for FmmDataHashmap<KiFmmHashMap<SingleNodeTree<U>, T, FftFieldTranslationKiFmm<U, T>, U>, U>
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

        // Form signals to use for convolution first
        let n = 2 * self.fmm.order - 1;
        let ntargets = targets.len();

        // Pad the signal
        let &(m, n, o) = &(n, n, n);

        let p = m + 1;
        let q = n + 1;
        let r = o + 1;
        let size = p * q * r;
        let size_real = p * q * (r / 2 + 1);
        let pad_size = (p - m, q - n, r - o);
        let pad_index = (p - m, q - n, r - o);
        // let mut padded_signals = rlst_col_vec![U, size * ntargets];
        let mut padded_signals = vec![U::zero(); size * ntargets];

        // let chunks = padded_signals.data_mut().par_chunks_exact_mut(size);
        let chunks = padded_signals.par_chunks_exact_mut(size);

        let range = (0..chunks.len()).into_par_iter();
        range.zip(chunks).for_each(|(i, chunk)| {
            let target = targets[i];
            let source_multipole_arc = Arc::clone(self.multipoles.get(&target).unwrap());
            let source_multipole_lock = source_multipole_arc.lock().unwrap();
            let signal = self
                .fmm
                .m2l
                .compute_signal(self.fmm.order, source_multipole_lock.data());

            let padded_signal = pad3(&signal, pad_size, pad_index);

            chunk.copy_from_slice(padded_signal.get_data());
        });
        // let mut padded_signals_hat = rlst_col_vec![Complex<U>, size_real * ntargets];
        let mut padded_signals_hat = vec![Complex::<U>::default(); size_real * ntargets];

        U::rfft3_fftw_par_vec(&mut padded_signals, &mut padded_signals_hat, &[p, q, r]);

        let kernel_data_halo = &self.fmm.m2l.operator_data.kernel_data_rearranged;
        let ntargets = targets.len();
        let nparents = ntargets / 8;
        // let mut global_check_potentials_hat = rlst_col_vec![Complex<U>, size_real * ntargets];
        // let mut global_check_potentials = rlst_col_vec![U, size * ntargets];
        let mut global_check_potentials_hat = vec![Complex::<U>::default(); size_real * ntargets];
        let mut global_check_potentials = vec![U::default(); size * ntargets];

        // Get check potentials in frequency order
        let mut global_check_potentials_hat_freq = vec![Vec::new(); size_real];

        unsafe {
            // let ptr = global_check_potentials_hat.get_pointer_mut();
            let ptr = global_check_potentials_hat.as_mut_ptr();
            for (i, elem) in global_check_potentials_hat_freq
                .iter_mut()
                .enumerate()
                .take(size_real)
            {
                for j in 0..ntargets {
                    let raw = ptr.offset((j * size_real + i).try_into().unwrap());
                    let send_ptr = SendPtrMut { raw };
                    elem.push(send_ptr);
                }
            }
        }

        // Get signals into frequency order
        let mut padded_signals_hat_freq = vec![Vec::new(); size_real];
        let zero = rlst_col_vec![Complex<U>, 8];
        unsafe {
            // let ptr = padded_signals_hat.get_pointer();
            let ptr = padded_signals_hat.as_ptr();

            for (i, elem) in padded_signals_hat_freq
                .iter_mut()
                .enumerate()
                .take(size_real)
            {
                for j in 0..ntargets {
                    let raw = ptr.offset((j * size_real + i).try_into().unwrap());
                    let send_ptr = SendPtr { raw };
                    elem.push(send_ptr);
                }
                // put in a bunch of zeros at the end
                let ptr = zero.get_pointer();
                for _ in 0..8 {
                    let send_ptr = SendPtr { raw: ptr };
                    elem.push(send_ptr)
                }
            }
        }

        // Create a map between targets and index positions in vec of len 'ntargets'
        let mut target_map = HashMap::new();

        for (i, t) in targets.iter().enumerate() {
            target_map.insert(t, i);
        }

        // Find all the displacements used for saving results
        let mut all_displacements = Vec::new();
        targets.chunks_exact(8).for_each(|sibling_chunk| {
            // not in Morton order (refer to sort method when called on 'neighbours')
            let parent_neighbours: Vec<Option<MortonKey>> =
                sibling_chunk[0].parent().all_neighbors();

            let displacements = parent_neighbours
                .iter()
                .map(|pn| {
                    let mut tmp = Vec::new();
                    if let Some(pn) = pn {
                        if self.fmm.tree.keys_set.contains(pn) {
                            let mut children = pn.children();
                            children.sort();
                            for child in children {
                                // tmp.push(*target_map.get(&child).unwrap() as i64)
                                tmp.push(*target_map.get(&child).unwrap())
                            }
                        } else {
                            for i in 0..8 {
                                tmp.push(ntargets + i)
                            }
                        }
                    } else {
                        for i in 0..8 {
                            tmp.push(ntargets + i)
                        }
                    }

                    assert!(tmp.len() == 8);
                    tmp
                })
                .collect_vec();
            all_displacements.push(displacements);
        });

        let scale = Complex::from(self.m2l_scale(level));

        (0..size_real).into_par_iter().for_each(|freq| {
            // Extract frequency component of signal (ntargets long)
            let padded_signal_freq = &padded_signals_hat_freq[freq];

            // Extract frequency components of save locations (ntargets long)
            let check_potential_freq = &global_check_potentials_hat_freq[freq];

            (0..nparents).for_each(|sibling_idx| {
                // lookup associated save locations for our current sibling set
                let save_locations =
                    &check_potential_freq[(sibling_idx * 8)..(sibling_idx + 1) * 8];
                let save_locations_raw = save_locations.iter().map(|s| s.raw).collect_vec();

                // for each halo position compute convolutions to a given sibling set
                for (i, kernel_data) in kernel_data_halo.iter().enumerate().take(26) {
                    let frequency_offset = 64 * freq;
                    let kernel_data_i = &kernel_data[frequency_offset..(frequency_offset + 64)];

                    // Find displacements for signal being translated
                    let displacements = &all_displacements[sibling_idx][i];

                    // Lookup signal to be translated if a translation is to be performed
                    let signal = &padded_signal_freq[(displacements[0])..=(displacements[7])];
                    for j in 0..8 {
                        let kernel_data_ij = &kernel_data_i[j * 8..(j + 1) * 8];
                        let sig = signal[j].raw;
                        unsafe {
                            save_locations_raw
                                .iter()
                                .zip(kernel_data_ij.iter())
                                .for_each(|(&sav, &ker)| *sav += scale * ker * *sig)
                        }
                    } // inner loop
                }
            }); // over each sibling set
        });

        U::irfft_fftw_par_vec(
            &mut global_check_potentials_hat,
            &mut global_check_potentials,
            &[p, q, r],
        );

        // Compute local expansion coefficients and save to data tree
        let (_, multi_indices) = MortonKey::surface_grid::<U>(self.fmm.order);

        let check_potentials = global_check_potentials
            // .data()
            .chunks_exact(size)
            .flat_map(|chunk| {
                let m = 2 * self.fmm.order - 1;
                let p = m + 1;
                let mut potentials = Array3D::new((p, p, p));
                potentials.get_data_mut().copy_from_slice(chunk);

                let mut tmp = Vec::new();
                let ntargets = multi_indices.len() / 3;
                let xs = &multi_indices[0..ntargets];
                let ys = &multi_indices[ntargets..2 * ntargets];
                let zs = &multi_indices[2 * ntargets..];

                for i in 0..ntargets {
                    let val = potentials.get(zs[i], ys[i], xs[i]).unwrap();
                    tmp.push(*val);
                }
                tmp
            })
            .collect_vec();

        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
        let check_potentials = unsafe {
            rlst_pointer_mat!['a, U, check_potentials.as_ptr(), (ncoeffs, ntargets), (1, ncoeffs)]
        };

        let mut tmp = self
            .fmm
            .dc2e_inv_1
            .dot(&self.fmm.dc2e_inv_2.dot(&check_potentials))
            .eval();
        tmp.data_mut()
            .iter_mut()
            .for_each(|d| *d *= self.fmm.kernel.scale(level));
        let locals = tmp;

        for (i, target) in targets.iter().enumerate() {
            let target_local_arc = Arc::clone(self.locals.get(target).unwrap());
            let mut target_local_lock = target_local_arc.lock().unwrap();

            let top_left = (0, i);
            let dim = (ncoeffs, 1);
            let target_local_owned = locals.block(top_left, dim);

            *target_local_lock.deref_mut() =
                (target_local_lock.deref() + target_local_owned).eval();
        }
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
