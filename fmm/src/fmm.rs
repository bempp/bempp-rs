extern crate blas_src;

use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    sync::{Arc, Mutex},
    time::Instant,
};

use itertools::Itertools;
use ndarray::*;
use ndarray_linalg::SVDDC;
use rayon::prelude::*;

use bempp_traits::{
    fmm::{Fmm, FmmLoop, InteractionLists, SourceTranslation, TargetTranslation},
    kernel::Kernel,
    tree::Tree,
};
use bempp_tree::{
    constants::ROOT,
    types::{
        domain::Domain,
        morton::{MortonKey, MortonKeys},
        point::Point,
        single_node::SingleNodeTree,
    },
};

use crate::{charge::Charges, laplace::LaplaceKernel, linalg::pinv};

pub struct FmmData<T: Fmm> {
    fmm: Arc<T>,
    multipoles: HashMap<MortonKey, Arc<Mutex<Vec<f64>>>>,
    locals: HashMap<MortonKey, Arc<Mutex<Vec<f64>>>>,
    potentials: HashMap<MortonKey, Arc<Mutex<Vec<f64>>>>,
    points: HashMap<MortonKey, Vec<Point>>,
    charges: HashMap<MortonKey, Arc<Vec<f64>>>,
}

pub struct KiFmm<T: Tree, S: Kernel> {
    order: usize,

    uc2e_inv: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),

    dc2e_inv: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),

    alpha_inner: f64,
    alpha_outer: f64,

    m2m: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>>,
    l2l: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>>,
    transfer_vectors: Vec<usize>,
    m2l: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),

    tree: T,
    kernel: S,
}

#[allow(dead_code)]
impl KiFmm<SingleNodeTree, LaplaceKernel> {
    /// Scaling function for the M2L operator at a given level.
    pub fn m2l_scale(level: u64) -> f64 {
        if level < 2 {
            panic!("M2L only performed on level 2 and below")
        }

        if level == 2 {
            1. / 2.
        } else {
            2_f64.powf((level - 3) as f64)
        }
    }
    /// Algebraically defined list of unique M2L interactions, called 'transfer vectors', for 3D FMM.
    pub fn find_unique_v_list_interactions() -> (Vec<MortonKey>, Vec<MortonKey>, Vec<usize>) {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        // Encode point in centre of domain
        let key = MortonKey::from_point(&point, &domain, 3);

        // Add neighbours, and their resp. siblings to v list.
        let mut neighbours = key.neighbors();
        let mut keys: Vec<MortonKey> = Vec::new();
        keys.push(key);
        keys.append(&mut neighbours);

        for key in neighbours.iter() {
            let mut siblings = key.siblings();
            keys.append(&mut siblings);
        }

        // Keep only unique keys
        let keys: Vec<&MortonKey> = keys.iter().unique().collect();

        let mut transfer_vectors: Vec<usize> = Vec::new();
        let mut targets: Vec<MortonKey> = Vec::new();
        let mut sources: Vec<MortonKey> = Vec::new();

        for key in keys.iter() {
            // Dense v_list
            let v_list = key
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| !key.is_adjacent(pnc))
                .collect_vec();

            // Find transfer vectors for everything in dense v list of each key
            let tmp: Vec<usize> = v_list
                .iter()
                .map(|v| key.find_transfer_vector(v))
                .collect_vec();

            transfer_vectors.extend(&mut tmp.iter().cloned());
            sources.extend(&mut v_list.iter().cloned());

            let tmp_targets = vec![**key; tmp.len()];
            targets.extend(&mut tmp_targets.iter().cloned());
        }

        let mut unique_transfer_vectors = Vec::new();
        let mut unique_indices = HashSet::new();

        for (i, vec) in transfer_vectors.iter().enumerate() {
            if !unique_transfer_vectors.contains(vec) {
                unique_transfer_vectors.push(*vec);
                unique_indices.insert(i);
            }
        }

        let unique_sources: Vec<MortonKey> = sources
            .iter()
            .enumerate()
            .filter(|(i, _)| unique_indices.contains(i))
            .map(|(_, x)| *x)
            .collect_vec();

        let unique_targets: Vec<MortonKey> = targets
            .iter()
            .enumerate()
            .filter(|(i, _)| unique_indices.contains(i))
            .map(|(_, x)| *x)
            .collect_vec();

        (unique_targets, unique_sources, unique_transfer_vectors)
    }

    /// Number of coefficients related to a given expansion order.
    pub fn ncoeffs(order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }

    pub fn new(
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

        let downward_equivalent_surface = ROOT
            .compute_surface(tree.get_domain(), order, alpha_outer)
            .into_iter()
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();

        let downward_check_surface = ROOT
            .compute_surface(tree.get_domain(), order, alpha_inner)
            .into_iter()
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();

        // Compute upward check to equivalent, and downward check to equivalent Gram matrices
        // as well as their inverses using DGESVD.
        let uc2e = kernel
            .gram(&upward_equivalent_surface, &upward_check_surface)
            .unwrap();

        let dc2e = kernel
            .gram(&downward_equivalent_surface, &downward_check_surface)
            .unwrap();

        let mut m2m: Vec<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = Vec::new();
        let mut l2l: Vec<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = Vec::new();

        let nrows = KiFmm::ncoeffs(order);
        let ncols = KiFmm::ncoeffs(order);

        let uc2e = Array1::from(uc2e)
            .to_shape((nrows, ncols))
            .unwrap()
            .to_owned();

        let (a, b, c) = pinv(&uc2e);
        let uc2e_inv = (a.to_owned(), b.dot(&c).to_owned());

        let dc2e = Array1::from(dc2e)
            .to_shape((nrows, ncols))
            .unwrap()
            .to_owned();
        let (a, b, c) = pinv(&dc2e);
        let dc2e_inv = (a.to_owned(), b.dot(&c).to_owned());

        // Calculate M2M/L2L matrices
        let children = ROOT.children();

        for child in children.iter() {
            let child_upward_equivalent_surface = child
                .compute_surface(tree.get_domain(), order, alpha_inner)
                .into_iter()
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            let child_downward_check_surface = child
                .compute_surface(tree.get_domain(), order, alpha_inner)
                .into_iter()
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            let pc2ce = kernel
                .gram(&child_upward_equivalent_surface, &upward_check_surface)
                .unwrap();

            let pc2e = Array::from_shape_vec((nrows, ncols), pc2ce).unwrap();

            m2m.push(uc2e_inv.0.dot(&uc2e_inv.1.dot(&pc2e)));

            let cc2pe = kernel
                .gram(&downward_equivalent_surface, &child_downward_check_surface)
                .unwrap();
            let cc2pe = Array::from_shape_vec((ncols, nrows), cc2pe).unwrap();

            l2l.push(kernel.scale(child.level()) * dc2e_inv.0.dot(&dc2e_inv.1.dot(&cc2pe)))
        }

        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)
        let (targets, sources, transfer_vectors) = KiFmm::find_unique_v_list_interactions();

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let mut se2tc: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
            Array2::zeros((nrows, ncols * sources.len()));

        for (((i, _), source), target) in transfer_vectors
            .iter()
            .enumerate()
            .zip(sources.iter())
            .zip(targets.iter())
        {
            let source_equivalent_surface = source
                .compute_surface(tree.get_domain(), order, alpha_inner)
                .into_iter()
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            let target_check_surface = target
                .compute_surface(tree.get_domain(), order, alpha_inner)
                .into_iter()
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            let tmp_gram = kernel
                .gram(&source_equivalent_surface[..], &target_check_surface[..])
                .unwrap();

            let tmp_gram = Array::from_shape_vec((nrows, ncols), tmp_gram).unwrap();
            let lidx_sources = i * ncols;
            let ridx_sources = lidx_sources + ncols;

            se2tc
                .slice_mut(s![.., lidx_sources..ridx_sources])
                .assign(&tmp_gram);
        }

        let l: usize = 0;
        let r: usize = std::cmp::min(50, nrows);

        let (u, s, vt) = se2tc.svddc(ndarray_linalg::JobSvd::Some).unwrap();

        let u = u.unwrap().slice(s![.., l..r]).to_owned();
        let s = Array2::from_diag(&s.slice(s![l..r]));
        let vt = vt.unwrap().slice(s![l..r, ..]).to_owned();

        let m2l = (u, s, vt);

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
            transfer_vectors,
            m2l,
        }
    }
}

#[allow(dead_code)]
impl FmmData<KiFmm<SingleNodeTree, LaplaceKernel>> {
    pub fn new(fmm: KiFmm<SingleNodeTree, LaplaceKernel>, _charges: Charges) -> Self {
        let mut multipoles = HashMap::new();
        let mut locals = HashMap::new();
        let mut potentials = HashMap::new();
        let mut points = HashMap::new();
        let mut charges = HashMap::new();

        if let Some(keys) = fmm.tree().get_all_keys() {
            for key in keys.iter() {
                multipoles.insert(*key, Arc::new(Mutex::new(Vec::new())));
                locals.insert(*key, Arc::new(Mutex::new(Vec::new())));
                potentials.insert(*key, Arc::new(Mutex::new(Vec::new())));
                if let Some(point_data) = fmm.tree().get_points(key) {
                    points.insert(*key, point_data.iter().cloned().collect_vec());

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

impl SourceTranslation for FmmData<KiFmm<SingleNodeTree, LaplaceKernel>> {
    fn p2m(&self) {
        if let Some(leaves) = self.fmm.tree.get_leaves() {
            leaves.par_iter().for_each(move |&leaf| {
                let leaf_multipole_arc = Arc::clone(self.multipoles.get(&leaf).unwrap());
                let fmm_arc = Arc::clone(&self.fmm);
                let leaf_charges_arc = Arc::clone(self.charges.get(&leaf).unwrap());

                if let Some(leaf_points) = self.points.get(&leaf) {
                    // Lookup data
                    let leaf_coordinates = leaf_points
                        .iter()
                        .map(|p| p.coordinate)
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();

                    let upward_check_surface = leaf
                        .compute_surface(&fmm_arc.tree().domain, fmm_arc.order, fmm_arc.alpha_outer)
                        .into_iter()
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();

                    let leaf_charges_view = ArrayView::from(leaf_charges_arc.deref());
                    let leaf_charges_slice = leaf_charges_view.as_slice().unwrap();

                    // Calculate check potential
                    let mut check_potential =
                        vec![0.; upward_check_surface.len() / self.fmm.kernel.dim()];

                    fmm_arc.kernel.potential(
                        &leaf_coordinates[..],
                        leaf_charges_slice,
                        &upward_check_surface[..],
                        &mut check_potential[..],
                    );
                    let check_potential = Array1::from_vec(check_potential);

                    // Calculate multipole expansion
                    let leaf_multipole_owned = fmm_arc.kernel.scale(leaf.level())
                        * fmm_arc
                            .uc2e_inv
                            .0
                            .dot(&fmm_arc.uc2e_inv.1.dot(&check_potential));

                    let mut leaf_multipole_lock = leaf_multipole_arc.lock().unwrap();

                    if !leaf_multipole_lock.is_empty() {
                        leaf_multipole_lock
                            .iter_mut()
                            .zip(leaf_multipole_owned.iter())
                            .for_each(|(c, m)| *c += *m);
                    } else {
                        leaf_multipole_lock.extend(leaf_multipole_owned);
                    }
                }
            });
        }
    }

    fn m2m(&self, level: u64) {
        // Parallelise over nodes at a given level
        if let Some(sources) = self.fmm.tree.get_keys(level) {
            sources.par_iter().for_each(move |&source| {
                let source_multipole_arc = Arc::clone(self.multipoles.get(&source).unwrap());
                let source_multipole_lock = source_multipole_arc.lock().unwrap();

                if !source_multipole_lock.is_empty() {
                    let target_multipole_arc =
                        Arc::clone(self.multipoles.get(&source.parent()).unwrap());
                    let fmm_arc = Arc::clone(&self.fmm);

                    let operator_index =
                        source.siblings().iter().position(|&x| x == source).unwrap();

                    let source_multipole_view = ArrayView::from(source_multipole_lock.deref());

                    let target_multipole_owned =
                        fmm_arc.m2m[operator_index].dot(&source_multipole_view);
                    let mut target_multipole_lock = target_multipole_arc.lock().unwrap();

                    if !target_multipole_lock.is_empty() {
                        target_multipole_lock
                            .iter_mut()
                            .zip(target_multipole_owned.iter())
                            .for_each(|(c, m)| *c += *m);
                    } else {
                        target_multipole_lock.extend(target_multipole_owned);
                    }
                }
            })
        }
    }
}

impl TargetTranslation for FmmData<KiFmm<SingleNodeTree, LaplaceKernel>> {
    fn m2l(&self, level: u64) {
        if let Some(targets) = self.fmm.tree().get_keys(level) {
            targets.par_iter().for_each(move |&target| {
                let fmm_arc = Arc::clone(&self.fmm);
                let target_local_arc = Arc::clone(self.locals.get(&target).unwrap());

                // Assemble source data v list as a single BLAS operation
                if let Some(v_list) = fmm_arc.get_v_list(&target) {
                    let ncoeffs = KiFmm::ncoeffs(fmm_arc.order);
                    let mut vt =
                        Array2::<f64>::zeros((fmm_arc.m2l.2.dim().0, ncoeffs * v_list.len()));
                    let mut source_multipoles = Array1::<f64>::zeros(ncoeffs * v_list.len());

                    for (i, source) in v_list.iter().enumerate() {
                        // Locate correct components of compressed M2L matrix.
                        let transfer_vector = target.find_transfer_vector(source);

                        // Lookup appropriate M2L matrix
                        let v_idx = fmm_arc
                            .transfer_vectors
                            .iter()
                            .position(|&x| x == transfer_vector)
                            .unwrap();
                        let v_lidx = v_idx * ncoeffs;
                        let v_ridx = v_lidx + ncoeffs;
                        let vt_sub = fmm_arc.m2l.2.slice(s![.., v_lidx..v_ridx]);

                        vt.slice_mut(s![.., i * ncoeffs..(i * ncoeffs) + ncoeffs])
                            .assign(&vt_sub);

                        let source_multipole_arc = Arc::clone(self.multipoles.get(source).unwrap());
                        let source_multipole_lock = source_multipole_arc.lock().unwrap();
                        let source_multipole_view = ArrayView::from(source_multipole_lock.deref());

                        source_multipoles
                            .slice_mut(s![i * ncoeffs..(i * ncoeffs) + ncoeffs])
                            .assign(&source_multipole_view);

                        // // Compute translation
                        // let target_local_owned = KiFmm::m2l_scale(target.level())
                        //     * fmm_arc.kernel.scale(target.level())
                        //     * fmm_arc.dc2e_inv.0.dot(
                        //         &self.fmm.dc2e_inv.1.dot(
                        //             &fmm_arc
                        //                 .m2l
                        //                 .0
                        //                 .dot(&fmm_arc.m2l.1.dot(&vt_sub.dot(&source_multipole_view))),
                        //         ),
                        //     );

                        // // Store computation
                        // let mut target_local_lock = target_local_arc.lock().unwrap();

                        // if !target_local_lock.is_empty() {
                        //     target_local_lock
                        //         .iter_mut()
                        //         .zip(target_local_owned.iter())
                        //         .for_each(|(c, m)| *c += *m);
                        // } else {
                        //     target_local_lock.extend(target_local_owned);
                        // }

                    }

                    // Compute translation
                    let target_local_owned = KiFmm::m2l_scale(target.level())
                        * fmm_arc.kernel.scale(target.level())
                        * fmm_arc.dc2e_inv.0.dot(
                            &self.fmm.dc2e_inv.1.dot(
                                &fmm_arc
                                    .m2l
                                    .0
                                    .dot(&fmm_arc.m2l.1.dot(&vt.dot(&source_multipoles))),
                            ),
                        );

                    // Store computation
                    let mut target_local_lock = target_local_arc.lock().unwrap();

                    if !target_local_lock.is_empty() {
                        target_local_lock
                            .iter_mut()
                            .zip(target_local_owned.iter())
                            .for_each(|(c, m)| *c += *m);
                    } else {
                        target_local_lock.extend(target_local_owned);
                    }


                }
            })
        }
    }

    fn l2l(&self, level: u64) {
        if let Some(targets) = self.fmm.tree.get_keys(level) {
            targets.par_iter().for_each(move |&target| {
                let source_local_arc = Arc::clone(self.locals.get(&target.parent()).unwrap());
                let target_local_arc = Arc::clone(self.locals.get(&target).unwrap());
                let fmm = Arc::clone(&self.fmm);

                let operator_index = target.siblings().iter().position(|&x| x == target).unwrap();

                let source_local_lock = source_local_arc.lock().unwrap();
                let source_local_view = ArrayView::from(source_local_lock.deref());

                let target_local_owned = fmm.l2l[operator_index].dot(&source_local_view);
                let mut target_local_lock = target_local_arc.lock().unwrap();

                if !target_local_lock.is_empty() {
                    target_local_lock
                        .iter_mut()
                        .zip(target_local_owned.iter())
                        .for_each(|(c, m)| *c += *m);
                } else {
                    target_local_lock.extend(target_local_owned);
                }
            })
        }
    }

    fn m2p(&self) {
        if let Some(targets) = self.fmm.tree.get_leaves() {
            targets.par_iter().for_each(move |&target| {
                let fmm_arc = Arc::clone(&self.fmm);
                let target_potential_arc = Arc::clone(self.potentials.get(&target).unwrap());

                if let Some(points) = fmm_arc.tree().get_points(&target) {
                    if let Some(w_list) = fmm_arc.get_w_list(&target) {
                        for source in w_list.iter() {
                            let source_multipole_arc =
                                Arc::clone(self.multipoles.get(source).unwrap());

                            let upward_equivalent_surface = source
                                .compute_surface(
                                    fmm_arc.tree().get_domain(),
                                    fmm_arc.order(),
                                    fmm_arc.alpha_inner,
                                )
                                .into_iter()
                                .flat_map(|[x, y, z]| vec![x, y, z])
                                .collect_vec();

                            let source_multipole_lock = source_multipole_arc.lock().unwrap();
                            let source_multipole_view =
                                ArrayView::from(source_multipole_lock.deref());
                            let source_multipole_slice = source_multipole_view.as_slice().unwrap();

                            let target_coordinates = points
                                .iter()
                                .map(|p| p.coordinate)
                                .flat_map(|[x, y, z]| vec![x, y, z])
                                .collect_vec();

                            let mut target_potential =
                                vec![0f64; target_coordinates.len() / self.fmm.kernel.dim()];

                            fmm_arc.kernel().potential(
                                &upward_equivalent_surface[..],
                                source_multipole_slice,
                                &target_coordinates[..],
                                &mut target_potential,
                            );

                            let mut target_potential_lock = target_potential_arc.lock().unwrap();

                            if !target_potential_lock.is_empty() {
                                target_potential_lock
                                    .iter_mut()
                                    .zip(target_potential.iter())
                                    .for_each(|(p, n)| *p += *n);
                            } else {
                                target_potential_lock.extend(target_potential);
                            }
                        }
                    }
                }
            })
        }
    }

    fn l2p(&self) {
        if let Some(targets) = self.fmm.tree().get_leaves() {
            targets.par_iter().for_each(move |&leaf| {
                let fmm_arc = Arc::clone(&self.fmm);
                let target_potential_arc = Arc::clone(self.potentials.get(&leaf).unwrap());
                let source_local_arc = Arc::clone(self.locals.get(&leaf).unwrap());

                if let Some(target_points) = fmm_arc.tree().get_points(&leaf) {
                    // Lookup data
                    let target_coordinates = target_points
                        .iter()
                        .map(|p| p.coordinate)
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();

                    let downward_equivalent_surface = leaf
                        .compute_surface(&fmm_arc.tree().domain, fmm_arc.order, fmm_arc.alpha_outer)
                        .into_iter()
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();

                    let source_local_lock = source_local_arc.lock().unwrap();
                    let source_local_ref = ArrayView::from(source_local_lock.deref());
                    let source_local_slice = source_local_ref.as_slice().unwrap();

                    let mut target_potential =
                        vec![0f64; target_coordinates.len() / self.fmm.kernel.dim()];

                    fmm_arc.kernel().potential(
                        &downward_equivalent_surface[..],
                        source_local_slice,
                        &target_coordinates[..],
                        &mut target_potential,
                    );

                    let mut out_potential_lock = target_potential_arc.lock().unwrap();

                    if !out_potential_lock.is_empty() {
                        out_potential_lock
                            .iter_mut()
                            .zip(target_potential.iter())
                            .for_each(|(p, n)| *p += *n);
                    } else {
                        out_potential_lock.extend(target_potential);
                    }
                }
            })
        }
    }

    fn p2l(&self) {
        if let Some(targets) = self.fmm.tree().get_leaves() {
            targets.par_iter().for_each(move |&leaf| {
                let fmm_arc = Arc::clone(&self.fmm);
                let target_local_arc = Arc::clone(self.locals.get(&leaf).unwrap());

                if let Some(x_list) = fmm_arc.get_x_list(&leaf) {
                    for source in x_list.iter() {
                        if let Some(source_points) = fmm_arc.tree().get_points(source) {
                            let source_coordinates = source_points
                                .iter()
                                .map(|p| p.coordinate)
                                .flat_map(|[x, y, z]| vec![x, y, z])
                                .collect_vec();

                            let source_charges = self.charges.get(source).unwrap();
                            let source_charges_view = ArrayView::from(source_charges.deref());
                            let source_charges_slice = source_charges_view.as_slice().unwrap();

                            let downward_check_surface = leaf
                                .compute_surface(
                                    &fmm_arc.tree().domain,
                                    fmm_arc.order,
                                    fmm_arc.alpha_inner,
                                )
                                .into_iter()
                                .flat_map(|[x, y, z]| vec![x, y, z])
                                .collect_vec();

                            let mut downward_check_potential =
                                vec![0f64; downward_check_surface.len() / fmm_arc.kernel().dim()];

                            fmm_arc.kernel.potential(
                                &source_coordinates[..],
                                source_charges_slice,
                                &downward_check_surface[..],
                                &mut downward_check_potential[..],
                            );

                            let downward_check_potential =
                                ArrayView::from(&downward_check_potential);

                            let mut target_local_lock = target_local_arc.lock().unwrap();

                            let target_local_owned = fmm_arc.kernel().scale(leaf.level())
                                * &fmm_arc
                                    .dc2e_inv
                                    .0
                                    .dot(&fmm_arc.dc2e_inv.1.dot(&downward_check_potential));

                            if !target_local_lock.is_empty() {
                                target_local_lock
                                    .iter_mut()
                                    .zip(target_local_owned.iter())
                                    .for_each(|(o, l)| *o += *l);
                            } else {
                                target_local_lock.extend(target_local_owned);
                            }
                        }
                    }
                }
            })
        }
    }

    fn p2p(&self) {
        if let Some(targets) = self.fmm.tree.get_leaves() {
            targets.par_iter().for_each(move |&target| {
                let fmm_arc = Arc::clone(&self.fmm);
                let target_potential_arc = Arc::clone(self.potentials.get(&target).unwrap());

                if let Some(target_points) = fmm_arc.tree().get_points(&target) {
                    let target_coordinates = target_points
                        .iter()
                        .map(|p| p.coordinate)
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();

                    if let Some(u_list) = fmm_arc.get_u_list(&target) {
                        for source in u_list.iter() {
                            if let Some(source_points) = fmm_arc.tree().get_points(source) {
                                let source_coordinates = source_points
                                    .iter()
                                    .map(|p| p.coordinate)
                                    .flat_map(|[x, y, z]| vec![x, y, z])
                                    .collect_vec();

                                let source_charges_arc =
                                    Arc::clone(self.charges.get(source).unwrap());
                                let source_charges_view =
                                    ArrayView::from(source_charges_arc.deref());
                                let source_charges_slice = source_charges_view.as_slice().unwrap();

                                let mut target_potential =
                                    vec![0f64; target_coordinates.len() / self.fmm.kernel.dim()];

                                fmm_arc.kernel.potential(
                                    &source_coordinates[..],
                                    source_charges_slice,
                                    &target_coordinates[..],
                                    &mut target_potential,
                                );

                                let mut target_potential_lock =
                                    target_potential_arc.lock().unwrap();

                                if !target_potential_lock.is_empty() {
                                    target_potential_lock
                                        .iter_mut()
                                        .zip(target_potential.iter())
                                        .for_each(|(c, p)| *c += *p);
                                } else {
                                    target_potential_lock.extend(target_potential)
                                }
                            }
                        }
                    }
                }
            })
        }
    }
}

impl<T, U> InteractionLists for KiFmm<T, U>
where
    T: Tree<NodeIndex = MortonKey, NodeIndices = MortonKeys>,
    U: Kernel,
{
    type Tree = T;

    fn get_u_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices> {
        let mut u_list = Vec::<MortonKey>::new();
        let neighbours = key.neighbors();

        // Child level
        let mut neighbors_children_adj: Vec<MortonKey> = neighbours
            .iter()
            .flat_map(|n| n.children())
            .filter(|nc| self.tree().get_all_keys_set().contains(nc) && key.is_adjacent(nc))
            .collect();

        // Key level
        let mut neighbors_adj: Vec<MortonKey> = neighbours
            .iter()
            .filter(|n| self.tree().get_all_keys_set().contains(n) && key.is_adjacent(n))
            .cloned()
            .collect();

        // Parent level
        let mut parent_neighbours_adj: Vec<MortonKey> = key
            .parent()
            .neighbors()
            .into_iter()
            .filter(|pn| self.tree().get_all_keys_set().contains(pn) && key.is_adjacent(pn))
            .collect();

        u_list.append(&mut neighbors_children_adj);
        u_list.append(&mut neighbors_adj);
        u_list.append(&mut parent_neighbours_adj);
        u_list.push(*key);

        if !u_list.is_empty() {
            Some(MortonKeys {
                keys: u_list,
                index: 0,
            })
        } else {
            None
        }
    }

    fn get_v_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices> {
        if key.level() >= 2 {
            let v_list = key
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| self.tree().get_all_keys_set().contains(pnc) && !key.is_adjacent(pnc))
                .collect_vec();

            if !v_list.is_empty() {
                return Some(MortonKeys {
                    keys: v_list,
                    index: 0,
                });
            } else {
                return None;
            }
        }
        None
    }

    fn get_w_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices> {
        // Child level
        let w_list = key
            .neighbors()
            .iter()
            .flat_map(|n| n.children())
            .filter(|nc| self.tree().get_all_keys_set().contains(nc) && !key.is_adjacent(nc))
            .collect_vec();

        if !w_list.is_empty() {
            Some(MortonKeys {
                keys: w_list,
                index: 0,
            })
        } else {
            None
        }
    }

    fn get_x_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices> {
        let x_list = key
            .parent()
            .neighbors()
            .into_iter()
            .filter(|pn| self.tree.get_all_keys_set().contains(pn) && !key.is_adjacent(pn))
            .collect_vec();

        if !x_list.is_empty() {
            Some(MortonKeys {
                keys: x_list,
                index: 0,
            })
        } else {
            None
        }
    }
}

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

impl<T> FmmLoop for FmmData<T>
where
    T: Fmm,
    FmmData<T>: SourceTranslation + TargetTranslation,
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
        let start = Instant::now();
        for level in 2..=depth {
            if level > 2 {
                self.l2l(level)
            }
            self.m2l(level);
        }
        println!("M2L/L2L = {:?}ms", start.elapsed().as_millis());

        let start = Instant::now();
        // Leaf level computations
        self.p2l();
        println!("P2L = {:?}ms", start.elapsed().as_millis());

        // Sum all potential contributions
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
mod test {
    use approx::{assert_relative_eq, RelativeEq};
    use rand::prelude::*;
    use rand::SeedableRng;

    use bempp_tree::types::point::PointType;
    use rayon::ThreadPool;

    use crate::laplace::LaplaceKernel;

    use super::*;

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

    // #[test]
    // fn test_m2l() {
    //     let kernel = LaplaceKernel {
    //         dim: 3,
    //         is_singular: true,
    //         value_dimension: 3,
    //     };

    //     // Create FmmTree
    //     let npoints: usize = 10000;
    //     let points = points_fixture(npoints);
    //     let depth = 2;
    //     let n_crit = 150;

    //     let tree = SingleNodeTree::new(&points, false, Some(n_crit), Some(depth));
    //     let order = 2;
    //     let alpha_inner = 1.05;
    //     let alpha_outer = 1.95;

    //     // New FMM
    //     let fmm = KiFmm::new(order, alpha_inner, alpha_outer, kernel, tree);

    //     let charges = Charges::new();

    //     // Attach to a data tree
    //     let datatree = FmmData::new(fmm, charges);

    //     // Run algorithm
    //     datatree.run();

    //     for target in datatree.fmm.tree().get_keys(2).unwrap().iter() {
    //         if let Some(v_list) = datatree.fmm.get_v_list(&target) {
    //             let downward_equivalent_surface = target
    //                 .compute_surface(
    //                     datatree.fmm.tree().get_domain(),
    //                     datatree.fmm.order,
    //                     datatree.fmm.alpha_outer,
    //                 )
    //                 .into_iter()
    //                 .flat_map(|[x, y, z]| vec![x, y, z])
    //                 .collect_vec();

    //             let downward_check_surface = target
    //                 .compute_surface(
    //                     datatree.fmm.tree().get_domain(),
    //                     datatree.fmm.order,
    //                     datatree.fmm.alpha_inner,
    //                 )
    //                 .into_iter()
    //                 .flat_map(|[x, y, z]| vec![x, y, z])
    //                 .collect_vec();

    //             let local_expansion_arc = Arc::clone(datatree.locals.get(&target).unwrap());
    //             let local_expansion_lock = local_expansion_arc.lock().unwrap();
    //             let local_expansion_view = ArrayView::from(local_expansion_lock.deref());
    //             let local_expansion_slice = local_expansion_view.as_slice().unwrap();

    //             let mut equivalent = vec![0f64; local_expansion_view.len()];

    //             datatree.fmm.kernel().potential(
    //                 &downward_equivalent_surface[..],
    //                 &local_expansion_slice,
    //                 &downward_check_surface[..],
    //                 &mut equivalent,
    //             );

    //             let mut direct = vec![0f64; local_expansion_view.len()];

    //             for source in v_list.iter() {
    //                 let upward_equivalent_surface = source
    //                     .compute_surface(
    //                         datatree.fmm.tree().get_domain(),
    //                         datatree.fmm.order,
    //                         datatree.fmm.alpha_inner,
    //                     )
    //                     .into_iter()
    //                     .flat_map(|[x, y, z]| vec![x, y, z])
    //                     .collect_vec();

    //                 let multipole_expansion_arc =
    //                     Arc::clone(datatree.multipoles.get(&source).unwrap());
    //                 let multipole_expansion_lock = multipole_expansion_arc.lock().unwrap();
    //                 let multipole_expansion_view =
    //                     ArrayView::from(multipole_expansion_lock.deref());
    //                 let multipole_expansion_slice = multipole_expansion_view.as_slice().unwrap();

    //                 let mut tmp: Vec<f64> = vec![0f64; local_expansion_view.len()];

    //                 datatree.fmm.kernel().potential(
    //                     &upward_equivalent_surface[..],
    //                     &multipole_expansion_slice,
    //                     &downward_check_surface[..],
    //                     &mut tmp,
    //                 );

    //                 direct
    //                     .iter_mut()
    //                     .zip(tmp.iter())
    //                     .for_each(|(d, t)| *d += *t);
    //             }

    //             for (a, b) in equivalent.iter().zip(direct.iter()) {
    //                 let are_equal = a.relative_eq(&b, 1e-5, 1e-5);
    //                 assert!(are_equal);
    //             }
    //         }
    //     }
    // }
    use rayon::ThreadPoolBuilder;

        #[test]
        fn test_fmm() {
            // let _ = rayon::ThreadPoolBuilder::new().num_threads(2).build_global().unwrap();
            let npoints = 1000000;
            let points = points_fixture(npoints);
            let points_clone = points.clone();
            let depth = 5;
            let n_crit = 150;

            let order = 9;
            let alpha_inner = 1.05;
            let alpha_outer = 2.9;
            let adaptive = false;

            let kernel = LaplaceKernel::new(3, false, 3);

            let start = Instant::now();
            let tree = SingleNodeTree::new(&points, adaptive, Some(n_crit), Some(depth));
            println!("Tree = {:?}ms", start.elapsed().as_millis());

            let start = Instant::now();
            let fmm = KiFmm::new(order, alpha_inner, alpha_outer, kernel, tree);
            println!("FMM operators = {:?}ms", start.elapsed().as_millis());

            let charges = Charges::new();

            let datatree = FmmData::new(fmm, charges);
            
            datatree.run();
            
            let leaf = &datatree.fmm.tree.get_leaves().unwrap()[0];

            let potentials = datatree.potentials.get(&leaf).unwrap().lock().unwrap();
            let pts = datatree.fmm.tree().get_points(&leaf).unwrap();

            let mut direct = vec![0f64; pts.len()];
            let all_point_coordinates = points_clone
                .iter()
                .map(|p| p.coordinate)
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();

            let leaf_coordinates = pts
                .iter()
                .map(|p| p.coordinate)
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();
            let all_charges = vec![1f64; points_clone.len()];

            let kernel = LaplaceKernel {
                dim: 3,
                is_singular: false,
                value_dimension: 3,
            };
            kernel.potential(
                &all_point_coordinates[..],
                &all_charges[..],
                &leaf_coordinates[..],
                &mut direct[..],
            );

            // for (a, b) in potentials.iter().zip(direct.iter()) {
            //     let are_equal = a.relative_eq(&b, 1e-4, 1e-4);
            //     assert!(are_equal);
            // }

            let abs_error: f64 = potentials
                .iter()
                .zip(direct.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

            println!("p={:?} rel_error={:?}\n", order, rel_error);
            assert!(false)
    }
}
