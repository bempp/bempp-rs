use bempp_traits::fmm::FmmLoop;
use bempp_traits::fmm::{Fmm, SourceTranslation, TargetTranslation};
use bempp_traits::kernel::Kernel;
use bempp_traits::tree::{FmmInteractionLists, MortonKeyInterface, Tree};
use bempp_tree::types::domain::Domain;
use bempp_tree::types::morton::MortonKey;
use bempp_tree::types::morton::MortonKeys;
use bempp_tree::types::point::Point;
use bempp_tree::types::single_node::SingleNodeTree;
use itertools::Itertools;
use ndarray_linalg::SVDDC;
use std::collections::HashSet;
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
    locals: HashMap<MortonKey, Arc<Mutex<Vec<f64>>>>,
    potentials: HashMap<MortonKey, Arc<Mutex<Vec<f64>>>>,
    points: HashMap<MortonKey, Vec<Point>>,
    charges: HashMap<MortonKey, Arc<Mutex<Vec<f64>>>>,
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

impl KiFmm<SingleNodeTree, LaplaceKernel> {
    /// Scaling function for the M2L operator at a given level.
    fn m2l_scale(level: u64) -> f64 {
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
    fn find_unique_v_list_interactions() -> (Vec<MortonKey>, Vec<MortonKey>, Vec<usize>) {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        let level = 3;

        // Encode point in centre of domain
        let key = MortonKey::from_point(&point, &domain, level);

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

        let (u, s, vt) = se2tc.svddc(ndarray_linalg::JobSvd::Some).unwrap();
        let u = u.unwrap();
        let s = Array2::from_diag(&s);
        let vt = vt.unwrap();
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
                locals.insert(key.clone(), Arc::new(Mutex::new(Vec::new())));
                potentials.insert(key.clone(), Arc::new(Mutex::new(Vec::new())));
                if let Some(point_data) = fmm.tree().get_points(key) {
                    points.insert(key.clone(), point_data.iter().cloned().collect_vec());

                    // TODO: Replace with a global index lookup at some point
                    charges.insert(
                        key.clone(),
                        Arc::new(Mutex::new(vec![1.0; point_data.len()])),
                    );
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
            let charges = Arc::clone(&self.charges.get(&leaf).unwrap());
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

                let charges_lock = charges.lock().unwrap();
                let charges_ref = ArrayView::from(charges_lock.deref());

                // Calculate check potential
                let mut check_potential =
                    vec![0.; upward_check_surface.len() / self.fmm.kernel.dim()];

                fmm.kernel.potential(
                    &coordinates[..],
                    &charges_ref.as_slice().unwrap(),
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

                let operator_index = node.siblings().iter().position(|&x| x == node).unwrap();

                let in_multipole_lock = in_node.lock().unwrap();
                let in_multipole_ref = ArrayView::from(in_multipole_lock.deref());

                let out_expansion = fmm.m2m[operator_index].dot(&in_multipole_ref);
                let mut out_multipole = out_node.lock().unwrap();

                if !out_multipole.is_empty() {
                    out_multipole
                        .iter_mut()
                        .zip(out_expansion.iter())
                        .for_each(|(c, m)| *c += *m);
                } else {
                    out_multipole.extend(out_expansion);
                }
            })
        }
    }
}

impl<T, U> FmmInteractionLists for KiFmm<T, U>
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

impl TargetTranslation for FmmDataTree<KiFmm<SingleNodeTree, LaplaceKernel>> {
    fn m2l(&self, level: u64) {
        if let Some(nodes) = self.fmm.tree.get_keys(level) {
            nodes.par_iter().for_each(move |&node| {
                let fmm = Arc::clone(&self.fmm);
                let locals = Arc::clone(&self.locals.get(&node).unwrap());

                // Get interaction list for node
                if let Some(v_list) = fmm.get_v_list(&node) {
                    // Clone references for interior mutability
                    let mut multipoles = vec![];
                    for node in v_list.iter() {
                        multipoles.push(Arc::clone(&self.multipoles.get(&node).unwrap()));
                    }

                    for (i, out_node) in v_list.iter().enumerate() {
                        let ncoeffs = KiFmm::ncoeffs(self.fmm.order);

                        // Locate correct components of compressed M2L matrix.
                        let transfer_vector = out_node.find_transfer_vector(&node);

                        let v_idx = fmm
                            .transfer_vectors
                            .iter()
                            .position(|&x| x == transfer_vector)
                            .unwrap();

                        let v_lidx = v_idx * ncoeffs;
                        let v_ridx = v_lidx + ncoeffs;
                        let vt_sub = self.fmm.m2l.2.slice(s![.., v_lidx..v_ridx]);

                        let in_multipole = &multipoles[i];
                        let in_multipole_lock = in_multipole.lock().unwrap();
                        let in_multipole_ref = ArrayView::from(in_multipole_lock.deref());

                        let out_expansion = KiFmm::m2l_scale(node.level())
                            * self.fmm.kernel.scale(node.level())
                            * self.fmm.dc2e_inv.0.dot(
                                &self.fmm.dc2e_inv.1.dot(
                                    &fmm.m2l
                                        .0
                                        .dot(&fmm.m2l.1.dot(&vt_sub.dot(&in_multipole_ref))),
                                ),
                            );

                        let out_expansion = out_expansion.as_slice().unwrap();

                        let mut out_node = locals.lock().unwrap();

                        if !out_node.is_empty() {
                            out_node
                                .iter_mut()
                                .zip(out_expansion.iter())
                                .for_each(|(c, m)| *c += *m);
                        } else {
                            out_node.extend(out_expansion);
                        }
                    }
                }
            })
        }
    }

    fn l2l(&self, level: u64) {
        if let Some(nodes) = self.fmm.tree.get_keys(level) {
            nodes.par_iter().for_each(move |&node| {
                let in_node = Arc::clone(&self.multipoles.get(&node.parent()).unwrap());
                let out_node = Arc::clone(&self.multipoles.get(&node).unwrap());
                let fmm = Arc::clone(&self.fmm);

                let operator_index = node.siblings().iter().position(|&x| x == node).unwrap();

                let in_local_lock = in_node.lock().unwrap();
                let in_local_ref = ArrayView::from(in_local_lock.deref());

                let out_expansion = fmm.l2l[operator_index].dot(&in_local_ref);
                let mut out_local = out_node.lock().unwrap();

                if !out_local.is_empty() {
                    out_local
                        .iter_mut()
                        .zip(out_expansion.iter())
                        .for_each(|(c, m)| *c += *m);
                } else {
                    out_local.extend(out_expansion);
                }
            })
        }
    }

    fn m2p(&self) {
        let leaves = self.fmm.tree.get_leaves();

        leaves.par_iter().for_each(move |&leaf| {
            let fmm = Arc::clone(&self.fmm);
            let out_node = Arc::clone(&self.potentials.get(&leaf).unwrap());

            if let Some(points) = fmm.tree().get_points(&leaf) {
                if let Some(w_list) = fmm.get_w_list(&leaf) {
                    for source in w_list.iter() {
                        let in_node = Arc::clone(&self.multipoles.get(&source).unwrap());

                        let upward_equivalent_surface = source
                            .compute_surface(fmm.tree().get_domain(), fmm.order(), fmm.alpha_inner)
                            .into_iter()
                            .flat_map(|[x, y, z]| vec![x, y, z])
                            .collect_vec();

                        let multipole_expansion_lock = in_node.lock().unwrap();
                        let multipole_expansion_ref =
                            ArrayView::from(multipole_expansion_lock.deref());

                        let coordinates = points
                            .iter()
                            .map(|p| p.coordinate)
                            .flat_map(|[x, y, z]| vec![x, y, z])
                            .collect_vec();

                        let mut potential = vec![0f64; coordinates.len()];
                        fmm.kernel().potential(
                            &upward_equivalent_surface[..],
                            &multipole_expansion_ref.as_slice().unwrap(),
                            &coordinates[..],
                            &mut potential,
                        );

                        let mut out_potential_lock = out_node.lock().unwrap();

                        if !out_potential_lock.is_empty() {
                            out_potential_lock
                                .iter_mut()
                                .zip(potential.iter())
                                .for_each(|(p, n)| *p += *n);
                        } else {
                            out_potential_lock.extend(potential);
                        }
                    }
                }
            }
        })
    }

    fn l2p(&self) {
        let leaves = self.fmm.tree.get_leaves();

        leaves.par_iter().for_each(move |&leaf| {
            let fmm = Arc::clone(&self.fmm);
            let out_potential = Arc::clone(&self.potentials.get(&leaf).unwrap());
            let local = Arc::clone(&self.locals.get(&leaf).unwrap());

            if let Some(points) = fmm.tree().get_points(&leaf) {
                // Lookup data
                let coordinates = points
                    .iter()
                    .map(|p| p.coordinate)
                    .flat_map(|[x, y, z]| vec![x, y, z])
                    .collect_vec();

                let downward_equivalent_surface = leaf
                    .compute_surface(&fmm.tree.domain, fmm.order, fmm.alpha_outer)
                    .into_iter()
                    .flat_map(|[x, y, z]| vec![x, y, z])
                    .collect_vec();

                let local_expansion_lock = local.lock().unwrap();
                let local_expansion_ref = ArrayView::from(local_expansion_lock.deref());

                let mut potential = vec![0f64; coordinates.len()];

                fmm.kernel().potential(
                    &downward_equivalent_surface[..],
                    &local_expansion_ref.as_slice().unwrap(),
                    &coordinates[..],
                    &mut potential,
                );

                let mut out_potential_lock = out_potential.lock().unwrap();

                if !out_potential_lock.is_empty() {
                    out_potential_lock
                        .iter_mut()
                        .zip(potential.iter())
                        .for_each(|(p, n)| *p += *n);
                } else {
                    out_potential_lock.extend(potential);
                }
            }
        })
    }

    fn p2l(&self) {
        let leaves = self.fmm.tree.get_leaves();

        leaves.par_iter().for_each(move |&leaf| {
            let fmm = Arc::clone(&self.fmm);
            let out_node = Arc::clone(&self.locals.get(&leaf).unwrap());

            if let Some(x_list) = fmm.tree().get_x_list(&leaf) {
                for source in x_list.iter() {
                    if let Some(points) = fmm.tree().get_points(&source) {
                        let coordinates = points
                            .iter()
                            .map(|p| p.coordinate)
                            .flat_map(|[x, y, z]| vec![x, y, z])
                            .collect_vec();
                        // TODO check whether I need to wrap anything in Arcs when using Rayon?
                        let charges = self.charges.get(&source).unwrap();
                        let charges_lock = charges.lock().unwrap();
                        let charges_ref = ArrayView::from(charges_lock.deref());

                        let downward_check_surface = leaf
                            .compute_surface(&fmm.tree.domain, fmm.order, fmm.alpha_inner)
                            .into_iter()
                            .flat_map(|[x, y, z]| vec![x, y, z])
                            .collect_vec();

                        let mut downward_check_potential =
                            vec![0f64; downward_check_surface.len() / fmm.kernel().dim()];
                        fmm.kernel.potential(
                            &coordinates[..],
                            &charges_ref.as_slice().unwrap(),
                            &downward_check_surface[..],
                            &mut downward_check_potential[..],
                        );

                        let downward_check_potential = ArrayView::from(&downward_check_potential);

                        let mut local_lock = out_node.lock().unwrap();

                        let out_local = fmm.kernel().scale(leaf.level())
                            * &fmm
                                .dc2e_inv
                                .0
                                .dot(&fmm.dc2e_inv.1.dot(&downward_check_potential));

                        if !local_lock.is_empty() {
                            local_lock
                                .iter_mut()
                                .zip(out_local.iter())
                                .for_each(|(o, l)| *o += *l);
                        } else {
                            local_lock.extend(out_local);
                        }
                    }
                }
            }
        })
    }

    fn p2p(&self) {
        let leaves = self.fmm.tree.get_leaves();

        leaves.par_iter().for_each(move |&leaf| {
            let fmm = Arc::clone(&self.fmm);
            let out_node = Arc::clone(&self.potentials.get(&leaf).unwrap());

            if let Some(out_points) = fmm.tree().get_points(&leaf) {
                let out_coordinates = out_points
                    .iter()
                    .map(|p| p.coordinate)
                    .flat_map(|[x, y, z]| vec![x, y, z])
                    .collect_vec();

                if let Some(u_list) = fmm.get_u_list(&leaf) {
                    for source in u_list.iter() {
                        if let Some(in_points) = fmm.tree().get_points(&source) {
                            let in_coordinates = out_points
                                .iter()
                                .map(|p| p.coordinate)
                                .flat_map(|[x, y, z]| vec![x, y, z])
                                .collect_vec();

                            let charges = Arc::clone(&self.charges.get(&leaf).unwrap());
                            let charges_lock = charges.lock().unwrap();
                            let charges_ref = ArrayView::from(charges_lock.deref());

                            let mut potential = vec![0f64; out_coordinates.len()];

                            fmm.kernel.potential(
                                &in_coordinates[..],
                                &charges_ref.as_slice().unwrap(),
                                &out_coordinates[..],
                                &mut potential,
                            );

                            let mut out_potential = out_node.lock().unwrap();

                            if !out_potential.is_empty() {
                                out_potential
                                    .iter_mut()
                                    .zip(potential.iter())
                                    .for_each(|(c, p)| *c += *p);
                            } else {
                                out_potential.extend(potential)
                            }
                        }
                    }
                }
            }
        })
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

impl<T> FmmLoop for FmmDataTree<T>
where
    T: Fmm,
    FmmDataTree<T>: SourceTranslation + TargetTranslation,
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

    fn downward_pass(&self) {
        let depth = self.fmm.tree().get_depth();
        for level in 2..=depth {
            self.m2l(level);
            self.l2l(level);
        }

        // Leaf level computations
        self.p2l();
        self.m2p();
        self.p2p();
        self.l2p();
    }

    fn run(&self) {
        self.upward_pass();
        self.downward_pass();
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

    // #[test]
    // fn test_p2m() {
    //     let npoints = 10000;
    //     let points = points_fixture(npoints);
    //     let depth = 3;
    //     let n_crit = 150;

    //     let order = 5;
    //     let alpha_inner = 1.05;
    //     let alpha_outer = 1.95;
    //     let adaptive = false;

    //     let kernel = LaplaceKernel {
    //         dim: 3,
    //         is_singular: false,
    //         value_dimension: 3,
    //     };

    //     let tree = SingleNodeTree::new(
    //         &points,
    //         adaptive,
    //         Some(n_crit),
    //         Some(depth)
    //     );

    //     let fmm = KiFmm::new(
    //         order,
    //         order,
    //         alpha_inner,
    //         alpha_outer,
    //         kernel,
    //         tree
    //     );

    //     let source_datatree = FmmDataTree::new(fmm);

    //     source_datatree.p2m();

    //     let leaf = source_datatree.fmm.tree.get_leaves()[0];

    //     let leaf_expansion = source_datatree.multipoles.get(&leaf).unwrap().lock().unwrap().deref().clone();
    //     let points = source_datatree.points.get(&leaf).unwrap();

    //     let distant_point = vec![1000., 0., 0.];
    //     let mut direct = vec![0.];
    //     let coordinates = points
    //         .iter()
    //         .map(|p| p.coordinate)
    //         .flat_map(|[x, y, z]| vec![x, y, z])
    //         .collect_vec();
    //     let charges = vec![1.; coordinates.len()];
    //     source_datatree.fmm.kernel.potential(&coordinates[..], &charges[..], &distant_point[..], &mut direct);

    //     let expansion = source_datatree.multipoles.get(&leaf).unwrap().lock().unwrap().deref().clone();
    //     let mut estimate = vec![0.];

    //     let tree = SingleNodeTree::new(
    //         &points,
    //         adaptive,
    //         Some(n_crit),
    //         Some(depth)
    //     );

    //     let domain = tree.get_domain();

    //     let equivalent_surface = leaf
    //         .compute_surface(&domain, order, alpha_inner)
    //         .into_iter()
    //         .flat_map(|[x, y, z]| vec![x, y, z])
    //         .collect_vec();

    //     source_datatree.fmm.kernel.potential(&equivalent_surface[..], &expansion[..], &distant_point[..], &mut estimate[..]);
    //     println!("key {:?} direct {:?} estimate {:?}", leaf, direct, estimate);
    //     assert!(false)
    // }

    #[test]
    fn test_downward_pass() {
        let npoints = 10000;
        let points = points_fixture(npoints);
        let depth = 3;
        let n_crit = 150;

        let order = 2;
        let alpha_inner = 1.05;
        let alpha_outer = 1.95;
        let adaptive = false;

        let kernel = LaplaceKernel {
            dim: 3,
            is_singular: false,
            value_dimension: 3,
        };

        let tree = SingleNodeTree::new(&points, adaptive, Some(n_crit), Some(depth));

        let fmm = KiFmm::new(order, alpha_inner, alpha_outer, kernel, tree);

        let datatree = FmmDataTree::new(fmm);

        datatree.run();

        let leaf = &datatree.fmm.tree.get_leaves()[0];

        println!("{:?}", datatree.locals.get(leaf));

        assert!(false)
    }

    // #[test]
    // fn test_upward_pass() {
    //     let npoints = 10000;
    //     let points = points_fixture(npoints);
    //     let depth = 3;
    //     let n_crit = 150;

    //     let order = 10;
    //     let alpha_inner = 1.05;
    //     let alpha_outer = 1.95;
    //     let adaptive = false;

    //     let kernel = LaplaceKernel {
    //         dim: 3,
    //         is_singular: false,
    //         value_dimension: 3,
    //     };

    //     let tree = SingleNodeTree::new(&points, adaptive, Some(n_crit), Some(depth));

    //     let fmm = KiFmm::new(order, alpha_inner, alpha_outer, kernel, tree);

    //     let source_datatree = FmmDataTree::new(fmm);

    //     source_datatree.upward_pass();

    //     let distant_point = vec![1000., 0., 0.];
    //     let mut direct = vec![0.];
    //     let coordinates = points
    //         .iter()
    //         .map(|p| p.coordinate)
    //         .flat_map(|[x, y, z]| vec![x, y, z])
    //         .collect_vec();
    //     let charges = vec![1.; coordinates.len()];
    //     source_datatree.fmm.kernel.potential(
    //         &coordinates[..],
    //         &charges[..],
    //         &distant_point[..],
    //         &mut direct,
    //     );
    //     println!("Direct {:?}", direct);

    //     let mut estimate = vec![0.];

    //     let tree = SingleNodeTree::new(&points, adaptive, Some(n_crit), Some(depth));

    //     let domain = tree.get_domain();

    //     let equivalent_surface = ROOT
    //         .compute_surface(&domain, order, alpha_inner)
    //         .into_iter()
    //         .flat_map(|[x, y, z]| vec![x, y, z])
    //         .collect_vec();
    //     let expansion = source_datatree
    //         .multipoles
    //         .get(&ROOT)
    //         .unwrap()
    //         .lock()
    //         .unwrap()
    //         .deref()
    //         .clone();

    //     source_datatree.fmm.kernel.potential(
    //         &equivalent_surface[..],
    //         &expansion[..],
    //         &distant_point[..],
    //         &mut estimate[..],
    //     );

    //     println!("FMM {:?}", estimate);
    //     assert!(false)
    // }
}
