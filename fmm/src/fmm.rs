extern crate blas_src;
// TODO Should be generic over kernel float type parmeter
// TODO SHould be generic over kernel evaluation type
// TODO should check what happens with rectangular distributions of points!

use cauchy::Scalar;
use itertools::Itertools;
// use ndarray::AssignElem;
// use ndarray::*;
// use ndarray_ndimage::{pad, PadMode};
// use ndrustfft::{ndfft, ndfft_r2c, ndifft, ndifft_r2c, Complex, FftHandler, R2cFftHandler};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, RwLock},
    time::Instant,
};

use rlst::algorithms::linalg::LinAlg;
use rlst::algorithms::traits::pseudo_inverse::Pinv;
use rlst::algorithms::traits::svd::{Mode, Svd};
use rlst::common::traits::{NewLikeSelf, NewLikeTranspose, Transpose};
use rlst::common::{
    tools::PrettyPrint,
    traits::{Copy, Eval},
};
use rlst::dense::{base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix};
use rlst::dense::{rlst_fixed_mat, rlst_mat, rlst_pointer_mat, traits::*, Dot, Shape};
use rlst::{
    self,
    common::traits::ColumnMajorIterator,
    dense::{rlst_col_vec, rlst_mut_pointer_mat},
};

use bempp_field::{
    FftFieldTranslationNaiveKiFmm, SvdFieldTranslationKiFmm, SvdFieldTranslationNaiveKiFmm,
};
use bempp_traits::{
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, FmmLoop, InteractionLists, SourceTranslation, TargetTranslation},
    kernel::{EvalType, Kernel},
    tree::Tree,
};
use bempp_tree::{
    constants::ROOT,
    types::{
        morton::{MortonKey, MortonKeys},
        point::Point,
        single_node::SingleNodeTree,
    },
};

use crate::charge::Charges;

type Expansions =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;
type Potentials =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

pub struct FmmData<T: Fmm> {
    fmm: Arc<T>,
    multipoles: HashMap<MortonKey, Arc<Mutex<Expansions>>>,
    locals: HashMap<MortonKey, Arc<Mutex<Expansions>>>,
    potentials: HashMap<MortonKey, Arc<Mutex<Potentials>>>,
    points: HashMap<MortonKey, Vec<Point>>,
    charges: HashMap<MortonKey, Arc<Vec<f64>>>,
    // multipoles: HashMap<MortonKey, Arc<Mutex<Vec<f64>>>>,
    // locals: HashMap<MortonKey, Arc<Mutex<Vec<f64>>>>,
    // potentials: HashMap<MortonKey, Arc<Mutex<Vec<f64>>>>,
    // points: HashMap<MortonKey, Vec<Point>>,
    // charges: HashMap<MortonKey, Arc<Vec<f64>>>,
}

type C2EType =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

pub struct KiFmm<T: Tree, U: Kernel, V: FieldTranslationData<U>> {
    order: usize,

    uc2e_inv: C2EType,

    dc2e_inv: C2EType,

    alpha_inner: f64,
    alpha_outer: f64,

    m2m: Vec<C2EType>,
    l2l: Vec<C2EType>,
    tree: T,
    kernel: U,
    m2l: V,
}

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

        let nrows = m2l.ncoeffs(order);
        let ncols = m2l.ncoeffs(order);

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

impl<T, U> SourceTranslation for FmmData<KiFmm<SingleNodeTree, T, U>>
where
    T: Kernel<T = f64> + std::marker::Send + std::marker::Sync,
    U: FieldTranslationData<T> + std::marker::Sync + std::marker::Send,
{
    fn p2m<'a>(&self) {
        if let Some(leaves) = self.fmm.tree.get_leaves() {
            leaves.par_iter().for_each(move |&leaf| {
                let leaf_multipole_arc = Arc::clone(self.multipoles.get(&leaf).unwrap());
                let fmm_arc = Arc::clone(&self.fmm);

                if let Some(leaf_points) = self.points.get(&leaf) {
                    let leaf_charges_arc = Arc::clone(self.charges.get(&leaf).unwrap());

                    // Lookup data
                    let leaf_coordinates = leaf_points
                        .iter()
                        .map(|p| p.coordinate)
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();
                    let nsources = leaf_coordinates.len() / self.fmm.kernel.space_dimension();

                    // Get into row major order
                    let leaf_coordinates = unsafe {
                        rlst_pointer_mat!['a, f64, leaf_coordinates.as_ptr(), (nsources, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                    }.eval();

                    let upward_check_surface = leaf.compute_surface(
                        &fmm_arc.tree().domain,
                        fmm_arc.order,
                        fmm_arc.alpha_outer,
                    );
                    let ntargets = upward_check_surface.len() / fmm_arc.kernel.space_dimension();

                    let leaf_charges = leaf_charges_arc.deref();

                    // Calculate check potential
                    let mut check_potential = rlst_col_vec![f64, ntargets];

                    fmm_arc.kernel.evaluate_st(
                        EvalType::Value,
                        leaf_coordinates.data(),
                        &upward_check_surface[..],
                        &leaf_charges[..],
                        check_potential.data_mut()
                    );

                    let leaf_multipole_owned = (
                        fmm_arc.kernel.scale(leaf.level())
                        * fmm_arc.uc2e_inv.dot(&check_potential)
                    ).eval();

                    let mut leaf_multipole_lock = leaf_multipole_arc.lock().unwrap();

                    for i in 0..leaf_multipole_lock.shape().0 {
                        leaf_multipole_lock[[i, 0]] += leaf_multipole_owned[[i, 0]];
                    }
                }
            });
        }
    }

    fn m2m<'a>(&self, level: u64) {
        // Parallelise over nodes at a given level
        if let Some(sources) = self.fmm.tree.get_keys(level) {
            sources.par_iter().for_each(move |&source| {
                let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

                let operator_index = source.siblings().iter().position(|&x| x == source).unwrap();
                let source_multipole_arc = Arc::clone(self.multipoles.get(&source).unwrap());
                let target_multipole_arc =
                    Arc::clone(self.multipoles.get(&source.parent()).unwrap());
                let fmm_arc = Arc::clone(&self.fmm);

                let source_multipole_lock = source_multipole_arc.lock().unwrap();

                let target_multipole_owned =
                    fmm_arc.m2m[operator_index].dot(&source_multipole_lock);

                let mut target_multipole_lock = target_multipole_arc.lock().unwrap();

                for i in 0..ncoeffs {
                    target_multipole_lock[[i, 0]] += target_multipole_owned[[i, 0]];
                }
            })
        }
    }
}

impl<T, U> TargetTranslation for FmmData<KiFmm<SingleNodeTree, T, U>>
where
    T: Kernel<T = f64> + std::marker::Sync + std::marker::Send,
    U: FieldTranslationData<T> + std::marker::Sync + std::marker::Send,
{
    fn l2l(&self, level: u64) {
        if let Some(targets) = self.fmm.tree.get_keys(level) {
            targets.par_iter().for_each(move |&target| {
                let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
                let source_local_arc = Arc::clone(self.locals.get(&target.parent()).unwrap());
                let target_local_arc = Arc::clone(self.locals.get(&target).unwrap());
                let fmm = Arc::clone(&self.fmm);

                let operator_index = target.siblings().iter().position(|&x| x == target).unwrap();

                let source_local_lock = source_local_arc.lock().unwrap();

                let target_local_owned = fmm.l2l[operator_index].dot(&source_local_lock);
                let mut target_local_lock = target_local_arc.lock().unwrap();

                for i in 0..ncoeffs {
                    target_local_lock[[i, 0]] += target_local_owned[[i, 0]];
                }
            })
        }
    }

    fn m2p<'a>(&self) {
        if let Some(targets) = self.fmm.tree.get_leaves() {
            targets.par_iter().for_each(move |&target| {
                let fmm_arc = Arc::clone(&self.fmm);
                let target_potential_arc = Arc::clone(self.potentials.get(&target).unwrap());
                let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

                if let Some(points) = fmm_arc.tree().get_points(&target) {
                    if let Some(w_list) = fmm_arc.get_w_list(&target) {
                        for source in w_list.iter() {
                            let source_multipole_arc =
                                Arc::clone(self.multipoles.get(source).unwrap());

                            let upward_equivalent_surface = source.compute_surface(
                                fmm_arc.tree().get_domain(),
                                fmm_arc.order(),
                                fmm_arc.alpha_inner,
                            );

                            let source_multipole_lock = source_multipole_arc.lock().unwrap();

                            let target_coordinates = points
                                .iter()
                                .map(|p| p.coordinate)
                                .flat_map(|[x, y, z]| vec![x, y, z])
                                .collect_vec();

                            let ntargets = target_coordinates.len() / self.fmm.kernel.space_dimension();

                            // Get into row major order
                            let target_coordinates = unsafe {
                                rlst_pointer_mat!['a, f64, target_coordinates.as_ptr(), (ntargets, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                            }.eval();

                            let mut target_potential = rlst_col_vec![f64, ntargets];

                            fmm_arc.kernel.evaluate_st(
                                EvalType::Value,
                                &upward_equivalent_surface[..],
                                target_coordinates.data(),
                                source_multipole_lock.data(),
                                target_potential.data_mut(),
                            );

                            let mut target_potential_lock = target_potential_arc.lock().unwrap();

                            for i in 0..ntargets {
                                target_potential_lock[[i, 0]] += target_potential[[i, 0]];
                            }
                        }
                    }
                }
            })
        }
    }

    fn l2p<'a>(&self) {
        if let Some(targets) = self.fmm.tree().get_leaves() {
            targets.par_iter().for_each(move |&leaf| {
                let fmm_arc = Arc::clone(&self.fmm);
                let target_potential_arc = Arc::clone(self.potentials.get(&leaf).unwrap());
                let source_local_arc = Arc::clone(self.locals.get(&leaf).unwrap());
                let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

                if let Some(target_points) = fmm_arc.tree().get_points(&leaf) {
                    // Lookup data
                    let target_coordinates = target_points
                        .iter()
                        .map(|p| p.coordinate)
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();
                    let ntargets = target_coordinates.len() / self.fmm.kernel.space_dimension();

                    // Get into row major order
                    let target_coordinates = unsafe {
                        rlst_pointer_mat!['a, f64, target_coordinates.as_ptr(), (ntargets, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                    }.eval();

                    let downward_equivalent_surface = leaf.compute_surface(
                        &fmm_arc.tree().domain,
                        fmm_arc.order,
                        fmm_arc.alpha_outer,
                    );

                    let source_local_lock = source_local_arc.lock().unwrap();

                    let mut target_potential = rlst_col_vec![f64, ntargets];

                    fmm_arc.kernel.evaluate_st(
                        EvalType::Value,
                        &downward_equivalent_surface[..],
                        target_coordinates.data(),
                        source_local_lock.data(),
                        target_potential.data_mut(),
                    );

                    let mut target_potential_lock = target_potential_arc.lock().unwrap();

                    for i in 0..ntargets {
                        target_potential_lock[[i, 0]] += target_potential[[i, 0]];
                    }
                }
            })
        }
    }

    fn p2l<'a>(&self) {
        if let Some(targets) = self.fmm.tree().get_leaves() {
            targets.par_iter().for_each(move |&leaf| {
                let fmm_arc = Arc::clone(&self.fmm);
                let target_local_arc = Arc::clone(self.locals.get(&leaf).unwrap());
                let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

                if let Some(x_list) = fmm_arc.get_x_list(&leaf) {
                    for source in x_list.iter() {
                        if let Some(source_points) = fmm_arc.tree().get_points(source) {
                            let source_coordinates = source_points
                                .iter()
                                .map(|p| p.coordinate)
                                .flat_map(|[x, y, z]| vec![x, y, z])
                                .collect_vec();

                            let nsources = source_coordinates.len() / self.fmm.kernel.space_dimension();

                            // Get into row major order
                            let source_coordinates = unsafe {
                                rlst_pointer_mat!['a, f64, source_coordinates.as_ptr(), (nsources, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                            }.eval();

                            let source_charges = self.charges.get(source).unwrap();

                            let downward_check_surface = leaf.compute_surface(
                                &fmm_arc.tree().domain,
                                fmm_arc.order,
                                fmm_arc.alpha_inner,
                            );
 
                            let ntargets = downward_check_surface.len() / fmm_arc.kernel.space_dimension();
                            let mut downward_check_potential = rlst_col_vec![f64, ntargets];

                            fmm_arc.kernel.evaluate_st(
                                EvalType::Value,
                                source_coordinates.data(),
                                &downward_check_surface[..],
                                &source_charges[..], 
                                downward_check_potential.data_mut() 
                            );


                            let mut target_local_lock = target_local_arc.lock().unwrap();

                            let target_local_owned = (fmm_arc.kernel.scale(leaf.level()) * fmm_arc.dc2e_inv.dot(&downward_check_potential)).eval();

                            for i in 0..ncoeffs {
                                target_local_lock[[i, 0]] += target_local_owned[[i, 0]];
                            }
                        }
                    }
                }
            })
        }
    }

    fn p2p<'a>(&self) {
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
                    
                    let ntargets= target_coordinates.len() / self.fmm.kernel.space_dimension();

                    // Get into row major order
                    let target_coordinates = unsafe {
                        rlst_pointer_mat!['a, f64, target_coordinates.as_ptr(), (ntargets, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                    }.eval();

                    if let Some(u_list) = fmm_arc.get_u_list(&target) {
                        for source in u_list.iter() {
                            if let Some(source_points) = fmm_arc.tree().get_points(source) {
                                let source_coordinates = source_points
                                    .iter()
                                    .map(|p| p.coordinate)
                                    .flat_map(|[x, y, z]| vec![x, y, z])
                                    .collect_vec();

                                let nsources = source_coordinates.len() / self.fmm.kernel.space_dimension();

                                // Get into row major order
                                let source_coordinates = unsafe {
                                    rlst_pointer_mat!['a, f64, source_coordinates.as_ptr(), (nsources, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                                }.eval(); 

                                let source_charges_arc =
                                    Arc::clone(self.charges.get(source).unwrap());

                                // let source_charges_view =
                                //     ArrayView::from(source_charges_arc.deref());
                                // let source_charges_slice = source_charges_view.as_slice().unwrap();

                                let mut target_potential = rlst_col_vec![f64, ntargets];
                                // let mut target_potential =
                                //     vec![0f64; target_coordinates.len() / self.fmm.kernel.dim()];

                                fmm_arc.kernel.evaluate_st(
                                    EvalType::Value,
                                    source_coordinates.data(),
                                    target_coordinates.data(), 
                                    &source_charges_arc[..],
                                    target_potential.data_mut(),
                                );

                                let mut target_potential_lock =
                                    target_potential_arc.lock().unwrap();
                                
                                for i in 0..ntargets {
                                    target_potential_lock[[i, 0]] += target_potential[[i, 0]];
                                }
                            }
                        }
                    }
                }
            })
        }
    }
}

impl<T> FieldTranslation for FmmData<KiFmm<SingleNodeTree, T, SvdFieldTranslationKiFmm<T>>>
where
    T: Kernel<T = f64> + std::marker::Sync + std::marker::Send + Default,
{
    fn m2l<'a>(&self, level: u64) {
        let Some(targets) = self.fmm.tree().get_keys(level) else { return };
        let mut transfer_vector_to_m2l =
            HashMap::<usize, Arc<Mutex<Vec<(MortonKey, MortonKey)>>>>::new();

        for tv in self.fmm.m2l.transfer_vectors.iter() {
            transfer_vector_to_m2l.insert(tv.vector, Arc::new(Mutex::new(Vec::new())));
        }

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
                    .position(|x| x.vector == *transfer_vector)
                    .unwrap();

                let c_lidx = c_idx * self.fmm.m2l.k;
                let c_ridx = c_lidx + self.fmm.m2l.k;
                // let c_sub = self.fmm.m2l.m2l.2.slice(s![.., c_lidx..c_ridx]);

                let (nrows, _) = self.fmm.m2l.m2l.2.shape();
                let top_left = (0, c_lidx);
                let dim = (nrows, self.fmm.m2l.k);

                // println!("{:?} {:?} {:?}", top_left, dim, self.fmm.m2l.m2l.2.shape());
                let c_sub = self.fmm.m2l.m2l.2.block(top_left, dim);

                let m2l_rw = m2l_arc.read().unwrap();
                // let mut multipoles = Array2::zeros((self.fmm.m2l.k, m2l_rw.len()));
                let mut multipoles = rlst_mat![f64, (self.fmm.m2l.k, m2l_rw.len())];

                let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

                for (i, (source, _)) in m2l_rw.iter().enumerate() {
                    let source_multipole_arc = Arc::clone(self.multipoles.get(source).unwrap());
                    let source_multipole_lock = source_multipole_arc.lock().unwrap();

                    // // let source_multipole_view = ArrayView::from(source_multipole_lock.deref());

                    // Compressed multipole
                    let compressed_source_multipole_owned =
                        self.fmm.m2l.m2l.1.dot(&source_multipole_lock).eval();

                    let first = i * self.fmm.m2l.k;
                    let last = first + self.fmm.m2l.k;

                    let multipole_slice = multipoles.get_slice_mut(first, last);
                    multipole_slice.copy_from_slice(compressed_source_multipole_owned.data());
                    // multipoles
                    //     .slice_mut(s![.., i])
                    //     .assign(&compressed_source_multipole_owned);
                }

                // // Compute convolution
                let compressed_check_potential_owned = c_sub.dot(&multipoles);

                // Post process to find check potential
                let check_potential_owned = self
                    .fmm
                    .m2l
                    .m2l
                    .0
                    .dot(&compressed_check_potential_owned)
                    .eval();

                // Compute local
                // // let locals_owned = self.m2l_scale(level)
                // //     * self.fmm.kernel.scale(level)
                // //     * self
                // //         .fmm
                // //         .dc2e_inv
                // //         .0
                // //         .dot(&self.fmm.dc2e_inv.1.dot(&check_potential_owned));
                let locals_owned = (self.fmm.dc2e_inv.dot(&check_potential_owned)
                    * self.fmm.kernel.scale(level)
                    * self.m2l_scale(level))
                .eval();

                // Assign locals
                for (i, (_, target)) in m2l_rw.iter().enumerate() {
                    let target_local_arc = Arc::clone(self.locals.get(target).unwrap());
                    let mut target_local_lock = target_local_arc.lock().unwrap();

                    let first = i * self.fmm.m2l.k;
                    let last = first + self.fmm.m2l.k;

                    let top_left = (0, i);
                    let dim = (self.fmm.m2l.k, 1);
                    let target_local_owned = locals_owned.block(top_left, dim);

                    // let target_local_owned = locals_owned.slice(s![.., i]);

                    // println!("target lock {:?}", target_local_lock.shape());
                    for i in 0..target_local_lock.shape().0 {
                        target_local_lock[[i, 0]] += target_local_owned[[i, 0]];
                    }
                }
            });
    }

    fn m2l_scale(&self, level: u64) -> f64 {
        if level < 2 {
            panic!("M2L only performed on level 2 and below")
        }

        if level == 2 {
            1. / 2.
        } else {
            2_f64.powf((level - 3) as f64)
        }
    }
}

// impl<T> FieldTranslation for FmmData<KiFmm<SingleNodeTree, T, SvdFieldTranslationNaiveKiFmm<T>>>
// where
//     T: Kernel + std::marker::Sync + std::marker::Send + Default,
// {
//     fn m2l(&self, level: u64) {
//         if let Some(targets) = self.fmm.tree().get_keys(level) {
//             // Find transfer vectors
//             targets.par_iter().for_each(move |&target| {
//                 let fmm_arc: Arc<KiFmm<SingleNodeTree, T, SvdFieldTranslationNaiveKiFmm<T>>> =
//                     Arc::clone(&self.fmm);
//                 let target_local_arc = Arc::clone(self.locals.get(&target).unwrap());

//                 if let Some(v_list) = fmm_arc.get_v_list(&target) {
//                     for (_i, source) in v_list.iter().enumerate() {
//                         // Locate correct components of compressed M2L matrix.
//                         let transfer_vector = target.find_transfer_vector(source);

//                         let c_idx = fmm_arc
//                             .m2l
//                             .transfer_vectors
//                             .iter()
//                             .position(|x| x.vector == transfer_vector)
//                             .unwrap();
//                         let c_lidx = c_idx * fmm_arc.m2l.k;
//                         let c_ridx = c_lidx + fmm_arc.m2l.k;
//                         let c_sub = fmm_arc.m2l.m2l.2.slice(s![.., c_lidx..c_ridx]);

//                         let source_multipole_arc = Arc::clone(self.multipoles.get(source).unwrap());
//                         let source_multipole_lock = source_multipole_arc.lock().unwrap();
//                         let source_multipole_view = ArrayView::from(source_multipole_lock.deref());

//                         // Compressed multipole
//                         let compressed_source_multipole_owned =
//                             fmm_arc.m2l.m2l.1.dot(&source_multipole_view);

//                         // Convolution to find compressed check potential
//                         let compressed_check_potential_owned =
//                             c_sub.dot(&compressed_source_multipole_owned);

//                         // Post process to find check potential
//                         let check_potential_owned =
//                             fmm_arc.m2l.m2l.0.dot(&compressed_check_potential_owned);

//                         // Compute local
//                         let target_local_owned = self.m2l_scale(target.level())
//                             * fmm_arc.kernel.scale(target.level())
//                             * fmm_arc
//                                 .dc2e_inv
//                                 .0
//                                 .dot(&self.fmm.dc2e_inv.1.dot(&check_potential_owned));

//                         // Store computation
//                         let mut target_local_lock = target_local_arc.lock().unwrap();

//                         if !target_local_lock.is_empty() {
//                             target_local_lock
//                                 .iter_mut()
//                                 .zip(target_local_owned.iter())
//                                 .for_each(|(c, m)| *c += *m);
//                         } else {
//                             target_local_lock.extend(target_local_owned);
//                         }
//                     }
//                 }
//             })
//         }
//     }

//     fn m2l_scale(&self, level: u64) -> f64 {
//         if level < 2 {
//             panic!("M2L only performed on level 2 and below")
//         }

//         if level == 2 {
//             1. / 2.
//         } else {
//             2_f64.powf((level - 3) as f64)
//         }
//     }
// }

// impl<T> FieldTranslation for FmmData<KiFmm<SingleNodeTree, T, FftFieldTranslationNaiveKiFmm<T>>>
// where
//     T: Kernel + std::marker::Sync + std::marker::Send + Default,
// {
//     fn m2l(&self, level: u64) {
//         if let Some(targets) = self.fmm.tree().get_keys(level) {
//             targets.par_iter().for_each(move |&target| {
//                 let fmm_arc = Arc::clone(&self.fmm);
//                 let target_local_arc = Arc::clone(self.locals.get(&target).unwrap());

//                 if let Some(v_list) = fmm_arc.get_v_list(&target) {
//                     for (_, source) in v_list.iter().enumerate() {
//                         let transfer_vector = target.find_transfer_vector(source);

//                         // Locate correct precomputed FFT of kernel interactions
//                         let k_idx = fmm_arc
//                             .m2l
//                             .transfer_vectors
//                             .iter()
//                             .position(|x| x.vector == transfer_vector)
//                             .unwrap();

//                         // Compute FFT of signal
//                         let source_multipole_arc = Arc::clone(self.multipoles.get(source).unwrap());
//                         let source_multipole_lock = source_multipole_arc.lock().unwrap();

//                         let signal = fmm_arc
//                             .m2l
//                             .compute_signal(fmm_arc.order, source_multipole_lock.deref());

//                         // 1. Pad the signal
//                         let m = signal.len();
//                         let n = signal[0].len();
//                         let k = signal[0][0].len();

//                         let p = 2 * m;
//                         let q = 2 * n;
//                         let r = 2 * k;

//                         let signal = Array3::from_shape_vec(
//                             (m, n, k),
//                             signal.into_iter().flatten().flatten().collect(),
//                         )
//                         .unwrap();

//                         let padding = [[p - m, 0], [q - n, 0], [r - k, 0]];
//                         let padded_signal = pad(&signal, &padding, PadMode::Constant(0.));

//                         // 2. FFT of the padded signal
//                         // 2.1 Init the handlers for FFTs along each axis
//                         let mut handler_ax0 = FftHandler::<f64>::new(p);
//                         let mut handler_ax1 = FftHandler::<f64>::new(q);
//                         let mut handler_ax2 = R2cFftHandler::<f64>::new(r);

//                         // 2.2 Compute the transform along each axis
//                         let mut padded_signal_hat: Array3<Complex<f64>> =
//                             Array3::zeros((p, q, r / 2 + 1));
//                         let mut tmp1: Array3<Complex<f64>> = Array3::zeros((p, q, r / 2 + 1));
//                         ndfft_r2c(&padded_signal, &mut tmp1, &mut handler_ax2, 2);
//                         let mut tmp2: Array3<Complex<f64>> = Array3::zeros((p, q, r / 2 + 1));
//                         ndfft(&tmp1, &mut tmp2, &mut handler_ax1, 1);
//                         ndfft(&tmp2, &mut padded_signal_hat, &mut handler_ax0, 0);

//                         // 3.Compute convolution to find check potential
//                         let padded_kernel_hat = &fmm_arc.m2l.m2l[k_idx];

//                         // Hadamard product
//                         let check_potential_hat = padded_kernel_hat * padded_signal_hat;

//                         // 3.1 Compute iFFT to find check potentials
//                         let mut check_potential: Array3<f64> = Array3::zeros((p, q, r));
//                         let mut tmp1: Array3<Complex<f64>> = Array3::zeros((p, q, r / 2 + 1));
//                         ndifft(&check_potential_hat, &mut tmp1, &mut handler_ax0, 0);
//                         let mut tmp2: Array3<Complex<f64>> = Array3::zeros((p, q, r / 2 + 1));
//                         ndifft(&tmp1, &mut tmp2, &mut handler_ax1, 1);
//                         ndifft_r2c(&tmp2, &mut check_potential, &mut handler_ax2, 2);

//                         // Filter check potentials
//                         let check_potential =
//                             check_potential.slice(s![p - m - 1..p, q - n - 1..q, r - k - 1..r]);

//                         let (_, target_surface_idxs) = target.surface_grid(fmm_arc.order);

//                         let mut tmp = Vec::new();
//                         for index in target_surface_idxs.chunks_exact(fmm_arc.kernel.dim()) {
//                             let element = check_potential[[index[0], index[1], index[2]]];
//                             tmp.push(element);
//                         }

//                         // Compute local coefficients from check potentials
//                         let check_potential = Array::from_shape_vec(
//                             target_surface_idxs.len() / fmm_arc.kernel.dim(),
//                             tmp,
//                         )
//                         .unwrap();

//                         // Compute local
//                         let target_local_owned = self.m2l_scale(target.level())
//                             * fmm_arc.kernel.scale(target.level())
//                             * fmm_arc
//                                 .dc2e_inv
//                                 .0
//                                 .dot(&self.fmm.dc2e_inv.1.dot(&check_potential));

//                         // Store computation
//                         let mut target_local_lock = target_local_arc.lock().unwrap();

//                         if !target_local_lock.is_empty() {
//                             target_local_lock
//                                 .iter_mut()
//                                 .zip(target_local_owned.iter())
//                                 .for_each(|(c, m)| *c += *m);
//                         } else {
//                             target_local_lock.extend(target_local_owned);
//                         }
//                     }
//                 }
//             })
//         }
//     }

//     fn m2l_scale(&self, level: u64) -> f64 {
//         if level < 2 {
//             panic!("M2L only performed on level 2 and below")
//         }

//         if level == 2 {
//             1. / 2.
//         } else {
//             2_f64.powf((level - 3) as f64)
//         }
//     }
// }

impl<T, U, V> InteractionLists for KiFmm<T, U, V>
where
    T: Tree<NodeIndex = MortonKey, NodeIndices = MortonKeys>,
    U: Kernel,
    V: FieldTranslationData<U>,
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
mod test {
    use bempp_kernel::laplace_3d::evaluate_laplace_one_target;
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
        //     let points = points_fixture(npoints);
        //     let points_clone = points.clone();
        //     let depth = 4;
        //     let n_crit = 150;
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

        //     // let m2l_data_svd_naive = SvdFieldTranslationNaiveKiFmm::new(
        //     //     kernel.clone(),
        //     //     Some(k),
        //     //     order,
        //     //     tree.get_domain().clone(),
        //     //     alpha_inner,
        //     // );

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
            &mut direct[..]
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
