//! Implementation of constructors for FMMs as well as the implementation of FmmData, Fmm traits.
use cauchy::Scalar;
use itertools::Itertools;
use num::{Float, ToPrimitive};
use std::time::Instant;

use rlst::{
    algorithms::{linalg::DenseMatrixLinAlgBuilder, traits::svd::Svd},
    common::traits::{Eval, Transpose},
    dense::{rlst_dynamic_mat, rlst_pointer_mat, traits::*, Dot, MultiplyAdd, VectorContainer},
};

use bempp_traits::{
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, FmmLoop, KiFmm as KiFmmTrait, SourceTranslation, TargetTranslation, TimeDict},
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
    types::EvalType,
};
use bempp_tree::{constants::ROOT, types::single_node::SingleNodeTree};

use crate::types::{ChargeDict, FmmDataLinear, SendPtrMut};
use crate::{
    pinv::{pinv, SvdScalar},
    types::KiFmmLinear,
};

/// Implementation of constructor for single node KiFMM
impl<'a, T, U, V> KiFmmLinear<SingleNodeTree<V>, T, U, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V>,
    U: FieldTranslationData<T>,
    V: Scalar<Real = V> + Default + Float,
    SvdScalar<V>: PartialOrd,
    SvdScalar<V>: Scalar + Float + ToPrimitive,
    DenseMatrixLinAlgBuilder<V>: Svd,
    V: MultiplyAdd<
        V,
        VectorContainer<V>,
        VectorContainer<V>,
        VectorContainer<V>,
        Dynamic,
        Dynamic,
        Dynamic,
    >,
    SvdScalar<V>: MultiplyAdd<
        SvdScalar<V>,
        VectorContainer<SvdScalar<V>>,
        VectorContainer<SvdScalar<V>>,
        VectorContainer<SvdScalar<V>>,
        Dynamic,
        Dynamic,
        Dynamic,
    >,
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
        alpha_inner: V,
        alpha_outer: V,
        kernel: T,
        tree: SingleNodeTree<V>,
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
            rlst_pointer_mat!['a, <V as cauchy::Scalar>::Real, upward_equivalent_surface.as_ptr(), (nequiv_surface, kernel.space_dimension()), (1, nequiv_surface)]
        };
        let upward_check_surface = unsafe {
            rlst_pointer_mat!['a, <V as cauchy::Scalar>::Real, upward_check_surface.as_ptr(), (ncheck_surface, kernel.space_dimension()), (1, ncheck_surface)]
        };
        let downward_equivalent_surface = unsafe {
            rlst_pointer_mat!['a, <V as cauchy::Scalar>::Real, downward_equivalent_surface.as_ptr(), (nequiv_surface, kernel.space_dimension()), (1, nequiv_surface)]
        };
        let downward_check_surface = unsafe {
            rlst_pointer_mat!['a, <V as cauchy::Scalar>::Real, downward_check_surface.as_ptr(), (ncheck_surface, kernel.space_dimension()), (1, ncheck_surface)]
        };

        // Compute upward check to equivalent, and downward check to equivalent Gram matrices
        // as well as their inverses using DGESVD.
        let mut uc2e = rlst_dynamic_mat![V, (ncheck_surface, nequiv_surface)];
        kernel.assemble_st(
            EvalType::Value,
            upward_equivalent_surface.data(),
            upward_check_surface.data(),
            uc2e.data_mut(),
        );

        // Need to tranapose so that rows correspond to targets and columns to sources
        let uc2e = uc2e.transpose().eval();

        let mut dc2e = rlst_dynamic_mat![V, (ncheck_surface, nequiv_surface)];
        kernel.assemble_st(
            EvalType::Value,
            downward_equivalent_surface.data(),
            downward_check_surface.data(),
            dc2e.data_mut(),
        );

        // Need to tranapose so that rows correspond to targets and columns to sources
        let dc2e = dc2e.transpose().eval();

        let (s, ut, v) = pinv::<V>(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_mat![SvdScalar<V>, (s.len(), s.len())];
        for i in 0..s.len() {
            mat_s[[i, i]] = SvdScalar::<V>::from_real(s[i]);
        }
        let uc2e_inv_1 = v.dot(&mat_s);
        let uc2e_inv_2 = ut;

        let uc2e_inv_1_shape = uc2e_inv_1.shape();
        let uc2e_inv_2_shape = uc2e_inv_2.shape();

        let uc2e_inv_1 = uc2e_inv_1
            .data()
            .iter()
            .map(|x| V::from(*x).unwrap())
            .collect_vec();
        let uc2e_inv_1 = unsafe {
            rlst_pointer_mat!['a, V, uc2e_inv_1.as_ptr(), uc2e_inv_1_shape, (1, uc2e_inv_1_shape.0)]
        }
        .eval();
        let uc2e_inv_2 = uc2e_inv_2
            .data()
            .iter()
            .map(|x| V::from(*x).unwrap())
            .collect_vec();
        let uc2e_inv_2 = unsafe {
            rlst_pointer_mat!['a, V, uc2e_inv_2.as_ptr(), uc2e_inv_2_shape, (1, uc2e_inv_2_shape.0)]
        }
        .eval();

        let (s, ut, v) = pinv::<V>(&dc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_mat![SvdScalar<V>, (s.len(), s.len())];
        for i in 0..s.len() {
            mat_s[[i, i]] = SvdScalar::<V>::from_real(s[i]);
        }

        let dc2e_inv_1 = v.dot(&mat_s);
        let dc2e_inv_2 = ut;

        let dc2e_inv_1_shape = dc2e_inv_1.shape();
        let dc2e_inv_2_shape = dc2e_inv_2.shape();

        let dc2e_inv_1 = dc2e_inv_1
            .data()
            .iter()
            .map(|x| V::from(*x).unwrap())
            .collect_vec();
        let dc2e_inv_1 = unsafe {
            rlst_pointer_mat!['a, V, dc2e_inv_1.as_ptr(), dc2e_inv_1_shape, (1, dc2e_inv_1_shape.0)]
        }
        .eval();
        let dc2e_inv_2 = dc2e_inv_2
            .data()
            .iter()
            .map(|x| V::from(*x).unwrap())
            .collect_vec();
        let dc2e_inv_2 = unsafe {
            rlst_pointer_mat!['a, V, dc2e_inv_2.as_ptr(), dc2e_inv_2_shape, (1, dc2e_inv_2_shape.0)]
        }
        .eval();

        // Calculate M2M/L2L matrices
        let children = ROOT.children();
        let mut m2m = rlst_dynamic_mat![V, (nequiv_surface, 8 * nequiv_surface)];
        // let mut l2l = rlst_dynamic_mat![V, (nequiv_surface, 8 * nequiv_surface)];
        let mut l2l = Vec::new();

        for (i, child) in children.iter().enumerate() {
            let child_upward_equivalent_surface =
                child.compute_surface(tree.get_domain(), order, alpha_inner);
            let child_downward_check_surface =
                child.compute_surface(tree.get_domain(), order, alpha_inner);
            let child_upward_equivalent_surface = unsafe {
                rlst_pointer_mat!['a, <V as cauchy::Scalar>::Real, child_upward_equivalent_surface.as_ptr(), (nequiv_surface, kernel.space_dimension()), (1, nequiv_surface)]
            };
            let child_downward_check_surface = unsafe {
                rlst_pointer_mat!['a, <V as cauchy::Scalar>::Real, child_downward_check_surface.as_ptr(), (ncheck_surface, kernel.space_dimension()), (1, ncheck_surface)]
            };

            let mut pc2ce = rlst_dynamic_mat![V, (ncheck_surface, nequiv_surface)];

            kernel.assemble_st(
                EvalType::Value,
                child_upward_equivalent_surface.data(),
                upward_check_surface.data(),
                pc2ce.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let pc2ce = pc2ce.transpose().eval();

            let tmp = uc2e_inv_1.dot(&uc2e_inv_2.dot(&pc2ce)).eval();
            let l = i * nequiv_surface * nequiv_surface;
            let r = l + nequiv_surface * nequiv_surface;

            m2m.data_mut()[l..r].copy_from_slice(tmp.data());

            let mut cc2pe = rlst_dynamic_mat![V, (ncheck_surface, nequiv_surface)];

            kernel.assemble_st(
                EvalType::Value,
                downward_equivalent_surface.data(),
                child_downward_check_surface.data(),
                cc2pe.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let cc2pe = cc2pe.transpose().eval();
            let mut tmp = dc2e_inv_1.dot(&dc2e_inv_2.dot(&cc2pe)).eval();
            tmp.data_mut()
                .iter_mut()
                .for_each(|d| *d *= kernel.scale(child.level()));

            // let l = i * nequiv_surface * nequiv_surface;
            // let r = l + nequiv_surface * nequiv_surface;
            // l2l.data_mut()[l..r].copy_from_slice(tmp.data());
            l2l.push(tmp);
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
impl<T, U, V> FmmDataLinear<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V>,
    U: FieldTranslationData<T>,
    V: Float + Scalar<Real = V> + Default,
{
    /// Constructor fo the KiFMM's associated FmmData on a single node.
    ///
    /// # Arguments
    /// `fmm` - A single node KiFMM object.
    /// `global_charges` - The charge data associated to the point data via unique global indices.
    pub fn new(
        fmm: KiFmmLinear<SingleNodeTree<V>, T, U, V>,
        global_charges: &ChargeDict<V>,
    ) -> Result<Self, String> {
        if let Some(keys) = fmm.tree().get_all_keys() {
            let ncoeffs = fmm.m2l.ncoeffs(fmm.order);
            let nkeys = keys.len();
            let leaves = fmm.tree().get_all_leaves().unwrap();
            let nleaves = leaves.len();
            let npoints = fmm.tree().get_all_points().unwrap().len();

            let multipoles = vec![V::default(); ncoeffs * nkeys];
            let locals = vec![V::default(); ncoeffs * nkeys];

            let mut potentials = vec![V::default(); npoints];
            let mut potentials_send_pointers = vec![SendPtrMut::default(); nleaves];

            let mut charges = vec![V::default(); npoints];
            let global_indices = vec![0usize; npoints];

            // Lookup leaf coordinates, and assign charges from within the data tree.
            for (i, g_idx) in fmm
                .tree()
                .get_all_global_indices()
                .unwrap()
                .iter()
                .enumerate()
            {
                let charge = global_charges.get(g_idx).unwrap();
                charges[i] = *charge;
            }

            let mut level_multipoles = vec![Vec::new(); (fmm.tree().get_depth() + 1) as usize];
            let mut level_locals = vec![Vec::new(); (fmm.tree().get_depth() + 1) as usize];
            for level in 0..=fmm.tree().get_depth() {
                let keys = fmm.tree().get_keys(level).unwrap();

                let mut tmp_multipoles = Vec::new();
                let mut tmp_locals = Vec::new();
                for key in keys.iter() {
                    let idx = fmm.tree().key_to_index.get(key).unwrap();
                    unsafe {
                        let raw = multipoles.as_ptr().add(ncoeffs * idx) as *mut V;
                        tmp_multipoles.push(SendPtrMut { raw });
                        let raw = locals.as_ptr().add(ncoeffs * idx) as *mut V;
                        tmp_locals.push(SendPtrMut { raw })
                    }
                }
                level_multipoles[level as usize] = tmp_multipoles;
                level_locals[level as usize] = tmp_locals;
            }

            let mut leaf_multipoles = Vec::new();
            let mut leaf_locals = Vec::new();
            for (i, key) in fmm.tree().get_all_keys().unwrap().iter().enumerate() {
                if fmm.tree().get_all_leaves_set().contains(key) {
                    unsafe {
                        let raw = multipoles.as_ptr().add(i * ncoeffs) as *mut V;
                        leaf_multipoles.push(SendPtrMut { raw });
                        let raw = locals.as_ptr().add(i * ncoeffs) as *mut V;
                        leaf_locals.push(SendPtrMut { raw });
                    }
                }
            }

            // Create an index pointer for the charge data
            let mut index_pointer = 0;
            let mut charge_index_pointer = vec![(0usize, 0usize); nleaves];

            let mut potential_raw_pointer = potentials.as_mut_ptr();

            let mut scales = vec![V::default(); nleaves * ncoeffs];
            for (i, leaf) in leaves.iter().enumerate() {
                // Assign scales
                let l = i * ncoeffs;
                let r = l + ncoeffs;
                scales[l..r]
                    .copy_from_slice(vec![fmm.kernel.scale(leaf.level()); ncoeffs].as_slice());

                // Assign potential pointers
                let npoints;
                if let Some(points) = fmm.tree().get_points(leaf) {
                    npoints = points.len();
                } else {
                    npoints = 0;
                }

                potentials_send_pointers[i] = SendPtrMut {
                    raw: potential_raw_pointer,
                };

                let bounds = (index_pointer, index_pointer + npoints);
                charge_index_pointer[i] = bounds;
                index_pointer += npoints;
                unsafe { potential_raw_pointer = potential_raw_pointer.add(npoints) };
            }

            let dim = fmm.kernel().space_dimension();
            let mut upward_surfaces = vec![V::default(); ncoeffs * nkeys * dim];
            let mut downward_surfaces = vec![V::default(); ncoeffs * nkeys * dim];

            // For each key form both upward and downward check surfaces
            for (i, key) in keys.iter().enumerate() {
                let upward_surface =
                    key.compute_surface(fmm.tree().get_domain(), fmm.order(), fmm.alpha_outer());

                let downward_surface =
                    key.compute_surface(fmm.tree().get_domain(), fmm.order(), fmm.alpha_inner());

                let l = i * ncoeffs * dim;
                let r = l + ncoeffs * dim;

                upward_surfaces[l..r].copy_from_slice(&upward_surface);
                downward_surfaces[l..r].copy_from_slice(&downward_surface);
            }

            let mut leaf_upward_surfaces = vec![V::default(); ncoeffs * nleaves * dim];
            let mut leaf_downward_surfaces = vec![V::default(); ncoeffs * nleaves * dim];
            for (i, leaf) in leaves.iter().enumerate() {
                let upward_surface =
                    leaf.compute_surface(fmm.tree().get_domain(), fmm.order(), fmm.alpha_outer());

                let downward_surface =
                    leaf.compute_surface(fmm.tree().get_domain(), fmm.order(), fmm.alpha_outer());

                let l = i * ncoeffs * dim;
                let r = l + ncoeffs * dim;
                leaf_upward_surfaces[l..r].copy_from_slice(&upward_surface);
                leaf_downward_surfaces[l..r].copy_from_slice(&downward_surface);
            }

            return Ok(Self {
                fmm,
                multipoles,
                level_multipoles,
                leaf_multipoles,
                locals,
                level_locals,
                leaf_locals,
                upward_surfaces,
                downward_surfaces,
                leaf_upward_surfaces,
                leaf_downward_surfaces,
                potentials,
                potentials_send_pointers,
                charges,
                charge_index_pointer,
                scales,
                global_indices,
            });
        }

        Err("Not a valid tree".to_string())
    }
}

impl<T, U, V, W> KiFmmTrait for KiFmmLinear<T, U, V, W>
where
    T: Tree,
    U: Kernel<T = W>,
    V: FieldTranslationData<U>,
    W: Scalar + Float + Default,
{
    fn alpha_inner(&self) -> <<Self as Fmm>::Kernel as Kernel>::T {
        self.alpha_inner
    }

    fn alpha_outer(&self) -> <<Self as Fmm>::Kernel as Kernel>::T {
        self.alpha_outer
    }
}

impl<T, U, V, W> Fmm for KiFmmLinear<T, U, V, W>
where
    T: Tree,
    U: Kernel<T = W>,
    V: FieldTranslationData<U>,
    W: Scalar + Float + Default,
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

impl<T, U> FmmLoop for FmmDataLinear<T, U>
where
    T: Fmm,
    U: Scalar<Real = U> + Float + Default,
    FmmDataLinear<T, U>: SourceTranslation + FieldTranslation<U> + TargetTranslation,
{
    fn upward_pass(&self, time: bool) -> Option<TimeDict> {
        match time {
            true => {
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
            false => {
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

    fn downward_pass(&self, time: bool) -> Option<TimeDict> {
        let depth = self.fmm.tree().get_depth();

        match time {
            true => {
                let mut times = TimeDict::default();
                let mut l2l_time = 0;
                let mut m2l_time = 0;

                for level in 2..=depth {
                    if level < depth {
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
            false => {
                for level in 2..=depth {
                    if level > 2 {
                        self.l2l(level);
                    }
                    self.m2l(level);
                }
                // Leaf level computations
                // self.p2l();

                // Sum all potential contributions
                // self.m2p();
                self.p2p();
                self.l2p();

                None
            }
        }
    }

    fn run(&self, time: bool) -> Option<TimeDict> {
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

mod test {

    use super::*;

    use std::env;

    use bempp_field::types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm};
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::implementations::helpers::points_fixture;

    use crate::charge::build_charge_dict;

    #[test]
    fn test_fmm_data_linear() {
        let npoints = 10000;
        let points = points_fixture::<f64>(npoints, None, None);
        let global_idxs = (0..npoints).collect_vec();
        let charges = vec![1.0; npoints];

        let order = 8;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let ncrit = 150;
        let depth = 3;

        // Uniform trees
        {
            let adaptive = false;
            let kernel = Laplace3dKernel::default();

            let tree = SingleNodeTree::new(
                points.data(),
                adaptive,
                Some(ncrit),
                Some(depth),
                &global_idxs[..],
            );

            let m2l_data_fft = FftFieldTranslationKiFmm::new(
                kernel.clone(),
                order,
                *tree.get_domain(),
                alpha_inner,
            );

            let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data_fft);
            // Form charge dict, matching charges with their associated global indices
            let charge_dict = build_charge_dict(&global_idxs[..], &charges[..]);

            let datatree = FmmDataLinear::new(fmm, &charge_dict).unwrap();

            let ncoeffs = datatree.fmm.m2l.ncoeffs(order);
            let nleaves = datatree.fmm.tree().get_all_leaves_set().len();
            let nkeys = datatree.fmm.tree().get_all_keys_set().len();

            // Test that the number of of coefficients is being correctly assigned
            assert_eq!(datatree.multipoles.len(), ncoeffs * nkeys);
            assert_eq!(datatree.leaf_multipoles.len(), nleaves);

            // Test that leaf indices are being mapped correctly to leaf multipoles
            let idx = 0;
            let leaf_key = &datatree.fmm.tree().get_all_leaves().unwrap()[idx];
            let &leaf_idx = datatree.fmm.tree().get_index(leaf_key).unwrap();
            unsafe {
                let result = datatree.multipoles.as_ptr().add(leaf_idx * ncoeffs);
                let expected =
                    datatree.multipoles[leaf_idx * ncoeffs..(leaf_idx + 1) * ncoeffs].as_ptr();
                assert_eq!(result, expected);

                let result = datatree.locals.as_ptr().add(leaf_idx * ncoeffs);
                let expected =
                    datatree.locals[leaf_idx * ncoeffs..(leaf_idx + 1) * ncoeffs].as_ptr();
                assert_eq!(result, expected);
            }

            // Test that level expansion information is referring to correct memory, and is in correct shape
            assert_eq!(
                datatree.level_multipoles.len() as u64,
                datatree.fmm.tree().get_depth() + 1
            );
            assert_eq!(
                datatree.level_locals.len() as u64,
                datatree.fmm.tree().get_depth() + 1
            );

            assert_eq!(
                datatree.level_multipoles[(datatree.fmm.tree().get_depth()) as usize].len(),
                nleaves
            );
            assert_eq!(
                datatree.level_locals[(datatree.fmm.tree().get_depth()) as usize].len(),
                nleaves
            );

            // Test that points are being assigned correctly
            assert_eq!(datatree.potentials_send_pointers.len(), nleaves);
            assert_eq!(datatree.potentials.len(), npoints);
        }
    }

    #[test]
    fn test_fmm_linear() {
        let npoints = 10000;
        let points = points_fixture::<f64>(npoints, None, None);
        let global_idxs = (0..npoints).collect_vec();
        let charges = vec![1.0; npoints];

        let order = 2;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let adaptive = false;
        let ncrit = 150;

        let depth = 3;
        let kernel = Laplace3dKernel::default();

        let tree = SingleNodeTree::new(
            points.data(),
            adaptive,
            Some(ncrit),
            Some(depth),
            &global_idxs[..],
        );

        let m2l_data_fft =
            FftFieldTranslationKiFmm::new(kernel.clone(), order, *tree.get_domain(), alpha_inner);

        let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data_fft);

        // Form charge dict, matching charges with their associated global indices
        let charge_dict = build_charge_dict(&global_idxs[..], &charges[..]);

        let datatree = FmmDataLinear::new(fmm, &charge_dict).unwrap();

        let times = datatree.run(false);

        // Test that direct computation is close to the FMM.
        let leaf = &datatree.fmm.tree.get_all_leaves().unwrap()[0];
        let leaf_idx = datatree.fmm.tree().get_leaf_index(leaf).unwrap();
        let key_idx = datatree.fmm.tree().get_index(leaf).unwrap();

        // Examine local expansion
        let ncoeffs = datatree.fmm.m2l.ncoeffs(order);
        let local_expansion = &datatree.locals[key_idx*ncoeffs..(key_idx+1)*ncoeffs];
        // println!("leaf {:?}", local_expansion);
        
        let (l, r) = datatree.charge_index_pointer[*leaf_idx];

        let potentials = &datatree.potentials[l..r];

        println!(
            "HERE {:?}",
            potentials,
        );

        let coordinates = datatree.fmm.tree().get_all_coordinates().unwrap();
        let (l, r) = datatree.charge_index_pointer[*leaf_idx];
        let leaf_coords = &coordinates[l*3..r*3];
        // println!("HERE {:?}", leaf_coords);

        // let pts = datatree.fmm.tree().get_points(leaf).unwrap();
        // let leaf_coordinates = pts
        //     .iter()
        //     .map(|p| p.coordinate)
        //     .flat_map(|[x, y, z]| vec![x, y, z])
        //     .collect_vec();

        // let ntargets = leaf_coordinates.len() / datatree.fmm.kernel.space_dimension();

        //   let leaf_coordinates = unsafe {
        //       rlst_pointer_mat!['static, f64, leaf_coordinates.as_ptr(), (ntargets, datatree.fmm.kernel.space_dimension()), (datatree.fmm.kernel.space_dimension(), 1)]
        //   }.eval();

        //   let mut direct = vec![0f64; pts.len()];
        //   let all_point_coordinates = points_fixture::<f64>(npoints, None, None);

        //   let all_charges = charge_dict.into_values().collect_vec();

        //   let kernel = Laplace3dKernel::default();

        //   kernel.evaluate_st(
        //       EvalType::Value,
        //       all_point_coordinates.data(),
        //       leaf_coordinates.data(),
        //       &all_charges[..],
        //       &mut direct[..],
        //   );

        // println!("direct {:?}", direct);
        // println!("potentials {:?}", potentials);
        
        //   let abs_error: f32 = potentials
        //       .data()
        //       .iter()
        //       .zip(direct.iter())
        //       .map(|(a, b)| (a - b).abs())
        //       .sum();
        //   let rel_error: f32 = abs_error / (direct.iter().sum::<f32>());

        //   assert!(rel_error <= 1e-4);

        // println!("potentials {:?}", datatree.potentials);

        // let s = Instant::now();
        // let times = datatree.run(true);
        // println!("linear upward pass {:?} {:?}", s.elapsed(), times.unwrap());

        // let kernel = Laplace3dKernel::default();

        // let tree = SingleNodeTree::new(
        //     points.data(),
        //     adaptive,
        //     Some(ncrit),
        //     Some(depth),
        //     &global_idxs[..],
        // );

        // let m2l_data_fft =
        //     FftFieldTranslationKiFmm::new(kernel.clone(), order, *tree.get_domain(), alpha_inner);
        // let fmm = KiFmm::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data_fft);

        // // // Form charge dict, matching charges with their associated global indices
        // let charge_dict = build_charge_dict(&global_idxs[..], &charges[..]);
        // let s = Instant::now();
        // let old_datatree = FmmData::new(fmm, &charge_dict);
        // println!("old data tree setup {:?}", s.elapsed());

        // let &idx = datatree.fmm.tree().key_to_index.get(&ROOT).unwrap();
        // let old_leaf = old_datatree.fmm.tree().get_all_leaves().unwrap()[idx];
        // let old_key = old_datatree.fmm.tree().get_all_keys().unwrap()[idx];
        // // let old_points = old_datatree.points.get(&old_leaf).unwrap();
        // // let old_points = old_points.iter().map(|p| p.coordinate).flat_map(|[x, y, z]| vec![x, y, z]).collect_vec();

        // let new_leaf = datatree.fmm.tree().get_all_leaves().unwrap()[idx];
        // let new_key = datatree.fmm.tree().get_all_keys().unwrap()[idx];
        // // println!("old {:?} new {:?} keys", old_key, new_key);

        // let (l, r) = datatree.charge_index_pointer[idx];
        // // let new_points = &datatree.fmm.tree().get_all_coordinates().unwrap()[l*3..r*3];

        // let s = Instant::now();
        // let times = old_datatree.run(true);
        // println!("old upward pass {:?} {:?}", s.elapsed(), times.unwrap());

        // // Check potentials
        // let midx = datatree.fmm.tree().key_to_index.get(&new_key).unwrap();
        // // let (l, r) = datatree.expansion_index_pointer[*midx];
        // let ncoeffs = datatree.fmm.m2l.ncoeffs(datatree.fmm.order);
        // let new_multipole = &datatree.multipoles[midx * ncoeffs..(midx + 1) * ncoeffs];
        // let old_multipole = old_datatree
        //     .multipoles
        //     .get(&old_key)
        //     .unwrap()
        //     .deref()
        //     .lock()
        //     .unwrap();

        // // println!("HERE {:?} {:?}", old_key, old_multipole.data());
        // // println!("HERE {:?} {:?}", new_key, new_multipole);
        // let abs_error: f64 = old_multipole
        //     .data()
        //     .iter()
        //     .zip(new_multipole.iter())
        //     .map(|(a, b)| (a - b).abs())
        //     .sum();

        // let rel_error = abs_error / (old_multipole.data().iter().sum::<f64>());
        // println!("rel error {:?}", rel_error);

        assert!(false)
    }
}
