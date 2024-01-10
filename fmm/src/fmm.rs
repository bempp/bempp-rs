//! Implementation of FmmData and Fmm traits.
use itertools::Itertools;
use num::Float;
use rlst_common::types::Scalar;
use std::time::Instant;

use rlst_dense::{
    array::{empty_array, Array},
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_dynamic_array2,
    traits::{MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, Shape, UnsafeRandomAccessMut},
};

use bempp_traits::{
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, FmmLoop, KiFmm as KiFmmTrait, SourceTranslation, TargetTranslation, TimeDict},
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
    types::EvalType,
};

use bempp_tree::{constants::ROOT, types::single_node::SingleNodeTree};

use crate::types::{FmmDataAdaptive, FmmDataUniform, FmmDataUniformMatrix, KiFmmLinearMatrix};
use crate::{pinv::pinv, types::KiFmmLinear};

/// Implementation of constructor for single node KiFMM
impl<T, U, V> KiFmmLinear<SingleNodeTree<V>, T, U, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V>,
    U: FieldTranslationData<T>,
    V: Scalar<Real = V> + Default + Float + rlst_blis::interface::gemm::Gemm,
    Array<V, BaseArray<V, VectorContainer<V>, 2>, 2>: MatrixSvd<Item = V>,
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

        // Compute upward check to equivalent, and downward check to equivalent Gram matrices
        // as well as their inverses using DGESVD.
        let mut uc2e_t = rlst_dynamic_array2!(V, [ncheck_surface, nequiv_surface]);
        kernel.assemble_st(
            EvalType::Value,
            &upward_equivalent_surface[..],
            &upward_check_surface[..],
            uc2e_t.data_mut(),
        );

        // Need to tranapose so that rows correspond to targets and columns to sources
        let mut uc2e = rlst_dynamic_array2!(V, [nequiv_surface, ncheck_surface]);
        uc2e.fill_from(uc2e_t.transpose());

        let mut dc2e_t = rlst_dynamic_array2!(V, [ncheck_surface, nequiv_surface]);
        kernel.assemble_st(
            EvalType::Value,
            &downward_equivalent_surface[..],
            &downward_check_surface[..],
            dc2e_t.data_mut(),
        );

        // Need to tranapose so that rows correspond to targets and columns to sources
        let mut dc2e = rlst_dynamic_array2!(V, [nequiv_surface, ncheck_surface]);
        dc2e.fill_from(dc2e_t.transpose());

        let (s, ut, v) = pinv::<V>(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(V, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = V::from_real(s[i]);
        }
        let uc2e_inv_1 = empty_array::<V, 2>().simple_mult_into_resize(v.view(), mat_s.view());
        let uc2e_inv_2 = ut;

        let (s, ut, v) = pinv::<V>(&dc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(V, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = V::from_real(s[i]);
        }

        let dc2e_inv_1 = empty_array::<V, 2>().simple_mult_into_resize(v.view(), mat_s.view());
        let dc2e_inv_2 = ut;

        // Calculate M2M/L2L matrices
        let children = ROOT.children();
        let mut m2m = rlst_dynamic_array2!(V, [nequiv_surface, 8 * nequiv_surface]);
        let mut l2l = Vec::new();

        for (i, child) in children.iter().enumerate() {
            let child_upward_equivalent_surface =
                child.compute_surface(tree.get_domain(), order, alpha_inner);
            let child_downward_check_surface =
                child.compute_surface(tree.get_domain(), order, alpha_inner);

            let mut pc2ce_t = rlst_dynamic_array2!(V, [ncheck_surface, nequiv_surface]);

            kernel.assemble_st(
                EvalType::Value,
                &child_upward_equivalent_surface,
                &upward_check_surface,
                pc2ce_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut pc2ce = rlst_dynamic_array2!(V, [nequiv_surface, ncheck_surface]);
            pc2ce.fill_from(pc2ce_t.transpose());

            let tmp = empty_array::<V, 2>().simple_mult_into_resize(
                uc2e_inv_1.view(),
                empty_array::<V, 2>().simple_mult_into_resize(uc2e_inv_2.view(), pc2ce.view()),
            );
            let l = i * nequiv_surface * nequiv_surface;
            let r = l + nequiv_surface * nequiv_surface;

            m2m.data_mut()[l..r].copy_from_slice(tmp.data());

            let mut cc2pe_t = rlst_dynamic_array2!(V, [ncheck_surface, nequiv_surface]);

            kernel.assemble_st(
                EvalType::Value,
                &downward_equivalent_surface,
                &child_downward_check_surface,
                cc2pe_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut cc2pe = rlst_dynamic_array2!(V, [nequiv_surface, ncheck_surface]);
            cc2pe.fill_from(cc2pe_t.transpose());
            let mut tmp = empty_array::<V, 2>().simple_mult_into_resize(
                dc2e_inv_1.view(),
                empty_array::<V, 2>().simple_mult_into_resize(dc2e_inv_2.view(), cc2pe.view()),
            );
            tmp.data_mut()
                .iter_mut()
                .for_each(|d| *d *= kernel.scale(child.level()));

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

/// Implementation of constructor for single node KiFMM
impl<T, U, V> KiFmmLinearMatrix<SingleNodeTree<V>, T, U, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V>,
    U: FieldTranslationData<T>,
    V: Scalar<Real = V> + Default + Float + rlst_blis::interface::gemm::Gemm,
    Array<V, BaseArray<V, VectorContainer<V>, 2>, 2>: MatrixSvd<Item = V>,
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

        // Compute upward check to equivalent, and downward check to equivalent Gram matrices
        // as well as their inverses using DGESVD.
        let mut uc2e_t = rlst_dynamic_array2!(V, [ncheck_surface, nequiv_surface]);
        kernel.assemble_st(
            EvalType::Value,
            &upward_equivalent_surface,
            &upward_check_surface,
            uc2e_t.data_mut(),
        );

        // Need to tranapose so that rows correspond to targets and columns to sources
        let mut uc2e = rlst_dynamic_array2!(V, [nequiv_surface, ncheck_surface]);
        uc2e.fill_from(uc2e_t.transpose());

        let mut dc2e_t = rlst_dynamic_array2!(V, [ncheck_surface, nequiv_surface]);
        kernel.assemble_st(
            EvalType::Value,
            &downward_equivalent_surface,
            &downward_check_surface,
            dc2e_t.data_mut(),
        );

        // Need to tranapose so that rows correspond to targets and columns to sources
        let mut dc2e = rlst_dynamic_array2!(V, [nequiv_surface, ncheck_surface]);
        dc2e.fill_from(dc2e_t.transpose());

        let (s, ut, v) = pinv::<V>(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(V, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = V::from_real(s[i]);
        }
        let uc2e_inv_1 = empty_array::<V, 2>().simple_mult_into_resize(v.view(), mat_s.view());
        let uc2e_inv_2 = ut;

        let uc2e_inv_1_shape = uc2e_inv_1.shape();
        let uc2e_inv_2_shape = uc2e_inv_2.shape();

        // TODO: this can be simplified
        let uc2e_inv_1 = uc2e_inv_1
            .data()
            .iter()
            .map(|x| V::from(*x).unwrap())
            .collect_vec();
        let mut uc2e_inv_1_mat = rlst_dynamic_array2!(V, uc2e_inv_1_shape);
        for j in 0..uc2e_inv_1_shape[1] {
            for i in 0..uc2e_inv_1_shape[0] {
                unsafe {
                    *uc2e_inv_1_mat.get_unchecked_mut([i, j]) =
                        uc2e_inv_1[uc2e_inv_1_shape[0] * j + i]
                }
            }
        }

        let uc2e_inv_2 = uc2e_inv_2
            .data()
            .iter()
            .map(|x| V::from(*x).unwrap())
            .collect_vec();

        let mut uc2e_inv_2_mat = rlst_dynamic_array2!(V, uc2e_inv_2_shape);
        for j in 0..uc2e_inv_2_shape[1] {
            for i in 0..uc2e_inv_2_shape[0] {
                unsafe {
                    *uc2e_inv_2_mat.get_unchecked_mut([i, j]) =
                        uc2e_inv_2[uc2e_inv_2_shape[0] * j + i]
                }
            }
        }

        let (s, ut, v) = pinv::<V>(&dc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(V, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = V::from_real(s[i]);
        }

        let dc2e_inv_1 = empty_array::<V, 2>().simple_mult_into_resize(v.view(), mat_s.view());
        let dc2e_inv_2 = ut;

        let dc2e_inv_1_shape = dc2e_inv_1.shape();
        let dc2e_inv_2_shape = dc2e_inv_2.shape();

        let dc2e_inv_1 = dc2e_inv_1
            .data()
            .iter()
            .map(|x| V::from(*x).unwrap())
            .collect_vec();
        let mut dc2e_inv_1_mat = rlst_dynamic_array2!(V, dc2e_inv_1_shape);
        for j in 0..dc2e_inv_1_shape[1] {
            for i in 0..dc2e_inv_1_shape[0] {
                unsafe {
                    *dc2e_inv_1_mat.get_unchecked_mut([i, j]) =
                        dc2e_inv_1[dc2e_inv_1_shape[0] * j + i]
                }
            }
        }

        let dc2e_inv_2 = dc2e_inv_2
            .data()
            .iter()
            .map(|x| V::from(*x).unwrap())
            .collect_vec();
        let mut dc2e_inv_2_mat = rlst_dynamic_array2!(V, dc2e_inv_2_shape);
        for j in 0..dc2e_inv_2_shape[1] {
            for i in 0..dc2e_inv_2_shape[0] {
                unsafe {
                    *dc2e_inv_2_mat.get_unchecked_mut([i, j]) =
                        dc2e_inv_2[dc2e_inv_2_shape[0] * j + i]
                }
            }
        }

        // Calculate M2M/L2L matrices
        let children = ROOT.children();
        let mut m2m = Vec::new();
        let mut l2l = Vec::new();

        for child in children.iter() {
            let child_upward_equivalent_surface =
                child.compute_surface(tree.get_domain(), order, alpha_inner);
            let child_downward_check_surface =
                child.compute_surface(tree.get_domain(), order, alpha_inner);

            let mut pc2ce_t = rlst_dynamic_array2!(V, [ncheck_surface, nequiv_surface]);

            kernel.assemble_st(
                EvalType::Value,
                &child_upward_equivalent_surface,
                &upward_check_surface,
                pc2ce_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut pc2ce = rlst_dynamic_array2!(V, [nequiv_surface, ncheck_surface]);
            pc2ce.fill_from(pc2ce_t.transpose());

            let tmp = empty_array::<V, 2>().simple_mult_into_resize(
                uc2e_inv_1_mat.view(),
                empty_array::<V, 2>().simple_mult_into_resize(uc2e_inv_2_mat.view(), pc2ce.view()),
            );
            m2m.push(tmp);

            let mut cc2pe_t = rlst_dynamic_array2!(V, [ncheck_surface, nequiv_surface]);

            kernel.assemble_st(
                EvalType::Value,
                &downward_equivalent_surface,
                &child_downward_check_surface,
                cc2pe_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut cc2pe = rlst_dynamic_array2!(V, [nequiv_surface, ncheck_surface]);
            cc2pe.fill_from(cc2pe_t.transpose());
            let mut tmp = empty_array::<V, 2>().simple_mult_into_resize(
                dc2e_inv_1_mat.view(),
                empty_array::<V, 2>().simple_mult_into_resize(dc2e_inv_2_mat.view(), cc2pe.view()),
            );
            tmp.data_mut()
                .iter_mut()
                .for_each(|d| *d *= kernel.scale(child.level()));

            l2l.push(tmp);
        }

        Self {
            order,
            uc2e_inv_1: uc2e_inv_1_mat,
            uc2e_inv_2: uc2e_inv_2_mat,
            dc2e_inv_1: dc2e_inv_1_mat,
            dc2e_inv_2: dc2e_inv_2_mat,
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

impl<T, U, V, W> KiFmmTrait for KiFmmLinearMatrix<T, U, V, W>
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

impl<T, U, V, W> Fmm for KiFmmLinearMatrix<T, U, V, W>
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

impl<T, U> FmmLoop for FmmDataAdaptive<T, U>
where
    T: Fmm,
    U: Scalar<Real = U> + Float + Default,
    FmmDataAdaptive<T, U>: SourceTranslation + FieldTranslation<U> + TargetTranslation,
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
                let mut p2l_time = 0;

                for level in 2..=depth {
                    if level > 2 {
                        let start = Instant::now();
                        self.l2l(level);
                        l2l_time += start.elapsed().as_millis();
                    }
                    let start = Instant::now();
                    self.p2l(level);
                    p2l_time += start.elapsed().as_millis();

                    let start = Instant::now();
                    self.m2l(level);
                    m2l_time += start.elapsed().as_millis();
                }

                times.insert("l2l".to_string(), l2l_time);
                times.insert("m2l".to_string(), m2l_time);
                times.insert("p2l".to_string(), p2l_time);

                // Leaf level computations
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
                    self.p2l(level);
                }
                // Leaf level computations

                // Sum all potential contributions
                self.m2p();
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

impl<T, U> FmmLoop for FmmDataUniform<T, U>
where
    T: Fmm,
    U: Scalar<Real = U> + Float + Default,
    FmmDataUniform<T, U>: SourceTranslation + FieldTranslation<U> + TargetTranslation,
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

                // Sum all potential contributions
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
                // Sum all potential contributions
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

impl<T, U> FmmLoop for FmmDataUniformMatrix<T, U>
where
    T: Fmm,
    U: Scalar<Real = U> + Float + Default,
    FmmDataUniformMatrix<T, U>: SourceTranslation + FieldTranslation<U> + TargetTranslation,
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

                // Sum all potential contributions
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
                // Sum all potential contributions
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

#[cfg(test)]
mod test {

    use super::*;

    use bempp_field::types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm};
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::implementations::helpers::{points_fixture, points_fixture_sphere};

    use crate::charge::build_charge_dict;

    #[allow(clippy::too_many_arguments)]
    fn test_uniform_f64(
        points: &Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
        charges: &[f64],
        global_idxs: &[usize],
        order: usize,
        alpha_inner: f64,
        alpha_outer: f64,
        sparse: bool,
        depth: u64,
    ) {
        // Test with FFT based field translation
        {
            let tree =
                SingleNodeTree::new(points.data(), false, None, Some(depth), global_idxs, sparse);

            let kernel = Laplace3dKernel::default();
            let m2l_data: FftFieldTranslationKiFmm<f64, Laplace3dKernel<f64>> =
                FftFieldTranslationKiFmm::new(
                    kernel.clone(),
                    order,
                    *tree.get_domain(),
                    alpha_inner,
                );

            let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

            // Form charge dict, matching charges with their associated global indices
            let charge_dict = build_charge_dict(global_idxs, charges);

            let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

            datatree.run(false);

            // Test that direct computation is close to the FMM.
            let mut test_idx_vec = Vec::new();
            for (idx, index_pointer) in datatree.charge_index_pointer.iter().enumerate() {
                if index_pointer.1 - index_pointer.0 > 0 {
                    test_idx_vec.push(idx);
                }
            }
            let leaf = &datatree.fmm.tree().get_all_leaves().unwrap()[test_idx_vec[3]];

            let leaf_idx = datatree.fmm.tree().get_leaf_index(leaf).unwrap();

            let (l, r) = datatree.charge_index_pointer[*leaf_idx];

            let potentials = &datatree.potentials[l..r];

            let coordinates = datatree.fmm.tree().get_all_coordinates().unwrap();
            let (l, r) = datatree.charge_index_pointer[*leaf_idx];
            let leaf_coordinates = &coordinates[l * 3..r * 3];

            let ntargets = leaf_coordinates.len() / datatree.fmm.kernel.space_dimension();

            let mut direct = vec![0f64; ntargets];

            let all_charges = charge_dict.into_values().collect_vec();

            let kernel = Laplace3dKernel::default();

            kernel.evaluate_st(
                EvalType::Value,
                points.data(),
                leaf_coordinates,
                &all_charges,
                &mut direct,
            );

            let abs_error: f64 = potentials
                .iter()
                .zip(direct.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());
            assert!(rel_error <= 1e-5);
        }

        // Test with SVD field translation
        {
            let tree =
                SingleNodeTree::new(points.data(), false, None, Some(depth), global_idxs, sparse);

            let kernel = Laplace3dKernel::default();

            let m2l_data = SvdFieldTranslationKiFmm::new(
                kernel.clone(),
                Some(1000),
                order,
                *tree.get_domain(),
                alpha_inner,
            );

            let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

            // Form charge dict, matching charges with their associated global indices
            let charge_dict = build_charge_dict(global_idxs, charges);

            let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

            datatree.run(false);

            // Test that direct computation is close to the FMM.
            let mut test_idx_vec = Vec::new();
            for (idx, index_pointer) in datatree.charge_index_pointer.iter().enumerate() {
                if index_pointer.1 - index_pointer.0 > 0 {
                    test_idx_vec.push(idx);
                }
            }
            let leaf = &datatree.fmm.tree().get_all_leaves().unwrap()[test_idx_vec[3]];

            let leaf_idx = datatree.fmm.tree().get_leaf_index(leaf).unwrap();

            let (l, r) = datatree.charge_index_pointer[*leaf_idx];

            let potentials = &datatree.potentials[l..r];

            let coordinates = datatree.fmm.tree().get_all_coordinates().unwrap();
            let (l, r) = datatree.charge_index_pointer[*leaf_idx];
            let leaf_coordinates = &coordinates[l * 3..r * 3];

            let ntargets = leaf_coordinates.len() / datatree.fmm.kernel.space_dimension();

            let mut direct = vec![0f64; ntargets];

            let all_charges = charge_dict.into_values().collect_vec();

            datatree.fmm.kernel().evaluate_st(
                EvalType::Value,
                points.data(),
                leaf_coordinates,
                &all_charges,
                &mut direct,
            );

            let abs_error: f64 = potentials
                .iter()
                .zip(direct.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());
            assert!(rel_error <= 1e-5);
        }
    }

    fn test_adaptive_f64(
        points: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
        charges: &[f64],
        global_idxs: &[usize],
        ncrit: u64,
        order: usize,
        alpha_inner: f64,
        alpha_outer: f64,
    ) {
        // Test with FFT based field translation
        {
            let tree =
                SingleNodeTree::new(points.data(), true, Some(ncrit), None, global_idxs, false);

            let kernel = Laplace3dKernel::default();
            let m2l_data: FftFieldTranslationKiFmm<f64, Laplace3dKernel<f64>> =
                FftFieldTranslationKiFmm::new(
                    kernel.clone(),
                    order,
                    *tree.get_domain(),
                    alpha_inner,
                );

            let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

            // Form charge dict, matching charges with their associated global indices
            let charge_dict = build_charge_dict(global_idxs, charges);

            let datatree = FmmDataAdaptive::new(fmm, &charge_dict).unwrap();

            datatree.run(false);

            // Test that direct computation is close to the FMM.
            let mut test_idx_vec = Vec::new();
            for (idx, index_pointer) in datatree.charge_index_pointer.iter().enumerate() {
                if index_pointer.1 - index_pointer.0 > 0 {
                    test_idx_vec.push(idx);
                }
            }

            let leaf = &datatree.fmm.tree().get_all_leaves().unwrap()[test_idx_vec[3]];

            let leaf_idx = datatree.fmm.tree().get_leaf_index(leaf).unwrap();

            let (l, r) = datatree.charge_index_pointer[*leaf_idx];

            let potentials = &datatree.potentials[l..r];

            let coordinates = datatree.fmm.tree().get_all_coordinates().unwrap();
            let (l, r) = datatree.charge_index_pointer[*leaf_idx];
            let leaf_coordinates = &coordinates[l * 3..r * 3];

            let ntargets = leaf_coordinates.len() / datatree.fmm.kernel.space_dimension();

            let mut direct = vec![0f64; ntargets];

            let all_charges = charge_dict.into_values().collect_vec();

            let kernel = Laplace3dKernel::default();

            kernel.evaluate_st(
                EvalType::Value,
                points.data(),
                leaf_coordinates,
                &all_charges,
                &mut direct,
            );

            let abs_error: f64 = potentials
                .iter()
                .zip(direct.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());
            assert!(rel_error <= 1e-5);
        }

        // Test with SVD field translation
        {
            let tree =
                SingleNodeTree::new(points.data(), true, Some(ncrit), None, global_idxs, false);
            let kernel = Laplace3dKernel::default();

            let m2l_data = SvdFieldTranslationKiFmm::new(
                kernel.clone(),
                Some(1000),
                order,
                *tree.get_domain(),
                alpha_inner,
            );

            let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

            // Form charge dict, matching charges with their associated global indices
            let charge_dict = build_charge_dict(global_idxs, charges);

            let datatree = FmmDataAdaptive::new(fmm, &charge_dict).unwrap();

            datatree.run(false);

            // Test that direct computation is close to the FMM.
            let mut test_idx_vec = Vec::new();
            for (idx, index_pointer) in datatree.charge_index_pointer.iter().enumerate() {
                if index_pointer.1 - index_pointer.0 > 0 {
                    test_idx_vec.push(idx);
                }
            }
            let leaf = &datatree.fmm.tree().get_all_leaves().unwrap()[test_idx_vec[3]];

            let leaf_idx = datatree.fmm.tree().get_leaf_index(leaf).unwrap();

            let (l, r) = datatree.charge_index_pointer[*leaf_idx];

            let potentials = &datatree.potentials[l..r];

            let coordinates = datatree.fmm.tree().get_all_coordinates().unwrap();
            let (l, r) = datatree.charge_index_pointer[*leaf_idx];
            let leaf_coordinates = &coordinates[l * 3..r * 3];

            let ntargets = leaf_coordinates.len() / datatree.fmm.kernel.space_dimension();

            let mut direct = vec![0f64; ntargets];

            let all_charges = charge_dict.into_values().collect_vec();

            let kernel = Laplace3dKernel::default();

            kernel.evaluate_st(
                EvalType::Value,
                points.data(),
                leaf_coordinates,
                &all_charges,
                &mut direct,
            );

            let abs_error: f64 = potentials
                .iter()
                .zip(direct.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());
            assert!(rel_error <= 1e-5);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn test_uniform_matrix_f64(
        order: usize,
        alpha_inner: f64,
        alpha_outer: f64,
        depth: u64,
        sparse: bool,
        points: &[f64],
        global_idxs: &[usize],
        charge_mat: &Vec<Vec<f64>>,
    ) {
        // SVD based field translations
        {
            let ncharge_vecs = charge_mat.len();

            let kernel = Laplace3dKernel::default();

            // Create a tree
            let tree = SingleNodeTree::new(points, false, None, Some(depth), global_idxs, sparse);

            // Precompute the M2L data
            let m2l_data = SvdFieldTranslationKiFmm::new(
                kernel.clone(),
                Some(1000),
                order,
                *tree.get_domain(),
                alpha_inner,
            );

            let fmm =
                KiFmmLinearMatrix::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

            // Form charge dict, matching charges with their associated global indices
            let charge_dicts: Vec<_> = (0..ncharge_vecs)
                .map(|i| build_charge_dict(global_idxs, &charge_mat[i]))
                .collect();

            // Associate data with the FMM
            let datatree = FmmDataUniformMatrix::new(fmm, &charge_dicts).unwrap();

            datatree.run(false);

            // Test that direct computation is close to the FMM.
            let mut test_idx_vec = Vec::new();
            for (idx, index_pointer) in datatree.charge_index_pointer.iter().enumerate() {
                if index_pointer.1 - index_pointer.0 > 0 {
                    test_idx_vec.push(idx);
                }
            }
            let leaf = &datatree.fmm.tree().get_all_leaves().unwrap()[test_idx_vec[3]];

            let &leaf_idx = datatree.fmm.tree().get_leaf_index(leaf).unwrap();
            let (l, r) = datatree.charge_index_pointer[leaf_idx];

            let coordinates = datatree.fmm.tree().get_all_coordinates().unwrap();
            let leaf_coordinates = &coordinates[l * 3..r * 3];

            let ntargets = leaf_coordinates.len() / datatree.fmm.kernel.space_dimension();

            for (i, charge_dict) in charge_dicts
                .iter()
                .enumerate()
                .take(datatree.ncharge_vectors)
            {
                let potentials_ptr =
                    datatree.potentials_send_pointers[i * datatree.nleaves + leaf_idx].raw;
                let potentials = unsafe { std::slice::from_raw_parts(potentials_ptr, ntargets) };

                let all_charges = &charge_dict.values().cloned().collect_vec();

                let mut direct = vec![0f64; ntargets];

                datatree.fmm.kernel().evaluate_st(
                    EvalType::Value,
                    points,
                    leaf_coordinates,
                    all_charges,
                    &mut direct,
                );

                let abs_error: f64 = potentials
                    .iter()
                    .zip(direct.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());
                assert!(rel_error <= 1e-5);
            }
        }
    }

    #[test]
    fn test_uniform() {
        let npoints = 10000;

        let global_idxs = (0..npoints).collect_vec();
        let charges = vec![1.0; npoints];

        let order = 6;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;

        // Test case where points are distributed on surface of a sphere
        let points_sphere = points_fixture_sphere::<f64>(npoints);
        test_uniform_f64(
            &points_sphere,
            &charges,
            &global_idxs,
            order,
            alpha_inner,
            alpha_outer,
            true,
            3,
        );
        test_uniform_f64(
            &points_sphere,
            &charges,
            &global_idxs,
            order,
            alpha_inner,
            alpha_outer,
            false,
            3,
        );

        // Test case where points are distributed randomly in a box
        let points_cloud = points_fixture::<f64>(npoints, None, None);
        test_uniform_f64(
            &points_cloud,
            &charges,
            &global_idxs,
            order,
            alpha_inner,
            alpha_outer,
            true,
            3,
        );
        test_uniform_f64(
            &points_cloud,
            &charges,
            &global_idxs,
            order,
            alpha_inner,
            alpha_outer,
            false,
            3,
        );

        // Test matrix input
        let points = points_fixture::<f64>(npoints, None, None);
        let ncharge_vecs = 3;

        let mut charge_mat = vec![vec![0.0; npoints]; ncharge_vecs];
        charge_mat
            .iter_mut()
            .enumerate()
            .for_each(|(i, charge_mat_i)| *charge_mat_i = vec![i as f64 + 1.0; npoints]);

        test_uniform_matrix_f64(
            order,
            alpha_inner,
            alpha_outer,
            3,
            true,
            points.data(),
            &global_idxs,
            &charge_mat,
        )
    }

    #[test]
    fn test_adaptive() {
        let npoints = 10000;

        let global_idxs = (0..npoints).collect_vec();
        let charges = vec![1.0; npoints];

        let order = 6;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let ncrit = 100;

        // Test case where points are distributed on surface of a sphere
        let points_sphere = points_fixture_sphere::<f64>(npoints);
        test_adaptive_f64(
            points_sphere,
            &charges,
            &global_idxs,
            ncrit,
            order,
            alpha_inner,
            alpha_outer,
        );

        // Test case where points are distributed randomly in a box
        let points_cloud = points_fixture::<f64>(npoints, None, None);
        test_adaptive_f64(
            points_cloud,
            &charges,
            &global_idxs,
            ncrit,
            order,
            alpha_inner,
            alpha_outer,
        );
    }
}
