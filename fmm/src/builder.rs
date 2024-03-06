use std::collections::HashMap;

use bempp_field::helpers::ncoeffs_kifmm;

use bempp_traits::{
    field::SourceToTargetData,
    kernel::Kernel,
    tree::{FmmTree, Tree},
    types::EvalType,
};
use bempp_tree::{
    constants::ROOT,
    types::{domain::Domain, morton::MortonKey, single_node::SingleNodeTreeNew},
};
use rlst_dense::{
    array::{empty_array, Array},
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_dynamic_array2,
    traits::{MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, Shape},
};

use crate::{
    constants::{ALPHA_INNER, ALPHA_OUTER},
    fmm::KiFmm,
    helpers::homogenous_kernel_scale,
    pinv::pinv,
    traits::FmmScalar,
    tree::SingleNodeFmmTree,
    types::{Charges, Coordinates, SendPtrMut},
};

#[derive(Clone, Copy)]
pub enum FmmEvaluationMode {
    Vector,
    Matrix(usize),
}

#[derive(Default)]
pub struct KiFmmBuilderSingleNode<'builder, T, U, V>
where
    T: SourceToTargetData<V>,
    U: FmmScalar,
    V: Kernel,
{
    tree: Option<SingleNodeFmmTree<U>>,
    charges: Option<&'builder Charges<U>>,
    source_to_target: Option<T>,
    domain: Option<Domain<U>>,
    kernel: Option<V>,
    expansion_order: Option<usize>,
    ncoeffs: Option<usize>,
    eval_type: Option<EvalType>,
    eval_mode: Option<FmmEvaluationMode>,
}

impl<'builder, T, U, V> KiFmmBuilderSingleNode<'builder, T, U, V>
where
    T: SourceToTargetData<V, Domain = Domain<U>> + Default,
    U: FmmScalar,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: Kernel<T = U> + Clone + Default,
{
    // Start building with mandatory parameters
    pub fn new() -> Self {
        KiFmmBuilderSingleNode {
            tree: None,
            domain: None,
            source_to_target: None,
            kernel: None,
            expansion_order: None,
            ncoeffs: None,
            eval_type: None,
            charges: None,
            eval_mode: None,
        }
    }

    pub fn tree(
        mut self,
        sources: &Coordinates<U>,
        targets: &Coordinates<U>,
        charges: &'builder Charges<U>,
        n_crit: Option<u64>,
        sparse: bool,
    ) -> Self {
        // Source and target trees calcualted over the same domain
        let source_domain = Domain::from_local_points(sources.data());
        let target_domain = Domain::from_local_points(targets.data());

        // Calculate union of domains for source and target points, needed to define operators
        let domain = source_domain.union(&target_domain);
        self.domain = Some(domain);

        let source_tree = SingleNodeTreeNew::new(sources.data(), n_crit, sparse, self.domain);
        let target_tree = SingleNodeTreeNew::new(targets.data(), n_crit, sparse, self.domain);

        let fmm_tree = SingleNodeFmmTree {
            source_tree,
            target_tree,
            domain,
        };

        self.tree = Some(fmm_tree);
        self.charges = Some(charges);

        let [ncharges, nmatvec] = charges.shape();

        if nmatvec > 1 {
            self.eval_mode = Some(FmmEvaluationMode::Matrix(nmatvec))
        } else {
            self.eval_mode = Some(FmmEvaluationMode::Vector)
        }

        self
    }

    pub fn parameters(
        mut self,
        expansion_order: usize,
        kernel: V,
        eval_type: EvalType,
        mut source_to_target: T,
    ) -> Result<Self, String> {
        if self.tree.is_none() {
            Err("Must build tree before specifying FMM parameters".to_string())
        } else {
            self.expansion_order = Some(expansion_order);
            self.ncoeffs = Some(ncoeffs_kifmm(expansion_order));
            self.kernel = Some(kernel);
            self.eval_type = Some(eval_type);

            // Set source to target metadata
            // Set the expansion order
            source_to_target.set_expansion_order(self.expansion_order.unwrap());

            // Set the associated kernel
            let kernel = self.kernel.as_ref().unwrap().clone();
            source_to_target.set_kernel(kernel);

            // Compute the transfer vectors

            // Compute the field translation operators
            source_to_target.set_operator_data(self.expansion_order.unwrap(), self.domain.unwrap());

            self.source_to_target = Some(source_to_target);

            Ok(self)
        }
    }

    // Finalize and build the KiFmm
    pub fn build(self) -> Result<KiFmm<SingleNodeFmmTree<U>, T, V, U>, String> {
        if self.tree.is_none() || self.source_to_target.is_none() || self.expansion_order.is_none()
        {
            Err("Missing fields for constructing KiFmm".to_string())
        } else {
            // Configure with tree, expansion parameters and source to target field translation operators
            let mut result = KiFmm {
                tree: self.tree.unwrap(),
                expansion_order: self.expansion_order.unwrap(),
                ncoeffs: self.ncoeffs.unwrap(),
                kernel: self.kernel.unwrap(),
                source_to_target_data: self.source_to_target.unwrap(),
                eval_mode: self.eval_mode.unwrap(),
                ..Default::default()
            };

            // Compute the source to source and target to target field translation operators
            result.set_source_and_target_operator_data();

            // Compute metadata and allocate storage buffers for results
            result.set_metadata(self.eval_type.unwrap(), self.charges.unwrap());

            Ok(result)
        }
    }
}

impl<T, U, V, W> KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTreeNew<W>>,
    T::Tree: Tree<Domain = Domain<W>, Precision = W, NodeIndex = MortonKey>,
    U: SourceToTargetData<V>,
    V: Kernel<T = W>,
    W: FmmScalar,
    Array<W, BaseArray<W, VectorContainer<W>, 2>, 2>: MatrixSvd<Item = W>,
{
    fn set_source_and_target_operator_data(&mut self) {
        // Cast surface parameters
        let alpha_outer = W::from(ALPHA_OUTER).unwrap();
        let alpha_inner = W::from(ALPHA_INNER).unwrap();
        let domain = self.tree.get_domain();

        // Compute required surfaces
        let upward_equivalent_surface =
            ROOT.compute_surface(domain, self.expansion_order, alpha_inner);
        let upward_check_surface = ROOT.compute_surface(domain, self.expansion_order, alpha_outer);
        let downward_equivalent_surface =
            ROOT.compute_surface(domain, self.expansion_order, alpha_outer);
        let downward_check_surface =
            ROOT.compute_surface(domain, self.expansion_order, alpha_inner);

        let nequiv_surface = upward_equivalent_surface.len() / self.kernel.space_dimension();
        let ncheck_surface = upward_check_surface.len() / self.kernel.space_dimension();

        // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
        // As well as estimating their inverses using GESVD
        let mut uc2e_t = rlst_dynamic_array2!(W, [ncheck_surface, nequiv_surface]);
        self.kernel.assemble_st(
            EvalType::Value,
            &upward_equivalent_surface[..],
            &upward_check_surface[..],
            uc2e_t.data_mut(),
        );

        // Need to transpose so that rows correspond to targets and columns to sources
        let mut uc2e = rlst_dynamic_array2!(W, [nequiv_surface, ncheck_surface]);
        uc2e.fill_from(uc2e_t.transpose());

        let mut dc2e_t = rlst_dynamic_array2!(W, [ncheck_surface, nequiv_surface]);
        self.kernel.assemble_st(
            EvalType::Value,
            &downward_equivalent_surface[..],
            &downward_check_surface[..],
            dc2e_t.data_mut(),
        );

        // Need to transpose so that rows correspond to targets and columns to sources
        let mut dc2e = rlst_dynamic_array2!(W, [nequiv_surface, ncheck_surface]);
        dc2e.fill_from(dc2e_t.transpose());

        let (s, ut, v) = pinv::<W>(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(W, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = W::from_real(s[i]);
        }
        let uc2e_inv_1 = empty_array::<W, 2>().simple_mult_into_resize(v.view(), mat_s.view());
        let uc2e_inv_2 = ut;

        let (s, ut, v) = pinv::<W>(&dc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(W, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = W::from_real(s[i]);
        }

        let dc2e_inv_1 = empty_array::<W, 2>().simple_mult_into_resize(v.view(), mat_s.view());
        let dc2e_inv_2 = ut;

        // Calculate M2M and L2L operator matrices
        let children = ROOT.children();
        let mut m2m = rlst_dynamic_array2!(W, [nequiv_surface, 8 * nequiv_surface]);
        let mut m2m_vec = Vec::new();

        let mut l2l = Vec::new();

        for (i, child) in children.iter().enumerate() {
            let child_upward_equivalent_surface =
                child.compute_surface(domain, self.expansion_order, alpha_inner);
            let child_downward_check_surface =
                child.compute_surface(domain, self.expansion_order, alpha_inner);

            let mut pc2ce_t = rlst_dynamic_array2!(W, [ncheck_surface, nequiv_surface]);

            self.kernel.assemble_st(
                EvalType::Value,
                &child_upward_equivalent_surface,
                &upward_check_surface,
                pc2ce_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut pc2ce = rlst_dynamic_array2!(W, [nequiv_surface, ncheck_surface]);
            pc2ce.fill_from(pc2ce_t.transpose());

            let tmp = empty_array::<W, 2>().simple_mult_into_resize(
                uc2e_inv_1.view(),
                empty_array::<W, 2>().simple_mult_into_resize(uc2e_inv_2.view(), pc2ce.view()),
            );
            let l = i * nequiv_surface * nequiv_surface;
            let r = l + nequiv_surface * nequiv_surface;

            m2m.data_mut()[l..r].copy_from_slice(tmp.data());
            m2m_vec.push(tmp);

            let mut cc2pe_t = rlst_dynamic_array2!(W, [ncheck_surface, nequiv_surface]);

            self.kernel.assemble_st(
                EvalType::Value,
                &downward_equivalent_surface,
                &child_downward_check_surface,
                cc2pe_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut cc2pe = rlst_dynamic_array2!(W, [nequiv_surface, ncheck_surface]);
            cc2pe.fill_from(cc2pe_t.transpose());
            let mut tmp = empty_array::<W, 2>().simple_mult_into_resize(
                dc2e_inv_1.view(),
                empty_array::<W, 2>().simple_mult_into_resize(dc2e_inv_2.view(), cc2pe.view()),
            );
            tmp.data_mut()
                .iter_mut()
                .for_each(|d| *d *= homogenous_kernel_scale(child.level()));

            l2l.push(tmp);
        }

        self.source_data = m2m;
        self.source_data_vec = m2m_vec;
        self.target_data = l2l;
        self.dc2e_inv_1 = dc2e_inv_1;
        self.dc2e_inv_2 = dc2e_inv_2;
        self.uc2e_inv_1 = uc2e_inv_1;
        self.uc2e_inv_2 = uc2e_inv_2;
    }

    fn set_metadata(&mut self, eval_type: EvalType, charges: &Charges<W>) {
        let dim = self.kernel.space_dimension();
        let alpha_outer = W::from(ALPHA_OUTER).unwrap();
        let alpha_inner = W::from(ALPHA_INNER).unwrap();

        // Check if computing potentials, or potentials and derivatives
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => dim + 1,
        };

        // Check if we are computing matvec or matmul
        let [ncharges, nmatvecs] = charges.shape();

        let ntarget_points = self
            .tree
            .get_target_tree()
            .get_all_coordinates()
            .unwrap()
            .len()
            / dim;

        let nsource_points = self
            .tree
            .get_source_tree()
            .get_all_coordinates()
            .unwrap()
            .len()
            / dim;

        let nsource_keys = self.tree.get_source_tree().get_nall_keys().unwrap();
        let ntarget_keys = self.tree.get_target_tree().get_nall_keys().unwrap();
        let ntarget_leaves = self.tree.get_target_tree().get_nleaves().unwrap();
        let nsource_leaves = self.tree.get_source_tree().get_nleaves().unwrap();

        // Buffers to store all multipole and local data
        let multipoles = vec![W::default(); self.ncoeffs * nsource_keys * nmatvecs];
        let locals = vec![W::default(); self.ncoeffs * ntarget_keys * nmatvecs];

        // Mutable pointers to multipole and local data, indexed by level
        let mut level_multipoles = vec![
            Vec::new();
            (self.tree.get_source_tree().get_depth() + 1)
                .try_into()
                .unwrap()
        ];
        let mut level_locals = vec![
            Vec::new();
            (self.tree.get_target_tree().get_depth() + 1)
                .try_into()
                .unwrap()
        ];

        // Index pointers of multipole and local data, indexed by level
        let mut level_index_pointer_multipoles = vec![
            HashMap::new();
            (self.tree.get_source_tree().get_depth() + 1)
                .try_into()
                .unwrap()
        ];
        let mut level_index_pointer_locals = vec![
            HashMap::new();
            (self.tree.get_target_tree().get_depth() + 1)
                .try_into()
                .unwrap()
        ];

        // Mutable pointers to multipole and local data only at leaf level
        let mut leaf_multipoles = vec![Vec::new(); nsource_leaves];
        let mut leaf_locals = vec![Vec::new(); ntarget_leaves];

        // Buffer to store evaluated potentials and/or gradients at target points
        let mut potentials = vec![W::default(); ntarget_points * eval_size * nmatvecs];

        // Mutable pointers to potential data at each target leaf
        let mut potentials_send_pointers = vec![SendPtrMut::default(); ntarget_leaves * nmatvecs];

        // Index pointer of charge data at each target leaf
        let mut charge_index_pointer_sources = vec![(0usize, 0usize); ntarget_leaves];
        let mut charge_index_pointer_targets = vec![(0usize, 0usize); ntarget_leaves];

        // Kernel scale at each target and source leaf
        let mut target_leaf_scales = vec![W::default(); ntarget_leaves * self.ncoeffs * nmatvecs];
        let mut source_leaf_scales = vec![W::default(); nsource_leaves * self.ncoeffs * nmatvecs];

        // Pre compute check surfaces
        let mut upward_surfaces = vec![W::default(); self.ncoeffs * nsource_keys * dim];
        let mut downward_surfaces = vec![W::default(); self.ncoeffs * ntarget_keys * dim];
        let mut leaf_upward_surfaces = vec![W::default(); self.ncoeffs * nsource_leaves * dim];
        let mut leaf_downward_surfaces = vec![W::default(); self.ncoeffs * ntarget_leaves * dim];

        // Create mutable pointers to multipole and local data indexed by tree level
        {
            for level in 0..=self.tree.get_source_tree().get_depth() {
                let mut tmp_multipoles = Vec::new();

                let keys = self.tree.get_source_tree().get_keys(level).unwrap();
                for key in keys.into_iter() {
                    let &key_idx = self.tree.get_source_tree().get_index(key).unwrap();
                    let key_displacement = self.ncoeffs * nmatvecs * key_idx;
                    let mut key_multipoles = Vec::new();
                    for eval_idx in 0..nmatvecs {
                        let eval_displacement = self.ncoeffs * eval_idx;
                        let raw = unsafe {
                            multipoles
                                .as_ptr()
                                .add(key_displacement + eval_displacement)
                                as *mut W
                        };
                        key_multipoles.push(SendPtrMut { raw });
                    }
                    tmp_multipoles.push(key_multipoles)
                }
                level_multipoles[level as usize] = tmp_multipoles
            }

            for level in 0..=self.tree.get_target_tree().get_depth() {
                let mut tmp_locals = Vec::new();

                let keys = self.tree.get_target_tree().get_keys(level).unwrap();
                for key in keys.into_iter() {
                    let &key_idx = self.tree.get_target_tree().get_index(key).unwrap();
                    let key_displacement = self.ncoeffs * nmatvecs * key_idx;
                    let mut key_locals = Vec::new();
                    for eval_idx in 0..nmatvecs {
                        let eval_displacement = self.ncoeffs * eval_idx;
                        let raw = unsafe {
                            locals.as_ptr().add(key_displacement + eval_displacement) as *mut W
                        };
                        key_locals.push(SendPtrMut { raw });
                    }
                    tmp_locals.push(key_locals)
                }
                level_locals[level as usize] = tmp_locals
            }

            for level in 0..=self.tree.get_source_tree().get_depth() {
                let keys = self.tree.get_source_tree().get_keys(level).unwrap();
                for (level_idx, key) in keys.into_iter().enumerate() {
                    level_index_pointer_multipoles[level as usize].insert(*key, level_idx);
                }
            }

            for level in 0..=self.tree.get_target_tree().get_depth() {
                let keys = self.tree.get_target_tree().get_keys(level).unwrap();
                for (level_idx, key) in keys.into_iter().enumerate() {
                    level_index_pointer_locals[level as usize].insert(*key, level_idx);
                }
            }
        }

        // Create mutable pointers to multipole and local data at leaf level
        {
            for (leaf_idx, leaf) in self
                .tree
                .get_source_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let key_idx = self.tree.get_source_tree().get_index(leaf).unwrap();
                let key_displacement = self.ncoeffs * nmatvecs * key_idx;
                for eval_idx in 0..nmatvecs {
                    let eval_displacement = self.ncoeffs * eval_idx;
                    let raw = unsafe {
                        multipoles
                            .as_ptr()
                            .add(eval_displacement + key_displacement)
                            as *mut W
                    };

                    leaf_multipoles[leaf_idx].push(SendPtrMut { raw });
                }
            }

            for (leaf_idx, leaf) in self
                .tree
                .get_target_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let key_idx = self.tree.get_target_tree().get_index(leaf).unwrap();
                let key_displacement = self.ncoeffs * nmatvecs * key_idx;
                for eval_idx in 0..nmatvecs {
                    let eval_displacement = self.ncoeffs * eval_idx;
                    let raw = unsafe {
                        locals.as_ptr().add(eval_displacement + key_displacement) as *mut W
                    };
                    leaf_locals[leaf_idx].push(SendPtrMut { raw });
                }
            }
        }

        // Set index pointers for evaluated potentials
        {
            let mut index_pointer = 0;
            let mut potential_raw_pointers = Vec::new();
            for eval_idx in 0..nmatvecs {
                let ptr = unsafe {
                    potentials
                        .as_mut_ptr()
                        .add(eval_idx * ntarget_points * eval_size)
                };
                potential_raw_pointers.push(ptr)
            }

            for (i, leaf) in self
                .tree
                .get_target_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let l = i * self.ncoeffs;
                let r = l + self.ncoeffs;
                target_leaf_scales[l..r].copy_from_slice(
                    vec![homogenous_kernel_scale(leaf.level()); self.ncoeffs].as_slice(),
                );

                let npoints;
                let nevals;

                if let Some(coordinates) = self.tree.get_target_tree().get_coordinates(leaf) {
                    npoints = coordinates.len() / dim;
                    nevals = npoints * eval_size;
                } else {
                    npoints = 0;
                    nevals = 0;
                }

                // println!("nevals {:?}", nevals, potential_raw_pointers.len());
                for j in 0..nmatvecs {
                    potentials_send_pointers[ntarget_leaves * j + i] = SendPtrMut {
                        raw: potential_raw_pointers[j],
                    }
                }

                // Update charge index pointer
                let bounds_points = (index_pointer, index_pointer + npoints);
                charge_index_pointer_targets[i] = bounds_points;
                index_pointer += npoints;

                // Update raw pointer with number of points at this leaf
                for ptr in potential_raw_pointers.iter_mut() {
                    *ptr = unsafe { ptr.add(nevals) }
                }
            }

            let mut index_pointer = 0;

            for (i, leaf) in self
                .tree
                .get_source_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                // Assign scales
                let l = i * self.ncoeffs;
                let r = l + self.ncoeffs;
                source_leaf_scales[l..r].copy_from_slice(
                    vec![homogenous_kernel_scale(leaf.level()); self.ncoeffs].as_slice(),
                );

                let npoints;
                if let Some(coordinates) = self.tree.get_source_tree().get_coordinates(leaf) {
                    npoints = coordinates.len() / dim;
                } else {
                    npoints = 0;
                }

                let bounds_points = (index_pointer, index_pointer + npoints);
                charge_index_pointer_sources[i] = bounds_points;
                index_pointer += npoints;
            }
        }

        // Compute surfaces
        {
            // All upward and downward surfaces
            for (i, key) in self
                .tree
                .get_source_tree()
                .get_all_keys()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let l = i * self.ncoeffs * dim;
                let r = l + self.ncoeffs * dim;
                let upward_surface =
                    key.compute_surface(self.tree.get_domain(), self.expansion_order, alpha_outer);

                upward_surfaces[l..r].copy_from_slice(&upward_surface);
            }

            for (i, key) in self
                .tree
                .get_target_tree()
                .get_all_keys()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let l = i * self.ncoeffs * dim;
                let r = l + self.ncoeffs * dim;
                let downward_surface =
                    key.compute_surface(self.tree.get_domain(), self.expansion_order, alpha_outer);

                downward_surfaces[l..r].copy_from_slice(&downward_surface);
            }

            // Leaf upward and downward surfaces
            for (i, key) in self
                .tree
                .get_source_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let l = i * self.ncoeffs * dim;
                let r = l + self.ncoeffs * dim;
                let upward_surface =
                    key.compute_surface(self.tree.get_domain(), self.expansion_order, alpha_outer);

                leaf_upward_surfaces[l..r].copy_from_slice(&upward_surface);
            }

            for (i, key) in self
                .tree
                .get_target_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let l = i * self.ncoeffs * dim;
                let r = l + self.ncoeffs * dim;
                let downward_surface =
                    key.compute_surface(self.tree.get_domain(), self.expansion_order, alpha_inner);

                leaf_downward_surfaces[l..r].copy_from_slice(&downward_surface);
            }
        }

        // Set data
        {
            self.multipoles = multipoles;
            self.locals = locals;
            self.leaf_multipoles = leaf_multipoles;
            self.level_multipoles = level_multipoles;
            self.leaf_locals = leaf_locals;
            self.level_locals = level_locals;
            self.level_index_pointer_locals = level_index_pointer_locals;
            self.level_index_pointer_multipoles = level_index_pointer_multipoles;
            self.potentials = potentials;
            self.potentials_send_pointers = potentials_send_pointers;
            self.upward_surfaces = upward_surfaces;
            self.downward_surfaces = downward_surfaces;
            self.leaf_upward_surfaces = leaf_upward_surfaces;
            self.leaf_downward_surfaces = leaf_downward_surfaces;
            self.charges = charges.data().to_vec();
            self.charge_index_pointer_targets = charge_index_pointer_targets;
            self.charge_index_pointer_sources = charge_index_pointer_sources;
            self.target_scales = target_leaf_scales;
            self.source_scales = source_leaf_scales;
            self.eval_size = eval_size;
        }
    }
}

#[cfg(test)]
mod test {

    use bempp_field::types::FftFieldTranslationKiFmm;
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::implementations::helpers::points_fixture;
    use rlst_dense::{rlst_array_from_slice2, traits::RawAccess};

    use super::*;

    #[test]
    fn test_builder() {
        let npoints = 1000;
        let nvecs = 1;
        let sources = points_fixture::<f64>(npoints, None, None, Some(0));
        let targets = points_fixture::<f64>(npoints, None, None, Some(1));
        let tmp = vec![1.0; npoints * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [npoints, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let n_crit = Some(100);
        let expansion_order = 5;
        let sparse = true;

        let fmm = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, &charges, None, sparse)
            .parameters(
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::ValueDeriv,
                FftFieldTranslationKiFmm::default(),
            )
            .unwrap()
            .build()
            .unwrap();

        // fmm.evaluate_vec(&charges, &mut result);
    }
}
