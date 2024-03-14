//! Implementation of FmmData and Fmm traits.
use num::Float;
use rlst_dense::{
    rlst_dynamic_array2,
    traits::{RawAccess, Shape},
    types::RlstScalar,
};

use bempp_traits::{
    field::SourceToTargetData,
    fmm::{Fmm, SourceToTargetTranslation, SourceTranslation, TargetTranslation},
    kernel::Kernel,
    tree::{FmmTree, Tree},
    types::EvalType,
};

use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use crate::{
    helpers::{leaf_expansion_pointers, level_expansion_pointers, map_charges, potential_pointers},
    types::{Charges, FmmEvalType, KiFmm, KiFmmDummy},
};

impl<T, U, V, W> Fmm for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTree<W>, Node = MortonKey> + Send + Sync,
    U: SourceToTargetData<V> + Send + Sync,
    V: Kernel<T = W> + Send + Sync,
    W: RlstScalar<Real = W> + Float + Default,
    Self: SourceToTargetTranslation,
{
    type NodeIndex = T::Node;
    type Precision = W;
    type Kernel = V;
    type Tree = T;

    fn dim(&self) -> usize {
        self.dim
    }

    fn multipole(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        if let Some(index) = self.tree.source_tree().index(key) {
            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    Some(&self.multipoles[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvalType::Matrix(nmatvecs) => Some(
                    &self.multipoles
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn local(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        if let Some(index) = self.tree.target_tree().index(key) {
            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    Some(&self.locals[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvalType::Matrix(nmatvecs) => Some(
                    &self.locals
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn potential(&self, leaf: &Self::NodeIndex) -> Option<Vec<&[Self::Precision]>> {
        if let Some(&leaf_idx) = self.tree.target_tree().leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer_targets[leaf_idx];
            let ntargets = r - l;

            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(vec![
                    &self.potentials[l * self.kernel_eval_size..r * self.kernel_eval_size],
                ]),
                FmmEvalType::Matrix(nmatvecs) => {
                    let nleaves = self.tree.target_tree().nleaves().unwrap();
                    let mut slices = Vec::new();
                    for eval_idx in 0..nmatvecs {
                        let potentials_pointer =
                            self.potentials_send_pointers[eval_idx * nleaves + leaf_idx].raw;
                        slices.push(unsafe {
                            std::slice::from_raw_parts(
                                potentials_pointer,
                                ntargets * self.kernel_eval_size,
                            )
                        });
                    }
                    Some(slices)
                }
            }
        } else {
            None
        }
    }

    fn expansion_order(&self) -> usize {
        self.expansion_order
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn tree(&self) -> &Self::Tree {
        &self.tree
    }

    fn evaluate(&self) {
        // Upward pass
        {
            self.p2m();
            for level in (1..=self.tree.source_tree().depth()).rev() {
                self.m2m(level);
            }
        }

        // Downward pass
        {
            for level in 2..=self.tree.target_tree().depth() {
                if level > 2 {
                    self.l2l(level);
                }
                self.m2l(level);
                self.p2l(level)
            }

            // Leaf level computations
            self.m2p();
            self.p2p();
            self.l2p();
        }
    }

    fn clear(&mut self, charges: &Charges<W>) {
        let [_ncharges, nmatvecs] = charges.shape();
        let ntarget_points = self.tree().target_tree().ncoordinates_tot().unwrap();
        let nsource_leaves = self.tree().source_tree().nleaves().unwrap();
        let ntarget_leaves = self.tree().target_tree().nleaves().unwrap();

        // Clear buffers and set new buffers
        self.multipoles = vec![W::default(); self.multipoles.len()];
        self.locals = vec![W::default(); self.locals.len()];
        self.potentials = vec![W::default(); self.potentials.len()];
        self.charges = vec![W::default(); self.charges.len()];

        // Recreate mutable pointers for new buffers
        let potentials_send_pointers = potential_pointers(
            self.tree.target_tree(),
            nmatvecs,
            ntarget_leaves,
            ntarget_points,
            self.kernel_eval_size,
            &self.potentials,
        );

        let leaf_multipoles = leaf_expansion_pointers(
            self.tree().source_tree(),
            self.ncoeffs,
            nmatvecs,
            nsource_leaves,
            &self.multipoles,
        );

        let level_multipoles = level_expansion_pointers(
            self.tree().source_tree(),
            self.ncoeffs,
            nmatvecs,
            &self.multipoles,
        );

        let level_locals = level_expansion_pointers(
            self.tree().target_tree(),
            self.ncoeffs,
            nmatvecs,
            &self.locals,
        );

        let leaf_locals = leaf_expansion_pointers(
            self.tree().target_tree(),
            self.ncoeffs,
            nmatvecs,
            ntarget_leaves,
            &self.locals,
        );

        // Set mutable pointers
        self.level_locals = level_locals;
        self.level_multipoles = level_multipoles;
        self.leaf_locals = leaf_locals;
        self.leaf_multipoles = leaf_multipoles;
        self.potentials_send_pointers = potentials_send_pointers;

        // Set new charges
        self.charges = map_charges(
            self.tree.source_tree().all_global_indices().unwrap(),
            charges,
        )
        .data()
        .to_vec();
    }
}

impl<T, U, V, W> Default for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTree<W>> + Default,
    U: SourceToTargetData<V> + Default,
    V: Kernel + Default,
    W: RlstScalar<Real = W> + Float + Default,
{
    fn default() -> Self {
        let uc2e_inv_1 = rlst_dynamic_array2!(W, [1, 1]);
        let uc2e_inv_2 = rlst_dynamic_array2!(W, [1, 1]);
        let dc2e_inv_1 = rlst_dynamic_array2!(W, [1, 1]);
        let dc2e_inv_2 = rlst_dynamic_array2!(W, [1, 1]);
        let source = rlst_dynamic_array2!(W, [1, 1]);
        KiFmm {
            tree: T::default(),
            source_to_target_translation_data: U::default(),
            kernel: V::default(),
            expansion_order: 0,
            fmm_eval_type: FmmEvalType::Vector,
            kernel_eval_type: EvalType::Value,
            kernel_eval_size: 0,
            dim: 0,
            ncoeffs: 0,
            uc2e_inv_1,
            uc2e_inv_2,
            dc2e_inv_1,
            dc2e_inv_2,
            source_data: source,
            source_translation_data_vec: Vec::default(),
            target_data: Vec::default(),
            multipoles: Vec::default(),
            locals: Vec::default(),
            leaf_multipoles: Vec::default(),
            level_multipoles: Vec::default(),
            leaf_locals: Vec::default(),
            level_locals: Vec::default(),
            level_index_pointer_locals: Vec::default(),
            level_index_pointer_multipoles: Vec::default(),
            potentials: Vec::default(),
            potentials_send_pointers: Vec::default(),
            leaf_upward_surfaces_sources: Vec::default(),
            leaf_upward_surfaces_targets: Vec::default(),
            leaf_downward_surfaces: Vec::default(),
            charges: Vec::default(),
            charge_index_pointer_sources: Vec::default(),
            charge_index_pointer_targets: Vec::default(),
            leaf_scales_sources: Vec::default(),
            global_indices: Vec::default(),
        }
    }
}

impl<T, U, V> Fmm for KiFmmDummy<T, U, V>
where
    T: FmmTree<Tree = SingleNodeTree<U>, Node = MortonKey>,
    U: RlstScalar<Real = U> + Float + Default,
    V: Kernel<T = U> + Send + Sync,
{
    type NodeIndex = T::Node;
    type Precision = U;
    type Tree = T;
    type Kernel = V;

    fn dim(&self) -> usize {
        3
    }

    fn multipole(&self, _key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        None
    }

    fn local(&self, _key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        None
    }

    fn potential(&self, leaf: &Self::NodeIndex) -> Option<Vec<&[Self::Precision]>> {
        let ntarget_coordinates =
            self.tree.target_tree().all_coordinates().unwrap().len() / self.dim();

        if let Some(&leaf_idx) = self.tree.target_tree().leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer_targets[leaf_idx];

            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(vec![
                    &self.potentials[l * self.kernel_eval_size..r * self.kernel_eval_size],
                ]),
                FmmEvalType::Matrix(nmatvecs) => {
                    let mut slices = Vec::new();
                    for eval_idx in 0..nmatvecs {
                        let vec_displacement = eval_idx * ntarget_coordinates;
                        let slice = &self.potentials[vec_displacement + l..vec_displacement + r];
                        slices.push(slice);
                    }
                    Some(slices)
                }
            }
        } else {
            None
        }
    }

    fn evaluate(&self) {
        let all_target_coordinates = self.tree.target_tree().all_coordinates().unwrap();
        let ntarget_coordinates = all_target_coordinates.len() / self.dim();
        let all_source_coordinates = self.tree.source_tree().all_coordinates().unwrap();
        let nsource_coordinates = all_target_coordinates.len() / self.dim();

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                let charges = &self.charges;
                let res = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.potentials.as_ptr() as *mut U,
                        ntarget_coordinates,
                    )
                };
                self.kernel.evaluate_st(
                    self.kernel_eval_type,
                    all_source_coordinates,
                    all_target_coordinates,
                    charges,
                    res,
                )
            }

            FmmEvalType::Matrix(nmatvecs) => {
                for i in 0..nmatvecs {
                    let charges_i =
                        &self.charges[i * nsource_coordinates..(i + 1) * nsource_coordinates];
                    let res_i = unsafe {
                        std::slice::from_raw_parts_mut(
                            self.potentials.as_ptr().add(ntarget_coordinates) as *mut U,
                            ntarget_coordinates,
                        )
                    };
                    self.kernel.evaluate_st(
                        self.kernel_eval_type,
                        all_source_coordinates,
                        all_target_coordinates,
                        charges_i,
                        res_i,
                    )
                }
            }
        }
    }

    fn expansion_order(&self) -> usize {
        self.expansion_order
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn tree(&self) -> &Self::Tree {
        &self.tree
    }

    fn clear(&mut self, _charges: &Charges<U>) {}
}

#[cfg(test)]
mod test {

    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::constants::{ALPHA_INNER, ROOT};
    use bempp_tree::implementations::helpers::points_fixture;
    use num::Float;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rlst_dense::array::Array;
    use rlst_dense::base_array::BaseArray;
    use rlst_dense::data_container::VectorContainer;
    use rlst_dense::rlst_array_from_slice2;
    use rlst_dense::traits::{RawAccess, RawAccessMut, Shape};

    use crate::{tree::SingleNodeFmmTree, types::KiFmmBuilderSingleNode};
    use bempp_field::types::{BlasFieldTranslationKiFmm, FftFieldTranslationKiFmm};

    use super::*;

    fn test_single_node_fmm_vector_helper<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Precision = T,
                NodeIndex = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        eval_type: EvalType,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf: MortonKey = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::zero(); ntargets * eval_size];

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(T, leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);
        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        fmm.kernel().evaluate_st(
            eval_type,
            sources.data(),
            leaf_coordinates_col_major.data(),
            charges.data(),
            &mut direct,
        );

        direct.iter().zip(potential).for_each(|(&d, &p)| {
            let abs_error = num::Float::abs(d - p);
            let rel_error = abs_error / p;
            println!("err {:?} \nd {:?} \np {:?}", rel_error, direct, potential);
            assert!(rel_error <= threshold)
        });
    }

    fn test_single_node_fmm_matrix_helper<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Precision = T,
                NodeIndex = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        eval_type: EvalType,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf: MortonKey = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(T, leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);

        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        let [nsources, nmatvecs] = charges.shape();

        for i in 0..nmatvecs {
            let potential_i = fmm.potential(&leaf).unwrap()[i];
            let charges_i = &charges.data()[nsources * i..nsources * (i + 1)];
            let mut direct_i = vec![T::zero(); ntargets * eval_size];
            fmm.kernel().evaluate_st(
                eval_type,
                sources.data(),
                leaf_coordinates_col_major.data(),
                charges_i,
                &mut direct_i,
            );

            println!(
                "i {:?} \n direct_i {:?}\n potential_i {:?}",
                i, direct_i, potential_i
            );
            direct_i.iter().zip(potential_i).for_each(|(&d, &p)| {
                let abs_error = num::Float::abs(d - p);
                let rel_error = abs_error / p;
                assert!(rel_error <= threshold)
            })
        }
    }

    #[test]
    fn test_fmm_api() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;
        let threshold_pot = 1e-5;

        // Set charge data and evaluate an FMM
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let mut fmm = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                FftFieldTranslationKiFmm::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm.evaluate();

        // Reset Charge data and re-evaluate potential
        let mut rng = StdRng::seed_from_u64(1);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        fmm.clear(&charges);
        fmm.evaluate();

        let fmm = Box::new(fmm);
        test_single_node_fmm_vector_helper(
            fmm,
            bempp_traits::types::EvalType::Value,
            &sources,
            &charges,
            threshold_pot,
        );
    }

    #[test]
    fn test_laplace_fmm_vector() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;
        let threshold_pot = 1e-5;
        let threshold_deriv = 1e-4;
        let threshold_deriv_blas = 1e-3;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // fmm with fft based field translation
        {
            // Evaluate potentials
            let fmm_fft = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    bempp_traits::types::EvalType::Value,
                    FftFieldTranslationKiFmm::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_fft.evaluate();
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_fmm_vector_helper(
                fmm_fft,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let fmm_fft = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    bempp_traits::types::EvalType::ValueDeriv,
                    FftFieldTranslationKiFmm::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_fft.evaluate();
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_fmm_vector_helper(
                fmm_fft,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }

        // fmm with BLAS based field translation
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let fmm_blas = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationKiFmm::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_vector_helper(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let fmm_blas = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationKiFmm::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_vector_helper(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv_blas,
            );
        }
    }

    #[test]
    fn test_laplace_fmm_matrix() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));
        // FMM parameters
        let n_crit = Some(10);
        let expansion_order = 6;
        let sparse = true;
        let threshold = 1e-5;
        let threshold_deriv = 1e-3;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 5;
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        let mut rng = StdRng::seed_from_u64(0);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .for_each(|chunk| chunk.iter_mut().for_each(|elem| *elem += rng.gen::<f64>()));

        // fmm with blas based field translation
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let fmm_blas = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationKiFmm::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();

            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_matrix_helper(fmm_blas, eval_type, &sources, &charges, threshold);

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let fmm_blas = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationKiFmm::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_matrix_helper(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }
    }

    fn test_root_multipole_laplace_single_node<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Precision = T,
                NodeIndex = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {
        let multipole = fmm.multipole(&ROOT).unwrap();
        let upward_equivalent_surface = ROOT.compute_kifmm_surface(
            fmm.tree().domain(),
            fmm.expansion_order(),
            T::from(ALPHA_INNER).unwrap(),
        );

        let test_point = vec![T::from(100000.).unwrap(), T::zero(), T::zero()];
        let mut expected = vec![T::zero()];
        let mut found = vec![T::zero()];

        fmm.kernel().evaluate_st(
            bempp_traits::types::EvalType::Value,
            sources.data(),
            &test_point,
            charges.data(),
            &mut expected,
        );

        fmm.kernel().evaluate_st(
            bempp_traits::types::EvalType::Value,
            &upward_equivalent_surface,
            &test_point,
            multipole,
            &mut found,
        );

        let abs_error = num::Float::abs(expected[0] - found[0]);
        let rel_error = abs_error / expected[0];

        assert!(rel_error <= threshold);
    }

    #[test]
    fn test_upward_pass_vector() {
        // Setup random sources and targets
        let nsources = 10000;
        let ntargets = 10000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(1));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm_fft = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                FftFieldTranslationKiFmm::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_fft.evaluate();

        let svd_threshold = Some(1e-5);
        let fmm_svd = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                BlasFieldTranslationKiFmm::new(svd_threshold),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_svd.evaluate();

        let fmm_fft = Box::new(fmm_fft);
        let fmm_svd = Box::new(fmm_svd);
        test_root_multipole_laplace_single_node(fmm_fft, &sources, &charges, 1e-5);
        test_root_multipole_laplace_single_node(fmm_svd, &sources, &charges, 1e-5);
    }
}
