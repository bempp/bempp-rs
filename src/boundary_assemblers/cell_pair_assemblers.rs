//! Assemblers that assemble the contributions to the global matrix due to a single pair of cells

use crate::boundary_assemblers::helpers::{AssemblerGeometry, KernelEvaluator, RlstArray};
use green_kernels::traits::Kernel;
use itertools::izip;
use ndgrid::traits::GeometryMap;
use num::Zero;
use rlst::{
    rlst_dynamic_array2, rlst_dynamic_array3, DefaultIteratorMut, RawAccess, RawAccessMut,
    RlstScalar,
};
use std::collections::HashMap;

use super::integrands::BoundaryIntegrand;

/// Assembler for the contributions from pairs of neighbouring cells
pub struct SingularCellPairAssembler<
    'a,
    T: RlstScalar,
    I: BoundaryIntegrand<T = T>,
    G: GeometryMap<T = T::Real>,
    K: Kernel<T = T>,
> {
    integrand: &'a I,
    kernel: &'a KernelEvaluator<T, K>,
    test_evaluator: G,
    trial_evaluator: G,
    test_table: &'a RlstArray<T, 4>,
    trial_table: &'a RlstArray<T, 4>,
    k: RlstArray<T, 2>,
    test_mapped_pts: RlstArray<T::Real, 2>,
    trial_mapped_pts: RlstArray<T::Real, 2>,
    test_normals: RlstArray<T::Real, 2>,
    trial_normals: RlstArray<T::Real, 2>,
    test_jacobians: RlstArray<T::Real, 2>,
    trial_jacobians: RlstArray<T::Real, 2>,
    test_jdet: Vec<T::Real>,
    trial_jdet: Vec<T::Real>,
    weights: &'a [T::Real],
    test_cell: usize,
    trial_cell: usize,
}

impl<
        'a,
        T: RlstScalar,
        I: BoundaryIntegrand<T = T>,
        G: GeometryMap<T = T::Real>,
        K: Kernel<T = T>,
    > SingularCellPairAssembler<'a, T, I, G, K>
{
    #[allow(clippy::too_many_arguments)]
    /// Create new
    pub fn new(
        npts: usize,
        deriv_size: usize,
        integrand: &'a I,
        kernel: &'a KernelEvaluator<T, K>,
        test_evaluator: G,
        trial_evaluator: G,
        test_table: &'a RlstArray<T, 4>,
        trial_table: &'a RlstArray<T, 4>,
        weights: &'a [T::Real],
    ) -> Self {
        Self {
            integrand,
            kernel,
            test_evaluator,
            trial_evaluator,
            test_table,
            trial_table,
            k: rlst_dynamic_array2!(T, [deriv_size, npts]),
            test_mapped_pts: rlst_dynamic_array2!(T::Real, [3, npts]),
            trial_mapped_pts: rlst_dynamic_array2!(T::Real, [3, npts]),
            test_normals: rlst_dynamic_array2!(T::Real, [3, npts]),
            trial_normals: rlst_dynamic_array2!(T::Real, [3, npts]),
            test_jacobians: rlst_dynamic_array2!(T::Real, [6, npts]),
            trial_jacobians: rlst_dynamic_array2!(T::Real, [6, npts]),
            test_jdet: vec![T::Real::zero(); npts],
            trial_jdet: vec![T::Real::zero(); npts],
            weights,
            test_cell: 0,
            trial_cell: 0,
        }
    }

    pub fn set_test_cell(&mut self, test_cell: usize) {
        self.test_cell = test_cell;
        self.test_evaluator
            .points(test_cell, self.test_mapped_pts.data_mut());
        self.test_evaluator.jacobians_dets_normals(
            test_cell,
            self.test_jacobians.data_mut(),
            &mut self.test_jdet,
            self.test_normals.data_mut(),
        );
    }
    pub fn set_trial_cell(&mut self, trial_cell: usize) {
        self.trial_cell = trial_cell;
        self.trial_evaluator
            .points(trial_cell, self.trial_mapped_pts.data_mut());
        self.trial_evaluator.jacobians_dets_normals(
            trial_cell,
            self.trial_jacobians.data_mut(),
            &mut self.trial_jdet,
            self.trial_normals.data_mut(),
        );
    }
    pub fn assemble(&mut self, local_mat: &mut RlstArray<T, 2>) {
        self.kernel.assemble_pairwise_st(
            self.test_mapped_pts.data(),
            self.trial_mapped_pts.data(),
            self.k.data_mut(),
        );

        let test_geometry = AssemblerGeometry::new(
            &self.test_mapped_pts,
            &self.test_normals,
            &self.test_jacobians,
            &self.test_jdet,
        );
        let trial_geometry = AssemblerGeometry::new(
            &self.trial_mapped_pts,
            &self.trial_normals,
            &self.trial_jacobians,
            &self.trial_jdet,
        );

        for (trial_i, mut col) in local_mat.col_iter_mut().enumerate() {
            for (test_i, entry) in col.iter_mut().enumerate() {
                *entry = T::zero();
                for (index, wt) in self.weights.iter().enumerate() {
                    *entry += self.integrand.evaluate_singular(
                        self.test_table,
                        self.trial_table,
                        index,
                        test_i,
                        trial_i,
                        &self.k,
                        &test_geometry,
                        &trial_geometry,
                    ) * num::cast::<T::Real, T>(
                        *wt * unsafe {
                            *self.test_jdet.get_unchecked(index)
                                * *self.trial_jdet.get_unchecked(index)
                        },
                    )
                    .unwrap();
                }
            }
        }
    }
}

/// Assembler for the contributions from pairs of non-neighbouring cells with test geometry caching
pub struct NonsingularCellPairAssemblerWithTestCaching<
    'a,
    T: RlstScalar,
    I: BoundaryIntegrand<T = T>,
    TrialG: GeometryMap<T = T::Real>,
    K: Kernel<T = T>,
> {
    integrand: &'a I,
    kernel: &'a KernelEvaluator<T, K>,
    trial_evaluator: TrialG,
    test_table: &'a RlstArray<T, 4>,
    trial_table: &'a RlstArray<T, 4>,
    k: RlstArray<T, 3>,
    test_mapped_pts: Vec<RlstArray<T::Real, 2>>,
    trial_mapped_pts: RlstArray<T::Real, 2>,
    test_normals: Vec<RlstArray<T::Real, 2>>,
    trial_normals: RlstArray<T::Real, 2>,
    test_jacobians: Vec<RlstArray<T::Real, 2>>,
    trial_jacobians: RlstArray<T::Real, 2>,
    test_jdet: Vec<Vec<T::Real>>,
    trial_jdet: Vec<T::Real>,
    test_weights: &'a [T::Real],
    trial_weights: &'a [T::Real],
    test_cell: usize,
    trial_cell: usize,
    test_indices: HashMap<usize, usize>,
}

impl<
        'a,
        T: RlstScalar,
        I: BoundaryIntegrand<T = T>,
        TrialG: GeometryMap<T = T::Real>,
        K: Kernel<T = T>,
    > NonsingularCellPairAssemblerWithTestCaching<'a, T, I, TrialG, K>
{
    #[allow(clippy::too_many_arguments)]
    /// Create new
    pub fn new<TestG: GeometryMap<T = T::Real>>(
        npts_test: usize,
        npts_trial: usize,
        deriv_size: usize,
        test_cells: &[usize],
        integrand: &'a I,
        kernel: &'a KernelEvaluator<T, K>,
        test_evaluator: TestG,
        trial_evaluator: TrialG,
        test_table: &'a RlstArray<T, 4>,
        trial_table: &'a RlstArray<T, 4>,
        test_weights: &'a [T::Real],
        trial_weights: &'a [T::Real],
    ) -> Self {
        let mut test_mapped_pts = test_cells
            .iter()
            .map(|_| rlst_dynamic_array2!(T::Real, [3, npts_test]))
            .collect::<Vec<_>>();
        let mut test_normals = test_cells
            .iter()
            .map(|_| rlst_dynamic_array2!(T::Real, [3, npts_test]))
            .collect::<Vec<_>>();
        let mut test_jacobians = test_cells
            .iter()
            .map(|_| rlst_dynamic_array2!(T::Real, [6, npts_test]))
            .collect::<Vec<_>>();
        let mut test_jdet = vec![vec![T::Real::zero(); npts_test]; test_cells.len()];

        let mut test_indices = HashMap::new();

        for (i, (cell, pts, n, j, jdet)) in izip!(
            test_cells,
            test_mapped_pts.iter_mut(),
            test_normals.iter_mut(),
            test_jacobians.iter_mut(),
            test_jdet.iter_mut()
        )
        .enumerate()
        {
            test_indices.insert(*cell, i);
            test_evaluator.points(*cell, pts.data_mut());
            test_evaluator.jacobians_dets_normals(*cell, j.data_mut(), jdet, n.data_mut())
        }

        Self {
            integrand,
            kernel,
            trial_evaluator,
            test_table,
            trial_table,
            k: rlst_dynamic_array3!(T, [deriv_size, npts_test, npts_trial]),
            test_mapped_pts,
            trial_mapped_pts: rlst_dynamic_array2!(T::Real, [3, npts_trial]),
            test_normals,
            trial_normals: rlst_dynamic_array2!(T::Real, [3, npts_trial]),
            test_jacobians,
            trial_jacobians: rlst_dynamic_array2!(T::Real, [6, npts_trial]),
            test_jdet,
            trial_jdet: vec![T::Real::zero(); npts_trial],
            test_weights,
            trial_weights,
            test_cell: 0,
            trial_cell: 0,
            test_indices,
        }
    }
}

impl<
        'a,
        T: RlstScalar,
        I: BoundaryIntegrand<T = T>,
        TrialG: GeometryMap<T = T::Real>,
        K: Kernel<T = T>,
    > NonsingularCellPairAssemblerWithTestCaching<'a, T, I, TrialG, K>
{
    pub fn set_test_cell(&mut self, test_cell: usize) {
        self.test_cell = self.test_indices[&test_cell];
    }
    pub fn set_trial_cell(&mut self, trial_cell: usize) {
        self.trial_cell = trial_cell;
        self.trial_evaluator
            .points(trial_cell, self.trial_mapped_pts.data_mut());
        self.trial_evaluator.jacobians_dets_normals(
            trial_cell,
            self.trial_jacobians.data_mut(),
            &mut self.trial_jdet,
            self.trial_normals.data_mut(),
        );
    }
    pub fn assemble(&mut self, local_mat: &mut RlstArray<T, 2>) {
        self.kernel.assemble_st(
            unsafe { self.test_mapped_pts.get_unchecked(self.test_cell).data() },
            self.trial_mapped_pts.data(),
            self.k.data_mut(),
        );

        let test_geometry = unsafe {
            AssemblerGeometry::new(
                self.test_mapped_pts.get_unchecked(self.test_cell),
                self.test_normals.get_unchecked(self.test_cell),
                self.test_jacobians.get_unchecked(self.test_cell),
                self.test_jdet.get_unchecked(self.test_cell),
            )
        };
        let trial_geometry = AssemblerGeometry::new(
            &self.trial_mapped_pts,
            &self.trial_normals,
            &self.trial_jacobians,
            &self.trial_jdet,
        );

        for (trial_i, mut col) in local_mat.col_iter_mut().enumerate() {
            for (test_i, entry) in col.iter_mut().enumerate() {
                *entry = T::zero();
                for (test_index, test_wt) in self.test_weights.iter().enumerate() {
                    let test_integrand = unsafe {
                        num::cast::<T::Real, T>(
                            *test_wt
                                * *self
                                    .test_jdet
                                    .get_unchecked(self.test_cell)
                                    .get_unchecked(test_index),
                        )
                    }
                    .unwrap();
                    for (trial_index, trial_wt) in self.trial_weights.iter().enumerate() {
                        *entry += self.integrand.evaluate_nonsingular(
                            self.test_table,
                            self.trial_table,
                            test_index,
                            trial_index,
                            test_i,
                            trial_i,
                            &self.k,
                            &test_geometry,
                            &trial_geometry,
                        ) * num::cast::<T::Real, T>(
                            *trial_wt * unsafe { *self.trial_jdet.get_unchecked(trial_index) },
                        )
                        .unwrap()
                            * test_integrand;
                    }
                }
            }
        }
    }
}
