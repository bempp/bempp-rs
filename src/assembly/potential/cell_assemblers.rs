//! Assemblers that assemble the contributions to the global matrix due to a single cell

use crate::assembly::common::{AssemblerGeometry, RlstArray};
use crate::traits::{CellAssembler, KernelEvaluator, PotentialIntegrand};
use ndgrid::traits::GeometryMap;
use num::Zero;
use rlst::{
    rlst_dynamic_array2, rlst_dynamic_array3, DefaultIteratorMut, RawAccess, RawAccessMut,
    RlstScalar,
};

pub struct PotentialCellAssembler<
    'a,
    T: RlstScalar,
    I: PotentialIntegrand<T = T>,
    G: GeometryMap<T = T::Real>,
    K: KernelEvaluator<T = T>,
> {
    integrand: &'a I,
    kernel: &'a K,
    evaluator: G,
    table: &'a RlstArray<T, 4>,
    k: RlstArray<T, 3>,
    mapped_pts: RlstArray<T::Real, 2>,
    normals: RlstArray<T::Real, 2>,
    jacobians: RlstArray<T::Real, 2>,
    jdet: Vec<T::Real>,
    evaluation_points: &'a RlstArray<T::Real, 2>,
    weights: &'a [T::Real],
    cell: usize,
}

impl<
        'a,
        T: RlstScalar,
        I: PotentialIntegrand<T = T>,
        G: GeometryMap<T = T::Real>,
        K: KernelEvaluator<T = T>,
    > PotentialCellAssembler<'a, T, I, G, K>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        npts: usize,
        nevalpts: usize,
        deriv_size: usize,
        integrand: &'a I,
        kernel: &'a K,
        evaluator: G,
        table: &'a RlstArray<T, 4>,
        evaluation_points: &'a RlstArray<T::Real, 2>,
        weights: &'a [T::Real],
    ) -> Self {
        Self {
            integrand,
            kernel,
            evaluator,
            table,
            k: rlst_dynamic_array3!(T, [deriv_size, npts, nevalpts]),
            mapped_pts: rlst_dynamic_array2!(T::Real, [3, npts]),
            normals: rlst_dynamic_array2!(T::Real, [3, npts]),
            jacobians: rlst_dynamic_array2!(T::Real, [6, npts]),
            jdet: vec![T::Real::zero(); npts],
            evaluation_points,
            weights,
            cell: 0,
        }
    }
}
impl<
        'a,
        T: RlstScalar,
        I: PotentialIntegrand<T = T>,
        G: GeometryMap<T = T::Real>,
        K: KernelEvaluator<T = T>,
    > CellAssembler for PotentialCellAssembler<'a, T, I, G, K>
{
    type T = T;
    fn set_cell(&mut self, cell: usize) {
        self.cell = cell;
        self.evaluator.points(cell, self.mapped_pts.data_mut());
        self.evaluator.jacobians_dets_normals(
            cell,
            self.jacobians.data_mut(),
            &mut self.jdet,
            self.normals.data_mut(),
        );
    }

    fn assemble(&mut self, local_mat: &mut RlstArray<T, 2>) {
        self.kernel.assemble_st(
            self.mapped_pts.data(),
            self.evaluation_points.data(),
            self.k.data_mut(),
        );

        let geometry =
            AssemblerGeometry::new(&self.mapped_pts, &self.normals, &self.jacobians, &self.jdet);

        for (i, mut col) in local_mat.col_iter_mut().enumerate() {
            for (eval_index, entry) in col.iter_mut().enumerate() {
                *entry = T::zero();
                for (index, wt) in self.weights.iter().enumerate() {
                    *entry += self
                        .integrand
                        .evaluate(self.table, index, eval_index, i, &self.k, &geometry)
                        * num::cast::<T::Real, T>(*wt * self.jdet[index]).unwrap();
                }
            }
        }
    }
}
