//! Batched dense assembly
use crate::assembly::common::{RawData2D, SparseMatrixData};
use crate::function_space::SerialFunctionSpace;
use bempp_grid::common::{compute_det23, compute_normal_from_jacobian23};
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_kernel::helmholtz_3d::Helmholtz3dKernel;
use bempp_quadrature::duffy::quadrilateral::quadrilateral_duffy;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::simplex_rules::simplex_rule;
use bempp_quadrature::types::{CellToCellConnectivity, TestTrialNumericalQuadratureDefinition};
use bempp_traits::bem::FunctionSpace;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{CellType, GridType, ReferenceMapType, TopologyType};
use bempp_traits::kernel::Kernel;
use bempp_traits::types::EvalType;
use bempp_traits::types::ReferenceCellType;
use num::Float;
use rayon::prelude::*;
use rlst_dense::{
    array::Array,
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_dynamic_array2, rlst_dynamic_array3, rlst_dynamic_array4,
    traits::{
        RandomAccessMut, RawAccess, RawAccessMut, Shape, UnsafeRandomAccessByRef,
        UnsafeRandomAccessMut,
    },
    types::RlstScalar,
};
use rlst_sparse::sparse::csr_mat::CsrMatrix;
use std::collections::HashMap;

type RlstArray<T, const DIM: usize> = Array<T, BaseArray<T, VectorContainer<T>, DIM>, DIM>;

fn equal_grids<TestGrid: GridType, TrialGrid: GridType>(
    test_grid: &TestGrid,
    trial_grid: &TrialGrid,
) -> bool {
    std::ptr::addr_of!(*test_grid) as usize == std::ptr::addr_of!(*trial_grid) as usize
}
fn neighbours<TestGrid: GridType, TrialGrid: GridType>(
    test_grid: &TestGrid,
    trial_grid: &TrialGrid,
    test_cell: usize,
    trial_cell: usize,
) -> bool {
    if !equal_grids(test_grid, trial_grid) {
        false
    } else {
        let test_vertices = trial_grid
            .cell_from_index(test_cell)
            .topology()
            .vertex_indices()
            .collect::<Vec<_>>();
        for v in trial_grid
            .cell_from_index(trial_cell)
            .topology()
            .vertex_indices()
        {
            if test_vertices.contains(&v) {
                return true;
            }
        }
        false
    }
}

fn get_quadrature_rule(
    test_celltype: ReferenceCellType,
    trial_celltype: ReferenceCellType,
    pairs: &[(usize, usize)],
    npoints: usize,
) -> TestTrialNumericalQuadratureDefinition {
    if pairs.is_empty() {
        panic!("Non-singular rule");
    } else {
        // Singular rules
        if test_celltype == ReferenceCellType::Triangle {
            if trial_celltype != ReferenceCellType::Triangle {
                unimplemented!("Mixed meshes not yet supported");
            }
            triangle_duffy(
                &CellToCellConnectivity {
                    connectivity_dimension: if pairs.len() == 1 {
                        0
                    } else if pairs.len() == 2 {
                        1
                    } else {
                        2
                    },
                    local_indices: pairs.to_vec(),
                },
                npoints,
            )
            .unwrap()
        } else {
            if test_celltype != ReferenceCellType::Quadrilateral {
                unimplemented!("Only triangles and quadrilaterals are currently supported");
            }
            if trial_celltype != ReferenceCellType::Quadrilateral {
                unimplemented!("Mixed meshes not yet supported");
            }
            quadrilateral_duffy(
                &CellToCellConnectivity {
                    connectivity_dimension: if pairs.len() == 1 {
                        0
                    } else if pairs.len() == 2 {
                        1
                    } else {
                        2
                    },
                    local_indices: pairs.to_vec(),
                },
                npoints,
            )
            .unwrap()
        }
    }
}

pub trait BatchedAssembler: Sync {
    //! Batched assembler
    //!
    //! Assemble operators by processing batches of cells in parallel

    /// Real scalar type
    type RealT: RlstScalar<Real = Self::RealT> + Float;
    /// Scalar type
    type T: RlstScalar<Real = Self::RealT>;
    /// Number of derivatives
    const DERIV_SIZE: usize;

    /// Return the kernel value to use in the integrand when using a singular quadrature rule
    ///
    /// # Safety
    /// This method is unsafe to allow `get_unchecked` may be used
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 2>,
        test_normals: &RlstArray<Self::RealT, 2>,
        trial_normals: &RlstArray<Self::RealT, 2>,
        index: usize,
    ) -> Self::T;

    /// Return the kernel value to use in the integrand when using a non-singular quadrature rule
    ///
    /// # Safety
    /// This method is unsafe to allow `get_unchecked` may be used
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 3>,
        test_normals: &RlstArray<Self::RealT, 2>,
        trial_normals: &RlstArray<Self::RealT, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> Self::T;

    /// Evaluate the kernel values for all source and target pairs
    ///
    /// For each source, the kernel is evaluated for exactly one target. This is equivalent to taking the diagonal of the matrix assembled by `kernel_assemble_st`
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[Self::RealT],
        targets: &[Self::RealT],
        result: &mut [Self::T],
    );

    /// Evaluate the kernel values for all sources and all targets
    ///
    /// For every source, the kernel is evaluated for every target.
    fn kernel_assemble_st(
        &self,
        sources: &[Self::RealT],
        targets: &[Self::RealT],
        result: &mut [Self::T],
    );

    /// Assemble the contribution to the terms of a matrix for a batch of pairs of adjacent cells
    #[allow(clippy::too_many_arguments)]
    fn assemble_batch_singular<
        'a,
        TestGrid: GridType<T = Self::RealT>,
        TrialGrid: GridType<T = Self::RealT>,
    >(
        &self,
        shape: [usize; 2],
        trial_space: &SerialFunctionSpace<'a, Self::T, TrialGrid>,
        test_space: &SerialFunctionSpace<'a, Self::T, TestGrid>,
        cell_pairs: &[(usize, usize)],
        trial_points: &RlstArray<Self::RealT, 2>,
        test_points: &RlstArray<Self::RealT, 2>,
        weights: &[Self::RealT],
        trial_table: &RlstArray<Self::T, 4>,
        test_table: &RlstArray<Self::T, 4>,
    ) -> SparseMatrixData<Self::T> {
        let mut output = SparseMatrixData::<Self::T>::new_known_size(
            shape,
            cell_pairs.len() * trial_space.element().dim() * test_space.element().dim(),
        );
        let npts = weights.len();
        debug_assert!(weights.len() == npts);
        debug_assert!(test_points.shape()[0] == npts);
        debug_assert!(trial_points.shape()[0] == npts);

        let grid = test_space.grid();
        assert_eq!(grid.physical_dimension(), 3);
        assert_eq!(grid.domain_dimension(), 2);

        // Memory assignment to be moved elsewhere as passed into here mutable?
        let mut k = rlst_dynamic_array2!(Self::T, [Self::DERIV_SIZE, npts]);
        let zero = num::cast::<f64, Self::RealT>(0.0).unwrap();
        let mut test_jdet = vec![zero; npts];
        let mut jacobian = [zero; 6];
        let mut normal = [zero; 3];
        let mut point = [zero; 3];
        let mut test_mapped_pts = rlst_dynamic_array2!(Self::RealT, [npts, 3]);
        let mut test_normals = rlst_dynamic_array2!(Self::RealT, [npts, 3]);

        let mut trial_jdet = vec![zero; npts];
        let mut trial_mapped_pts = rlst_dynamic_array2!(Self::RealT, [npts, 3]);
        let mut trial_normals = rlst_dynamic_array2!(Self::RealT, [npts, 3]);

        let test_evaluator = grid.reference_to_physical_map(test_points.data());
        let trial_evaluator = grid.reference_to_physical_map(trial_points.data());

        for (test_cell, trial_cell) in cell_pairs {
            for pt in 0..npts {
                test_evaluator.jacobian(*test_cell, pt, &mut jacobian);
                test_jdet[pt] = compute_det23(&jacobian);
                compute_normal_from_jacobian23(&jacobian, &mut normal);
                for (i, n) in normal.iter().enumerate() {
                    unsafe {
                        *test_normals.get_unchecked_mut([pt, i]) = *n;
                    }
                }
                test_evaluator.reference_to_physical(*test_cell, pt, &mut point);
                for (i, p) in point.iter().enumerate() {
                    unsafe {
                        *test_mapped_pts.get_unchecked_mut([pt, i]) = *p;
                    }
                }

                trial_evaluator.jacobian(*trial_cell, pt, &mut jacobian);
                trial_jdet[pt] = compute_det23(&jacobian);
                compute_normal_from_jacobian23(&jacobian, &mut normal);
                for (i, n) in normal.iter().enumerate() {
                    unsafe {
                        *trial_normals.get_unchecked_mut([pt, i]) = *n;
                    }
                }
                trial_evaluator.reference_to_physical(*trial_cell, pt, &mut point);
                for (i, p) in point.iter().enumerate() {
                    unsafe {
                        *trial_mapped_pts.get_unchecked_mut([pt, i]) = *p;
                    }
                }
            }

            self.kernel_assemble_diagonal_st(
                test_mapped_pts.data(),
                trial_mapped_pts.data(),
                k.data_mut(),
            );

            let test_dofs = test_space.cell_dofs(*test_cell).unwrap();
            let trial_dofs = trial_space.cell_dofs(*trial_cell).unwrap();
            for (test_i, test_dof) in test_dofs.iter().enumerate() {
                for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                    let mut sum = num::cast::<f64, Self::T>(0.0).unwrap();

                    for (index, wt) in weights.iter().enumerate() {
                        unsafe {
                            sum += self.singular_kernel_value(
                                &k,
                                &test_normals,
                                &trial_normals,
                                index,
                            ) * *test_table.get_unchecked([0, index, test_i, 0])
                                * *trial_table.get_unchecked([0, index, trial_i, 0])
                                * num::cast::<Self::RealT, Self::T>(
                                    *wt * *test_jdet.get_unchecked(index)
                                        * *trial_jdet.get_unchecked(index),
                                )
                                .unwrap();
                        }
                    }
                    output.rows.push(*test_dof);
                    output.cols.push(*trial_dof);
                    output.data.push(sum);
                }
            }
        }
        output
    }

    /// Assemble the contribution to the terms of a matrix for a batch of non-adjacent cells
    #[allow(clippy::too_many_arguments)]
    fn assemble_batch_nonadjacent<
        'a,
        const NPTS_TEST: usize,
        const NPTS_TRIAL: usize,
        TestGrid: GridType<T = Self::RealT>,
        TrialGrid: GridType<T = Self::RealT>,
    >(
        &self,
        output: &RawData2D<Self::T>,
        trial_space: &SerialFunctionSpace<'a, Self::T, TrialGrid>,
        trial_cells: &[usize],
        test_space: &SerialFunctionSpace<'a, Self::T, TestGrid>,
        test_cells: &[usize],
        trial_points: &RlstArray<Self::RealT, 2>,
        trial_weights: &[Self::RealT],
        test_points: &RlstArray<Self::RealT, 2>,
        test_weights: &[Self::RealT],
        trial_table: &RlstArray<Self::T, 4>,
        test_table: &RlstArray<Self::T, 4>,
    ) -> usize {
        debug_assert!(test_weights.len() == NPTS_TEST);
        debug_assert!(test_points.shape()[0] == NPTS_TEST);
        debug_assert!(trial_weights.len() == NPTS_TRIAL);
        debug_assert!(trial_points.shape()[0] == NPTS_TRIAL);

        let test_grid = test_space.grid();
        let trial_grid = trial_space.grid();

        assert_eq!(test_grid.physical_dimension(), 3);
        assert_eq!(test_grid.domain_dimension(), 2);
        assert_eq!(trial_grid.physical_dimension(), 3);
        assert_eq!(trial_grid.domain_dimension(), 2);

        let mut k = rlst_dynamic_array3!(Self::T, [NPTS_TEST, Self::DERIV_SIZE, NPTS_TRIAL]);
        let zero = num::cast::<f64, Self::RealT>(0.0).unwrap();
        let mut jacobian = [zero; 6];
        let mut normal = [zero; 3];
        let mut point = [zero; 3];
        let mut test_jdet = [zero; NPTS_TEST];
        let mut test_mapped_pts = rlst_dynamic_array2!(Self::RealT, [NPTS_TEST, 3]);
        let mut test_normals = rlst_dynamic_array2!(Self::RealT, [NPTS_TEST, 3]);

        let test_evaluator = test_grid.reference_to_physical_map(test_points.data());
        let trial_evaluator = trial_grid.reference_to_physical_map(trial_points.data());

        let mut trial_jdet = vec![[zero; NPTS_TRIAL]; trial_cells.len()];
        let mut trial_mapped_pts = vec![];
        let mut trial_normals = vec![];
        for _i in 0..trial_cells.len() {
            trial_mapped_pts.push(rlst_dynamic_array2!(Self::RealT, [NPTS_TRIAL, 3]));
            trial_normals.push(rlst_dynamic_array2!(Self::RealT, [NPTS_TRIAL, 3]));
        }

        for (trial_cell_i, trial_cell) in trial_cells.iter().enumerate() {
            for pt in 0..NPTS_TRIAL {
                trial_evaluator.jacobian(*trial_cell, pt, &mut jacobian);
                trial_jdet[trial_cell_i][pt] = compute_det23(&jacobian);
                compute_normal_from_jacobian23(&jacobian, &mut normal);
                for (i, n) in normal.iter().enumerate() {
                    unsafe {
                        *trial_normals[trial_cell_i].get_unchecked_mut([pt, i]) = *n;
                    }
                }
                trial_evaluator.reference_to_physical(*trial_cell, pt, &mut point);
                for (i, p) in point.iter().enumerate() {
                    unsafe {
                        *trial_mapped_pts[trial_cell_i].get_unchecked_mut([pt, i]) = *p;
                    }
                }
            }
        }

        let mut sum: Self::T;
        let mut trial_integrands = [num::cast::<f64, Self::T>(0.0).unwrap(); NPTS_TRIAL];

        for test_cell in test_cells {
            for (pt, jdet) in test_jdet.iter_mut().enumerate() {
                test_evaluator.jacobian(*test_cell, pt, &mut jacobian);
                *jdet = compute_det23(&jacobian);
                compute_normal_from_jacobian23(&jacobian, &mut normal);
                for (i, n) in normal.iter().enumerate() {
                    unsafe {
                        *test_normals.get_unchecked_mut([pt, i]) = *n;
                    }
                }
                test_evaluator.reference_to_physical(*test_cell, pt, &mut point);
                for (i, p) in point.iter().enumerate() {
                    unsafe {
                        *test_mapped_pts.get_unchecked_mut([pt, i]) = *p;
                    }
                }
            }

            for (trial_cell_i, trial_cell) in trial_cells.iter().enumerate() {
                if neighbours(test_grid, trial_grid, *test_cell, *trial_cell) {
                    continue;
                }

                self.kernel_assemble_st(
                    test_mapped_pts.data(),
                    trial_mapped_pts[trial_cell_i].data(),
                    k.data_mut(),
                );

                let test_dofs = test_space.cell_dofs(*test_cell).unwrap();
                let trial_dofs = trial_space.cell_dofs(*trial_cell).unwrap();

                for (test_i, test_dof) in test_dofs.iter().enumerate() {
                    for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                        for (trial_index, trial_wt) in trial_weights.iter().enumerate() {
                            trial_integrands[trial_index] = unsafe {
                                num::cast::<Self::RealT, Self::T>(
                                    *trial_wt * trial_jdet[trial_cell_i][trial_index],
                                )
                                .unwrap()
                                    * *trial_table.get_unchecked([0, trial_index, trial_i, 0])
                            };
                        }
                        sum = num::cast::<f64, Self::T>(0.0).unwrap();
                        for (test_index, test_wt) in test_weights.iter().enumerate() {
                            let test_integrand = unsafe {
                                num::cast::<Self::RealT, Self::T>(*test_wt * test_jdet[test_index])
                                    .unwrap()
                                    * *test_table.get_unchecked([0, test_index, test_i, 0])
                            };
                            for trial_index in 0..NPTS_TRIAL {
                                sum += unsafe {
                                    self.nonsingular_kernel_value(
                                        &k,
                                        &test_normals,
                                        &trial_normals[trial_cell_i],
                                        test_index,
                                        trial_index,
                                    ) * test_integrand
                                        * *trial_integrands.get_unchecked(trial_index)
                                };
                            }
                        }
                        // TODO: should we write into a result array, then copy into output after this loop?
                        unsafe {
                            *output.data.add(*test_dof + output.shape[0] * *trial_dof) += sum;
                        }
                    }
                }
            }
        }
        1
    }

    /// Assemble the contribution to the terms of a matrix for a batch of pairs of adjacent cells if an (incorrect) non-singular quadrature rule was used
    #[allow(clippy::too_many_arguments)]
    fn assemble_batch_singular_correction<
        'a,
        const NPTS_TEST: usize,
        const NPTS_TRIAL: usize,
        TestGrid: GridType<T = Self::RealT>,
        TrialGrid: GridType<T = Self::RealT>,
    >(
        &self,
        shape: [usize; 2],
        trial_space: &SerialFunctionSpace<'a, Self::T, TrialGrid>,
        test_space: &SerialFunctionSpace<'a, Self::T, TestGrid>,
        cell_pairs: &[(usize, usize)],
        trial_points: &RlstArray<Self::RealT, 2>,
        trial_weights: &[Self::RealT],
        test_points: &RlstArray<Self::RealT, 2>,
        test_weights: &[Self::RealT],
        trial_table: &RlstArray<Self::T, 4>,
        test_table: &RlstArray<Self::T, 4>,
    ) -> SparseMatrixData<Self::T> {
        let mut output = SparseMatrixData::<Self::T>::new_known_size(
            shape,
            cell_pairs.len() * trial_space.element().dim() * test_space.element().dim(),
        );
        debug_assert!(test_weights.len() == NPTS_TEST);
        debug_assert!(test_points.shape()[0] == NPTS_TEST);
        debug_assert!(trial_weights.len() == NPTS_TRIAL);
        debug_assert!(trial_points.shape()[0] == NPTS_TRIAL);

        let grid = test_space.grid();
        assert_eq!(grid.physical_dimension(), 3);
        assert_eq!(grid.domain_dimension(), 2);

        let mut k = rlst_dynamic_array3!(Self::T, [NPTS_TEST, Self::DERIV_SIZE, NPTS_TRIAL]);

        let zero = num::cast::<f64, Self::RealT>(0.0).unwrap();
        let mut jacobian = [zero; 6];
        let mut normal = [zero; 3];
        let mut point = [zero; 3];

        let mut test_jdet = vec![zero; NPTS_TEST];
        let mut test_mapped_pts = rlst_dynamic_array2!(Self::RealT, [NPTS_TEST, 3]);
        let mut test_normals = rlst_dynamic_array2!(Self::RealT, [NPTS_TEST, 3]);

        let mut trial_jdet = vec![zero; NPTS_TRIAL];
        let mut trial_mapped_pts = rlst_dynamic_array2!(Self::RealT, [NPTS_TRIAL, 3]);
        let mut trial_normals = rlst_dynamic_array2!(Self::RealT, [NPTS_TRIAL, 3]);

        let test_evaluator = grid.reference_to_physical_map(test_points.data());
        let trial_evaluator = grid.reference_to_physical_map(trial_points.data());

        let mut sum: Self::T;
        let mut trial_integrands = [num::cast::<f64, Self::T>(0.0).unwrap(); NPTS_TRIAL];

        for (test_cell, trial_cell) in cell_pairs {
            for (pt, jdet) in test_jdet.iter_mut().enumerate() {
                test_evaluator.jacobian(*test_cell, pt, &mut jacobian);
                *jdet = compute_det23(&jacobian);
                compute_normal_from_jacobian23(&jacobian, &mut normal);
                for (i, n) in normal.iter().enumerate() {
                    unsafe {
                        *test_normals.get_unchecked_mut([pt, i]) = *n;
                    }
                }
                test_evaluator.reference_to_physical(*test_cell, pt, &mut point);
                for (i, p) in point.iter().enumerate() {
                    unsafe {
                        *test_mapped_pts.get_unchecked_mut([pt, i]) = *p;
                    }
                }
            }
            for (pt, jdet) in trial_jdet.iter_mut().enumerate() {
                trial_evaluator.jacobian(*trial_cell, pt, &mut jacobian);
                *jdet = compute_det23(&jacobian);
                compute_normal_from_jacobian23(&jacobian, &mut normal);
                for (i, n) in normal.iter().enumerate() {
                    unsafe {
                        *trial_normals.get_unchecked_mut([pt, i]) = *n;
                    }
                }
                trial_evaluator.reference_to_physical(*trial_cell, pt, &mut point);
                for (i, p) in point.iter().enumerate() {
                    unsafe {
                        *trial_mapped_pts.get_unchecked_mut([pt, i]) = *p;
                    }
                }
            }

            self.kernel_assemble_st(
                test_mapped_pts.data(),
                trial_mapped_pts.data(),
                k.data_mut(),
            );

            let test_dofs = test_space.cell_dofs(*test_cell).unwrap();
            let trial_dofs = trial_space.cell_dofs(*trial_cell).unwrap();
            for (test_i, test_dof) in test_dofs.iter().enumerate() {
                for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                    for (trial_index, trial_wt) in trial_weights.iter().enumerate() {
                        trial_integrands[trial_index] = unsafe {
                            num::cast::<Self::RealT, Self::T>(*trial_wt * trial_jdet[trial_index])
                                .unwrap()
                                * *trial_table.get_unchecked([0, trial_index, trial_i, 0])
                        };
                    }
                    sum = num::cast::<f64, Self::T>(0.0).unwrap();
                    for (test_index, test_wt) in test_weights.iter().enumerate() {
                        let test_integrand = unsafe {
                            num::cast::<Self::RealT, Self::T>(*test_wt * test_jdet[test_index])
                                .unwrap()
                                * *test_table.get_unchecked([0, test_index, test_i, 0])
                        };
                        for trial_index in 0..NPTS_TRIAL {
                            sum += unsafe {
                                self.nonsingular_kernel_value(
                                    &k,
                                    &test_normals,
                                    &trial_normals,
                                    test_index,
                                    trial_index,
                                ) * test_integrand
                                    * *trial_integrands.get_unchecked(trial_index)
                            };
                        }
                    }
                    output.rows.push(*test_dof);
                    output.cols.push(*trial_dof);
                    output.data.push(sum);
                }
            }
        }
        output
    }

    /// Assemble the singular contributions
    fn assemble_singular<
        'a,
        const QDEGREE: usize,
        const BLOCKSIZE: usize,
        TestGrid: GridType<T = Self::RealT> + Sync,
        TrialGrid: GridType<T = Self::RealT> + Sync,
    >(
        &self,
        shape: [usize; 2],
        trial_space: &SerialFunctionSpace<'a, Self::T, TrialGrid>,
        test_space: &SerialFunctionSpace<'a, Self::T, TestGrid>,
    ) -> SparseMatrixData<Self::T> {
        let mut output = SparseMatrixData::new(shape);

        if !equal_grids(test_space.grid(), trial_space.grid()) {
            // If the test and trial grids are different, there are no neighbouring triangles
            return output;
        }

        if !trial_space.is_serial() || !test_space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if shape[0] != test_space.global_size() || shape[1] != trial_space.global_size() {
            panic!("Matrix has wrong shape");
        }

        let grid = test_space.grid();

        let mut possible_pairs = vec![];
        // Vertex-adjacent
        for i in 0..3 {
            for j in 0..3 {
                possible_pairs.push(vec![(i, j)]);
            }
        }
        // edge-adjacent
        for i in 0..3 {
            for j in i + 1..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        if k != l {
                            possible_pairs.push(vec![(k, i), (l, j)]);
                        }
                    }
                }
            }
        }
        // Same cell
        possible_pairs.push(vec![(0, 0), (1, 1), (2, 2)]);

        let mut pair_indices: HashMap<Vec<(usize, usize)>, usize> = HashMap::new();
        for (i, pairs) in possible_pairs.iter().enumerate() {
            pair_indices.insert(pairs.clone(), i);
        }

        let mut qweights = vec![];
        let mut trial_points = vec![];
        let mut test_points = vec![];
        let mut trial_tables = vec![];
        let mut test_tables = vec![];
        for pairs in &possible_pairs {
            let qrule = get_quadrature_rule(
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
                pairs,
                QDEGREE,
            );
            let npts = qrule.weights.len();

            let mut points = rlst_dynamic_array2!(Self::RealT, [npts, 2]);
            for i in 0..npts {
                for j in 0..2 {
                    *points.get_mut([i, j]).unwrap() =
                        num::cast::<f64, Self::RealT>(qrule.trial_points[2 * i + j]).unwrap();
                }
            }
            let mut table = rlst_dynamic_array4!(
                Self::T,
                trial_space
                    .element()
                    .tabulate_array_shape(0, points.shape()[0])
            );
            trial_space.element().tabulate(&points, 0, &mut table);
            trial_points.push(points);
            trial_tables.push(table);

            let mut points = rlst_dynamic_array2!(Self::RealT, [npts, 2]);
            for i in 0..npts {
                for j in 0..2 {
                    *points.get_mut([i, j]).unwrap() =
                        num::cast::<f64, Self::RealT>(qrule.test_points[2 * i + j]).unwrap();
                }
            }
            let mut table = rlst_dynamic_array4!(
                Self::T,
                test_space
                    .element()
                    .tabulate_array_shape(0, points.shape()[0])
            );
            test_space.element().tabulate(&points, 0, &mut table);
            test_points.push(points);
            test_tables.push(table);
            qweights.push(
                qrule
                    .weights
                    .iter()
                    .map(|w| num::cast::<f64, Self::RealT>(*w).unwrap())
                    .collect::<Vec<_>>(),
            );
        }
        let mut cell_pairs: Vec<Vec<(usize, usize)>> = vec![vec![]; possible_pairs.len()];
        for vertex in 0..grid.number_of_vertices() {
            let cells = grid
                .vertex_to_cells(vertex)
                .iter()
                .map(|c| c.cell)
                .collect::<Vec<_>>();
            for test_cell in &cells {
                for trial_cell in &cells {
                    let mut smallest = true;
                    let mut pairs = vec![];
                    for (trial_i, trial_v) in grid
                        .cell_from_index(*trial_cell)
                        .topology()
                        .vertex_indices()
                        .enumerate()
                    {
                        for (test_i, test_v) in grid
                            .cell_from_index(*test_cell)
                            .topology()
                            .vertex_indices()
                            .enumerate()
                        {
                            if test_v == trial_v {
                                if test_v < vertex {
                                    smallest = false;
                                    break;
                                }
                                pairs.push((test_i, trial_i));
                            }
                        }
                        if !smallest {
                            break;
                        }
                    }
                    if smallest {
                        cell_pairs[pair_indices[&pairs]].push((*test_cell, *trial_cell));
                    }
                }
            }
        }
        for (i, cells) in cell_pairs.iter().enumerate() {
            let mut start = 0;
            let mut cell_blocks = vec![];
            while start < cells.len() {
                let end = if start + BLOCKSIZE < cells.len() {
                    start + BLOCKSIZE
                } else {
                    cells.len()
                };
                cell_blocks.push(&cells[start..end]);
                start = end;
            }

            let numtasks = cell_blocks.len();
            let r: SparseMatrixData<Self::T> = (0..numtasks)
                .into_par_iter()
                .map(&|t| {
                    self.assemble_batch_singular(
                        shape,
                        trial_space,
                        test_space,
                        cell_blocks[t],
                        &trial_points[i],
                        &test_points[i],
                        &qweights[i],
                        &trial_tables[i],
                        &test_tables[i],
                    )
                })
                .reduce(|| SparseMatrixData::<Self::T>::new(shape), |a, b| a.sum(b));

            output.add(r);
        }
        output
    }

    /// Assemble the singular correction
    ///
    /// The singular correction is the contribution is the terms for adjacent cells are assembled using an (incorrect) non-singular quadrature rule
    fn assemble_singular_correction<
        'a,
        const NPTS_TEST: usize,
        const NPTS_TRIAL: usize,
        const BLOCKSIZE: usize,
        TestGrid: GridType<T = Self::RealT> + Sync,
        TrialGrid: GridType<T = Self::RealT> + Sync,
    >(
        &self,
        shape: [usize; 2],
        trial_space: &SerialFunctionSpace<'a, Self::T, TrialGrid>,
        test_space: &SerialFunctionSpace<'a, Self::T, TestGrid>,
    ) -> SparseMatrixData<Self::T> {
        if !equal_grids(test_space.grid(), trial_space.grid()) {
            // If the test and trial grids are different, there are no neighbouring triangles
            return SparseMatrixData::new(shape);
        }

        if !trial_space.is_serial() || !test_space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if shape[0] != test_space.global_size() || shape[1] != trial_space.global_size() {
            panic!("Matrix has wrong shape");
        }

        if NPTS_TEST != NPTS_TRIAL {
            panic!("FMM with different test and trial quadrature rules not yet supported");
        }

        let grid = test_space.grid();

        // TODO: pass cell types into this function
        let qrule_test = simplex_rule(ReferenceCellType::Triangle, NPTS_TEST).unwrap();
        let mut qpoints_test = rlst_dynamic_array2!(Self::RealT, [NPTS_TEST, 2]);
        for i in 0..NPTS_TEST {
            for j in 0..2 {
                *qpoints_test.get_mut([i, j]).unwrap() =
                    num::cast::<f64, Self::RealT>(qrule_test.points[2 * i + j]).unwrap();
            }
        }
        let qweights_test = qrule_test
            .weights
            .iter()
            .map(|w| num::cast::<f64, Self::RealT>(*w).unwrap())
            .collect::<Vec<_>>();
        let qrule_trial = simplex_rule(ReferenceCellType::Triangle, NPTS_TRIAL).unwrap();
        let mut qpoints_trial = rlst_dynamic_array2!(Self::RealT, [NPTS_TRIAL, 2]);
        for i in 0..NPTS_TRIAL {
            for j in 0..2 {
                *qpoints_trial.get_mut([i, j]).unwrap() =
                    num::cast::<f64, Self::RealT>(qrule_trial.points[2 * i + j]).unwrap();
            }
        }
        let qweights_trial = qrule_trial
            .weights
            .iter()
            .map(|w| num::cast::<f64, Self::RealT>(*w).unwrap())
            .collect::<Vec<_>>();

        let mut test_table = rlst_dynamic_array4!(
            Self::T,
            test_space.element().tabulate_array_shape(0, NPTS_TEST)
        );
        test_space
            .element()
            .tabulate(&qpoints_test, 0, &mut test_table);

        let mut trial_table = rlst_dynamic_array4!(
            Self::T,
            trial_space.element().tabulate_array_shape(0, NPTS_TRIAL)
        );
        trial_space
            .element()
            .tabulate(&qpoints_test, 0, &mut trial_table);

        let mut cell_pairs: Vec<(usize, usize)> = vec![];

        for vertex in 0..grid.number_of_vertices() {
            let cells = grid
                .vertex_to_cells(vertex)
                .iter()
                .map(|c| c.cell)
                .collect::<Vec<_>>();
            for test_cell in &cells {
                for trial_cell in &cells {
                    let mut smallest = true;
                    for trial_v in grid
                        .cell_from_index(*trial_cell)
                        .topology()
                        .vertex_indices()
                    {
                        for test_v in grid.cell_from_index(*test_cell).topology().vertex_indices() {
                            if test_v == trial_v && test_v < vertex {
                                smallest = false;
                                break;
                            }
                        }
                        if !smallest {
                            break;
                        }
                    }
                    if smallest {
                        cell_pairs.push((*test_cell, *trial_cell));
                    }
                }
            }
        }

        let mut start = 0;
        let mut cell_blocks = vec![];
        while start < cell_pairs.len() {
            let end = if start + BLOCKSIZE < cell_pairs.len() {
                start + BLOCKSIZE
            } else {
                cell_pairs.len()
            };
            cell_blocks.push(&cell_pairs[start..end]);
            start = end;
        }

        let numtasks = cell_blocks.len();
        (0..numtasks)
            .into_par_iter()
            .map(&|t| {
                self.assemble_batch_singular_correction::<NPTS_TEST, NPTS_TRIAL, TestGrid, TrialGrid>(
                    shape,
                    trial_space,
                    test_space,
                    cell_blocks[t],
                    &qpoints_trial,
                    &qweights_trial,
                    &qpoints_test,
                    &qweights_test,
                    &trial_table,
                    &test_table,
                )
            })
            .reduce(|| SparseMatrixData::<Self::T>::new(shape), |a, b| a.sum(b))
    }

    /// Assemble the singular contributions into a dense matrix
    fn assemble_singular_into_dense<
        'a,
        const QDEGREE: usize,
        const BLOCKSIZE: usize,
        TestGrid: GridType<T = Self::RealT> + Sync,
        TrialGrid: GridType<T = Self::RealT> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &SerialFunctionSpace<'a, Self::T, TrialGrid>,
        test_space: &SerialFunctionSpace<'a, Self::T, TestGrid>,
    ) {
        let sparse_matrix = self.assemble_singular::<QDEGREE, BLOCKSIZE, TestGrid, TrialGrid>(
            output.shape(),
            trial_space,
            test_space,
        );
        let data = sparse_matrix.data;
        let rows = sparse_matrix.rows;
        let cols = sparse_matrix.cols;
        for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
            *output.get_mut([*i, *j]).unwrap() += *value;
        }
    }

    /// Assemble the singular contributions into a CSR sparse matrix
    fn assemble_singular_into_csr<
        'a,
        const QDEGREE: usize,
        const BLOCKSIZE: usize,
        TestGrid: GridType<T = Self::RealT> + Sync,
        TrialGrid: GridType<T = Self::RealT> + Sync,
    >(
        &self,
        trial_space: &SerialFunctionSpace<'a, Self::T, TrialGrid>,
        test_space: &SerialFunctionSpace<'a, Self::T, TestGrid>,
    ) -> CsrMatrix<Self::T> {
        let shape = [test_space.global_size(), trial_space.global_size()];
        let sparse_matrix = self.assemble_singular::<QDEGREE, BLOCKSIZE, TestGrid, TrialGrid>(
            shape,
            trial_space,
            test_space,
        );

        CsrMatrix::<Self::T>::from_aij(
            sparse_matrix.shape,
            &sparse_matrix.rows,
            &sparse_matrix.cols,
            &sparse_matrix.data,
        )
        .unwrap()
    }

    /// Assemble the singular correction into a dense matrix
    ///
    /// The singular correction is the contribution is the terms for adjacent cells are assembled using an (incorrect) non-singular quadrature rule
    fn assemble_singular_correction_into_dense<
        'a,
        const NPTS_TEST: usize,
        const NPTS_TRIAL: usize,
        const BLOCKSIZE: usize,
        TestGrid: GridType<T = Self::RealT> + Sync,
        TrialGrid: GridType<T = Self::RealT> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &SerialFunctionSpace<'a, Self::T, TrialGrid>,
        test_space: &SerialFunctionSpace<'a, Self::T, TestGrid>,
    ) {
        let sparse_matrix = self
            .assemble_singular_correction::<NPTS_TEST, NPTS_TRIAL, BLOCKSIZE, TestGrid, TrialGrid>(
                output.shape(),
                trial_space,
                test_space,
            );
        let data = sparse_matrix.data;
        let rows = sparse_matrix.rows;
        let cols = sparse_matrix.cols;
        for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
            *output.get_mut([*i, *j]).unwrap() += *value;
        }
    }

    /// Assemble the singular correction into a CSR matrix
    ///
    /// The singular correction is the contribution is the terms for adjacent cells are assembled using an (incorrect) non-singular quadrature rule
    fn assemble_singular_correction_into_csr<
        'a,
        const NPTS_TEST: usize,
        const NPTS_TRIAL: usize,
        const BLOCKSIZE: usize,
        TestGrid: GridType<T = Self::RealT> + Sync,
        TrialGrid: GridType<T = Self::RealT> + Sync,
    >(
        &self,
        trial_space: &SerialFunctionSpace<'a, Self::T, TrialGrid>,
        test_space: &SerialFunctionSpace<'a, Self::T, TestGrid>,
    ) -> CsrMatrix<Self::T> {
        let shape = [test_space.global_size(), trial_space.global_size()];
        let sparse_matrix = self
            .assemble_singular_correction::<NPTS_TEST, NPTS_TRIAL, BLOCKSIZE, TestGrid, TrialGrid>(
                shape,
                trial_space,
                test_space,
            );

        CsrMatrix::<Self::T>::from_aij(
            sparse_matrix.shape,
            &sparse_matrix.rows,
            &sparse_matrix.cols,
            &sparse_matrix.data,
        )
        .unwrap()
    }

    /// Assemble into a dense matrix
    fn assemble_into_dense<
        'a,
        const BLOCKSIZE: usize,
        TestGrid: GridType<T = Self::RealT> + Sync,
        TrialGrid: GridType<T = Self::RealT> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &SerialFunctionSpace<'a, Self::T, TrialGrid>,
        test_space: &SerialFunctionSpace<'a, Self::T, TestGrid>,
    ) {
        let test_colouring = test_space.compute_cell_colouring();
        let trial_colouring = trial_space.compute_cell_colouring();

        self.assemble_nonsingular_into_dense::<16, 16, BLOCKSIZE, TestGrid, TrialGrid>(
            output,
            trial_space,
            test_space,
            &trial_colouring,
            &test_colouring,
        );
        self.assemble_singular_into_dense::<4, BLOCKSIZE, TestGrid, TrialGrid>(
            output,
            trial_space,
            test_space,
        );
    }

    /// Assemble the non-singular contributions into a dense matrix
    fn assemble_nonsingular_into_dense<
        'a,
        const NPTS_TEST: usize,
        const NPTS_TRIAL: usize,
        const BLOCKSIZE: usize,
        TestGrid: GridType<T = Self::RealT> + Sync,
        TrialGrid: GridType<T = Self::RealT> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &SerialFunctionSpace<'a, Self::T, TrialGrid>,
        test_space: &SerialFunctionSpace<'a, Self::T, TestGrid>,
        trial_colouring: &Vec<Vec<usize>>,
        test_colouring: &Vec<Vec<usize>>,
    ) {
        if !trial_space.is_serial() || !test_space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if output.shape()[0] != test_space.global_size()
            || output.shape()[1] != trial_space.global_size()
        {
            panic!("Matrix has wrong shape");
        }

        // TODO: pass cell types into this function
        let qrule_test = simplex_rule(ReferenceCellType::Triangle, NPTS_TEST).unwrap();
        let mut qpoints_test = rlst_dynamic_array2!(Self::RealT, [NPTS_TEST, 2]);
        for i in 0..NPTS_TEST {
            for j in 0..2 {
                *qpoints_test.get_mut([i, j]).unwrap() =
                    num::cast::<f64, Self::RealT>(qrule_test.points[2 * i + j]).unwrap();
            }
        }
        let qweights_test = qrule_test
            .weights
            .iter()
            .map(|w| num::cast::<f64, Self::RealT>(*w).unwrap())
            .collect::<Vec<_>>();
        let qrule_trial = simplex_rule(ReferenceCellType::Triangle, NPTS_TRIAL).unwrap();
        let mut qpoints_trial = rlst_dynamic_array2!(Self::RealT, [NPTS_TRIAL, 2]);
        for i in 0..NPTS_TRIAL {
            for j in 0..2 {
                *qpoints_trial.get_mut([i, j]).unwrap() =
                    num::cast::<f64, Self::RealT>(qrule_trial.points[2 * i + j]).unwrap();
            }
        }
        let qweights_trial = qrule_trial
            .weights
            .iter()
            .map(|w| num::cast::<f64, Self::RealT>(*w).unwrap())
            .collect::<Vec<_>>();

        let mut test_table = rlst_dynamic_array4!(
            Self::T,
            test_space.element().tabulate_array_shape(0, NPTS_TEST)
        );
        test_space
            .element()
            .tabulate(&qpoints_test, 0, &mut test_table);

        let mut trial_table = rlst_dynamic_array4!(
            Self::T,
            trial_space.element().tabulate_array_shape(0, NPTS_TRIAL)
        );
        trial_space
            .element()
            .tabulate(&qpoints_test, 0, &mut trial_table);

        let output_raw = RawData2D {
            data: output.data_mut().as_mut_ptr(),
            shape: output.shape(),
        };

        for test_c in test_colouring {
            for trial_c in trial_colouring {
                let mut test_cells: Vec<&[usize]> = vec![];
                let mut trial_cells: Vec<&[usize]> = vec![];

                let mut test_start = 0;
                while test_start < test_c.len() {
                    let test_end = if test_start + BLOCKSIZE < test_c.len() {
                        test_start + BLOCKSIZE
                    } else {
                        test_c.len()
                    };

                    let mut trial_start = 0;
                    while trial_start < trial_c.len() {
                        let trial_end = if trial_start + BLOCKSIZE < trial_c.len() {
                            trial_start + BLOCKSIZE
                        } else {
                            trial_c.len()
                        };
                        test_cells.push(&test_c[test_start..test_end]);
                        trial_cells.push(&trial_c[trial_start..trial_end]);
                        trial_start = trial_end;
                    }
                    test_start = test_end
                }

                let numtasks = test_cells.len();
                let r: usize = (0..numtasks)
                    .into_par_iter()
                    .map(&|t| {
                        self.assemble_batch_nonadjacent::<NPTS_TEST, NPTS_TRIAL, TestGrid, TrialGrid>(
                            &output_raw,
                            trial_space,
                            trial_cells[t],
                            test_space,
                            test_cells[t],
                            &qpoints_trial,
                            &qweights_trial,
                            &qpoints_test,
                            &qweights_test,
                            &trial_table,
                            &test_table,
                        )
                    })
                    .sum();
                assert_eq!(r, numtasks);
            }
        }
    }
}

/// Single layer assembler
trait SingleLayerAssembler: Sync {
    /// Type of the kernel
    type K: Kernel;
    /// Get the kernsl
    fn kernel(&self) -> &Self::K;
}
impl<K: Kernel, A: SingleLayerAssembler<K=K>> BatchedAssembler for A
{
    const DERIV_SIZE: usize = 1;
    type RealT = <K::T as RlstScalar>::Real;
    type T = K::T;
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 2>,
        _test_normals: &RlstArray<Self::RealT, 2>,
        _trial_normals: &RlstArray<Self::RealT, 2>,
        index: usize,
    ) -> Self::T {
        *k.get_unchecked([0, index])
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 3>,
        _test_normals: &RlstArray<Self::RealT, 2>,
        _trial_normals: &RlstArray<Self::RealT, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> Self::T {
        *k.get_unchecked([test_index, 0, trial_index])
    }
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[Self::RealT],
        targets: &[Self::RealT],
        result: &mut [Self::T],
    ) {
        self.kernel()
            .assemble_diagonal_st(EvalType::Value, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[Self::RealT], targets: &[Self::RealT], result: &mut [Self::T]) {
        self.kernel()
            .assemble_st(EvalType::Value, sources, targets, result);
    }
}

/// Assembler for a Helmholtz single layer boundary operator
pub struct HelmholtzSingleLayerAssembler<T: RlstScalar<Complex=T>> {
    kernel: Helmholtz3dKernel<T>,
}
impl<T: RlstScalar<Complex=T>> HelmholtzSingleLayerAssembler<T> {
    fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
        }
    }
}
impl<T: RlstScalar<Complex=T>> SingleLayerAssembler for HelmholtzSingleLayerAssembler<T> {
    type K = Helmholtz3dKernel<T>;
    fn kernel(&self) -> &Self::K {
        &self.kernel
    }

}

/// Assembler for a Laplace single layer operator
pub struct LaplaceSingleLayerAssembler<T: RlstScalar> {
    kernel: Laplace3dKernel<T>,
}
impl<T: RlstScalar> Default for LaplaceSingleLayerAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
        }
    }
}
impl<T: RlstScalar> SingleLayerAssembler for LaplaceSingleLayerAssembler<T> {
    type K = Laplace3dKernel<T>;
    fn kernel(&self) -> &Self::K {
        &self.kernel
    }

}

/// Assembler for a Laplace double layer operator
pub struct LaplaceDoubleLayerAssembler<T: RlstScalar> {
    kernel: Laplace3dKernel<T>,
}
impl<T: RlstScalar> Default for LaplaceDoubleLayerAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
        }
    }
}
unsafe impl<T: RlstScalar> Sync for LaplaceDoubleLayerAssembler<T> {}
impl<T: RlstScalar> BatchedAssembler for LaplaceDoubleLayerAssembler<T> {
    const DERIV_SIZE: usize = 4;
    type RealT = T::Real;
    type T = T;
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<T, 2>,
        _test_normals: &RlstArray<<T as RlstScalar>::Real, 2>,
        trial_normals: &RlstArray<<T as RlstScalar>::Real, 2>,
        index: usize,
    ) -> T {
        *k.get_unchecked([1, index])
            * num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 0])).unwrap()
            + *k.get_unchecked([2, index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 1])).unwrap()
            + *k.get_unchecked([3, index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 2])).unwrap()
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        _test_normals: &RlstArray<<T as RlstScalar>::Real, 2>,
        trial_normals: &RlstArray<<T as RlstScalar>::Real, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> T {
        *k.get_unchecked([test_index, 1, trial_index])
            * num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 0])).unwrap()
            + *k.get_unchecked([test_index, 2, trial_index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 1])).unwrap()
            + *k.get_unchecked([test_index, 3, trial_index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 2])).unwrap()
    }
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::ValueDeriv, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(EvalType::ValueDeriv, sources, targets, result);
    }
}

/// Assembler for a Laplace adjoint double layer operator
pub struct LaplaceAdjointDoubleLayerAssembler<T: RlstScalar> {
    kernel: Laplace3dKernel<T>,
}
impl<T: RlstScalar> Default for LaplaceAdjointDoubleLayerAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
        }
    }
}
unsafe impl<T: RlstScalar> Sync for LaplaceAdjointDoubleLayerAssembler<T> {}
impl<T: RlstScalar> BatchedAssembler for LaplaceAdjointDoubleLayerAssembler<T> {
    const DERIV_SIZE: usize = 4;
    type RealT = T::Real;
    type T = T;
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<T, 2>,
        test_normals: &RlstArray<<T as RlstScalar>::Real, 2>,
        _trial_normals: &RlstArray<<T as RlstScalar>::Real, 2>,
        index: usize,
    ) -> T {
        -*k.get_unchecked([1, index])
            * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 0])).unwrap()
            - *k.get_unchecked([2, index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 1])).unwrap()
            - *k.get_unchecked([3, index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 2])).unwrap()
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        test_normals: &RlstArray<<T as RlstScalar>::Real, 2>,
        _trial_normals: &RlstArray<<T as RlstScalar>::Real, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> T {
        -*k.get_unchecked([test_index, 1, trial_index])
            * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 0])).unwrap()
            - *k.get_unchecked([test_index, 2, trial_index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 1])).unwrap()
            - *k.get_unchecked([test_index, 3, trial_index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 2])).unwrap()
    }
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::ValueDeriv, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(EvalType::ValueDeriv, sources, targets, result);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::function_space::SerialFunctionSpace;
    use approx::*;
    use bempp_element::element::{create_element, ElementFamily};
    use bempp_grid::{
        flat_triangle_grid::SerialFlatTriangleGrid, shapes::regular_sphere,
        traits_impl::WrappedGrid,
    };
    use bempp_traits::element::Continuity;
    use rlst_dense::traits::RandomAccessByRef;

    #[test]
    fn test_singular_dp0() {
        let grid = regular_sphere::<f64>(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
        let assembler = LaplaceSingleLayerAssembler::default();
        assembler.assemble_singular_into_dense::<4, 128, WrappedGrid<SerialFlatTriangleGrid<f64>>, WrappedGrid<SerialFlatTriangleGrid<f64>>>(&mut matrix, &space, &space);
        let csr = assembler.assemble_singular_into_csr::<4, 128, WrappedGrid<SerialFlatTriangleGrid<f64>>, WrappedGrid<SerialFlatTriangleGrid<f64>>>(&space, &space);

        let indptr = csr.indptr();
        let indices = csr.indices();
        let data = csr.data();

        let mut row = 0;
        for (i, j) in indices.iter().enumerate() {
            while i >= indptr[row + 1] {
                row += 1;
            }
            assert_relative_eq!(*matrix.get([row, *j]).unwrap(), data[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_singular_p1() {
        let grid = regular_sphere::<f64>(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
        let assembler = LaplaceSingleLayerAssembler::default();
        assembler.assemble_singular_into_dense::<4, 128, WrappedGrid<SerialFlatTriangleGrid<f64>>, WrappedGrid<SerialFlatTriangleGrid<f64>>>(&mut matrix, &space, &space);
        let csr = assembler.assemble_singular_into_csr::<4, 128, WrappedGrid<SerialFlatTriangleGrid<f64>>, WrappedGrid<SerialFlatTriangleGrid<f64>>>(&space, &space);

        let indptr = csr.indptr();
        let indices = csr.indices();
        let data = csr.data();

        let mut row = 0;
        for (i, j) in indices.iter().enumerate() {
            while i >= indptr[row + 1] {
                row += 1;
            }
            assert_relative_eq!(*matrix.get([row, *j]).unwrap(), data[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_singular_dp0_p1() {
        let grid = regular_sphere::<f64>(0);
        let element0 = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let element1 = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let space0 = SerialFunctionSpace::new(&grid, &element0);
        let space1 = SerialFunctionSpace::new(&grid, &element1);

        let ndofs0 = space0.global_size();
        let ndofs1 = space1.global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs1, ndofs0]);
        let assembler = LaplaceSingleLayerAssembler::default();
        assembler.assemble_singular_into_dense::<4, 128, WrappedGrid<SerialFlatTriangleGrid<f64>>, WrappedGrid<SerialFlatTriangleGrid<f64>>>(&mut matrix, &space0, &space1);
        let csr = assembler.assemble_singular_into_csr::<4, 128, WrappedGrid<SerialFlatTriangleGrid<f64>>, WrappedGrid<SerialFlatTriangleGrid<f64>>>(&space0, &space1);
        let indptr = csr.indptr();
        let indices = csr.indices();
        let data = csr.data();

        let mut row = 0;
        for (i, j) in indices.iter().enumerate() {
            while i >= indptr[row + 1] {
                row += 1;
            }
            assert_relative_eq!(*matrix.get([row, *j]).unwrap(), data[i], epsilon = 1e-8);
        }
    }
}
