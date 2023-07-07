//! Implementation of the Laplace kernel
use num;
use std::marker::PhantomData;

use crate::helpers::{check_dimensions_assemble, check_dimensions_evaluate};
use bempp_traits::{
    kernel::{Kernel, KernelScale},
    types::{EvalType, KernelType, Scalar},
};
use num::traits::FloatConst;
use rayon::prelude::*;

#[derive(Clone)]
pub struct Laplace3dKernel<T: Scalar> {
    kernel_type: KernelType,
    _phantom_t: std::marker::PhantomData<T>,
}

impl<T: Scalar<Real = T>> KernelScale for Laplace3dKernel<T> {
    type T = T;

    fn scale(&self, level: u64) -> Self::T {
        let numerator = T::from(1).unwrap();
        let denominator = T::from(2.).unwrap();
        let power = T::from(level).unwrap();
        let denominator = denominator.powf(power);
        numerator / denominator
    }
}

impl<T: Scalar> Laplace3dKernel<T> {
    pub fn new() -> Self {
        Self {
            kernel_type: KernelType::Laplace,
            _phantom_t: PhantomData,
        }
    }
}

impl<T: Scalar> Default for Laplace3dKernel<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Send + Sync> Kernel for Laplace3dKernel<T>
where
    <T as Scalar>::Real: Send + Sync,
{
    type T = T;

    fn domain_component_count(&self) -> usize {
        1
    }

    fn space_dimension(&self) -> usize {
        3
    }

    fn kernel_type(&self) -> &KernelType {
        &self.kernel_type
    }

    fn evaluate_st(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as Scalar>::Real],
        targets: &[<Self::T as Scalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
    ) {
        check_dimensions_evaluate(self, eval_type, sources, targets, charges, result);
        let ntargets = targets.len() / self.space_dimension();
        let range_dim = self.range_component_count(eval_type);

        result
            .chunks_exact_mut(range_dim)
            .enumerate()
            .for_each(|(target_index, my_chunk)| {
                let target = [
                    targets[target_index],
                    targets[ntargets + target_index],
                    targets[2 * ntargets + target_index],
                ];

                evaluate_laplace_one_target(eval_type, &target, sources, charges, my_chunk)
            });
    }

    fn evaluate_mt(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as Scalar>::Real],
        targets: &[<Self::T as Scalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
    ) {
        check_dimensions_evaluate(self, eval_type, sources, targets, charges, result);
        let ntargets = targets.len() / self.space_dimension();
        let range_dim = self.range_component_count(eval_type);

        result
            .par_chunks_exact_mut(range_dim)
            .enumerate()
            .for_each(|(target_index, my_chunk)| {
                let target = [
                    targets[target_index],
                    targets[ntargets + target_index],
                    targets[2 * ntargets + target_index],
                ];

                evaluate_laplace_one_target(eval_type, &target, sources, charges, my_chunk)
            });
    }

    fn assemble_st(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as Scalar>::Real],
        targets: &[<Self::T as Scalar>::Real],
        result: &mut [Self::T],
    ) {
        check_dimensions_assemble(self, eval_type, sources, targets, result);
        let ntargets = targets.len() / self.space_dimension();
        let nsources = sources.len() / self.space_dimension();
        let range_dim = self.range_component_count(eval_type);

        result
            .chunks_exact_mut(range_dim * nsources)
            .enumerate()
            .for_each(|(target_index, my_chunk)| {
                let target = [
                    targets[target_index],
                    targets[ntargets + target_index],
                    targets[2 * ntargets + target_index],
                ];

                assemble_laplace_one_target(eval_type, &target, sources, my_chunk)
            });
    }

    fn assemble_mt(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as Scalar>::Real],
        targets: &[<Self::T as Scalar>::Real],
        result: &mut [Self::T],
    ) {
        check_dimensions_assemble(self, eval_type, sources, targets, result);
        let ntargets = targets.len() / self.space_dimension();
        let nsources = sources.len() / self.space_dimension();
        let range_dim = self.range_component_count(eval_type);

        result
            .par_chunks_exact_mut(range_dim * nsources)
            .enumerate()
            .for_each(|(target_index, my_chunk)| {
                let target = [
                    targets[target_index],
                    targets[ntargets + target_index],
                    targets[2 * ntargets + target_index],
                ];

                assemble_laplace_one_target(eval_type, &target, sources, my_chunk)
            });
    }

    fn range_component_count(&self, eval_type: EvalType) -> usize {
        laplace_component_count(eval_type)
    }
}

pub fn evaluate_laplace_one_target<T: Scalar>(
    eval_type: EvalType,
    target: &[<T as Scalar>::Real],
    sources: &[<T as Scalar>::Real],
    charges: &[T],
    result: &mut [T],
) {
    let ncharges = charges.len();
    let nsources = ncharges;
    let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();
    let zero_real = <T::Real as num::Zero>::zero();
    let one_real = <T::Real as num::One>::one();

    let sources0 = &sources[0..nsources];
    let sources1 = &sources[nsources..2 * nsources];
    let sources2 = &sources[2 * nsources..3 * nsources];

    let mut diff0: T::Real;
    let mut diff1: T::Real;
    let mut diff2: T::Real;

    match eval_type {
        EvalType::Value => {
            let mut my_result = T::zero();
            for index in 0..nsources {
                diff0 = sources0[index] - target[0];
                diff1 = sources1[index] - target[1];
                diff2 = sources2[index] - target[2];
                let diff_norm = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2).sqrt();
                let inv_diff_norm = {
                    if diff_norm == zero_real {
                        zero_real
                    } else {
                        one_real / diff_norm
                    }
                };

                my_result += charges[index].mul_real(inv_diff_norm);
            }
            result[0] = my_result.mul_real(m_inv_4pi);
        }
        EvalType::ValueDeriv => {
            // Cannot simply use an array my_result as this is not
            // correctly auto-vectorized.

            let mut my_result0 = T::zero();
            let mut my_result1 = T::zero();
            let mut my_result2 = T::zero();
            let mut my_result3 = T::zero();

            for index in 0..nsources {
                diff0 = sources0[index] - target[0];
                diff1 = sources1[index] - target[1];
                diff2 = sources2[index] - target[2];
                let diff_norm = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2).sqrt();
                let inv_diff_norm = {
                    if diff_norm == zero_real {
                        zero_real
                    } else {
                        one_real / diff_norm
                    }
                };
                let inv_diff_norm_cubed = inv_diff_norm * inv_diff_norm * inv_diff_norm;

                my_result0 += charges[index].mul_real(inv_diff_norm);
                my_result1 += charges[index].mul_real(diff0 * inv_diff_norm_cubed);
                my_result2 += charges[index].mul_real(diff1 * inv_diff_norm_cubed);
                my_result3 += charges[index].mul_real(diff2 * inv_diff_norm_cubed);
            }

            result[0] = my_result0.mul_real(m_inv_4pi);
            result[1] = my_result1.mul_real(m_inv_4pi);
            result[2] = my_result2.mul_real(m_inv_4pi);
            result[3] = my_result3.mul_real(m_inv_4pi);
        }
    }
}

pub fn assemble_laplace_one_target<T: Scalar>(
    eval_type: EvalType,
    target: &[<T as Scalar>::Real],
    sources: &[<T as Scalar>::Real],
    result: &mut [T],
) {
    assert_eq!(sources.len() % 3, 0);
    assert_eq!(target.len(), 3);
    let nsources = sources.len() / 3;
    let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();
    let zero_real = <T::Real as num::Zero>::zero();
    let one_real = <T::Real as num::One>::one();

    let sources0 = &sources[0..nsources];
    let sources1 = &sources[nsources..2 * nsources];
    let sources2 = &sources[2 * nsources..3 * nsources];

    let mut diff0: T::Real;
    let mut diff1: T::Real;
    let mut diff2: T::Real;

    match eval_type {
        EvalType::Value => {
            let mut my_result;
            for index in 0..nsources {
                diff0 = sources0[index] - target[0];
                diff1 = sources1[index] - target[1];
                diff2 = sources2[index] - target[2];
                let diff_norm = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2).sqrt();
                let inv_diff_norm = {
                    if diff_norm == zero_real {
                        zero_real
                    } else {
                        one_real / diff_norm
                    }
                };

                my_result = inv_diff_norm * m_inv_4pi;
                result[index] = num::cast::<T::Real, T>(my_result).unwrap();
            }
        }
        EvalType::ValueDeriv => {
            // Cannot simply use an array my_result as this is not
            // correctly auto-vectorized.

            let mut my_result0;
            let mut my_result1;
            let mut my_result2;
            let mut my_result3;

            let mut chunks = result.chunks_exact_mut(nsources);

            let my_res0 = chunks.next().unwrap();
            let my_res1 = chunks.next().unwrap();
            let my_res2 = chunks.next().unwrap();
            let my_res3 = chunks.next().unwrap();

            for index in 0..nsources {
                //let my_res = &mut result[4 * index..4 * (index + 1)];
                diff0 = sources0[index] - target[0];
                diff1 = sources1[index] - target[1];
                diff2 = sources2[index] - target[2];
                let diff_norm = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2).sqrt();
                let inv_diff_norm = {
                    if diff_norm == zero_real {
                        zero_real
                    } else {
                        one_real / diff_norm
                    }
                };
                let inv_diff_norm_cubed = inv_diff_norm * inv_diff_norm * inv_diff_norm;

                my_result0 = T::one().mul_real(inv_diff_norm * m_inv_4pi);
                my_result1 = T::one().mul_real(diff0 * inv_diff_norm_cubed * m_inv_4pi);
                my_result2 = T::one().mul_real(diff1 * inv_diff_norm_cubed * m_inv_4pi);
                my_result3 = T::one().mul_real(diff2 * inv_diff_norm_cubed * m_inv_4pi);

                my_res0[index] = my_result0;
                my_res1[index] = my_result1;
                my_res2[index] = my_result2;
                my_res3[index] = my_result3;
            }
        }
    }
}

fn laplace_component_count(eval_type: EvalType) -> usize {
    match eval_type {
        EvalType::Value => 1,
        EvalType::ValueDeriv => 4,
    }
}

// pub fn simd_wrapper_evaluate(
//     eval_type: EvalType,
//     target: &[f32],
//     sources: &[f32],
//     charges: &[f32],
//     result: &mut [f32],
// ) {
//     evaluate_laplace_one_target(eval_type, target, sources, charges, result)
// }

// pub fn simd_wrapper_assemble(
//     eval_type: EvalType,
//     target: &[f32],
//     sources: &[f32],
//     result: &mut [f32],
// ) {
//     assemble_laplace_one_target(eval_type, target, sources, result);
// }

#[cfg(test)]
mod test {

    use super::*;
    use approx::assert_relative_eq;
    use bempp_traits::types::Scalar;
    use rlst;
    use rlst::common::traits::{Copy, Eval, Transpose};
    use rlst::dense::traits::*;

    #[test]
    fn test_laplace_3d() {
        let eps = 1E-8;

        let nsources = 5;
        let ntargets = 3;

        let sources = rlst::dense::rlst_rand_mat![f64, (nsources, 3)];
        let targets = rlst::dense::rlst_rand_mat![f64, (ntargets, 3)];
        let charges = rlst::dense::rlst_rand_col_vec![f64, nsources];
        let mut green_value = rlst::dense::rlst_rand_col_vec![f64, ntargets];

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            charges.data(),
            green_value.data_mut(),
        );

        for target_index in 0..ntargets {
            let mut expected: f64 = 0.0;
            for source_index in 0..nsources {
                let dist = ((targets[[target_index, 0]] - sources[[source_index, 0]]).square()
                    + (targets[[target_index, 1]] - sources[[source_index, 1]]).square()
                    + (targets[[target_index, 2]] - sources[[source_index, 2]]).square())
                .sqrt();

                expected += charges[[source_index, 0]] * 0.25 * f64::FRAC_1_PI() / dist;
            }

            assert_relative_eq!(green_value[[target_index, 0]], expected, epsilon = 1E-12);
        }

        let mut targets_x_eps = targets.copy();
        let mut targets_y_eps = targets.copy();
        let mut targets_z_eps = targets.copy();

        for index in 0..ntargets {
            targets_x_eps[[index, 0]] += eps;
            targets_y_eps[[index, 1]] += eps;
            targets_z_eps[[index, 2]] += eps;
        }

        let mut expected = rlst::dense::rlst_mat![f64, (4, ntargets)];

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            charges.data(),
            expected.data_mut(),
        );

        let mut green_value_x_eps = rlst::dense::rlst_col_vec![f64, ntargets];

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_x_eps.data(),
            charges.data(),
            green_value_x_eps.data_mut(),
        );

        let mut green_value_y_eps = rlst::dense::rlst_col_vec![f64, ntargets];

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_y_eps.data(),
            charges.data(),
            green_value_y_eps.data_mut(),
        );
        let mut green_value_z_eps = rlst::dense::rlst_col_vec![f64, ntargets];

        Laplace3dKernel::<f64>::default().evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_z_eps.data(),
            charges.data(),
            green_value_z_eps.data_mut(),
        );

        let x_deriv = ((green_value_x_eps - &green_value) * (1.0 / eps)).eval();
        let y_deriv = ((green_value_y_eps - &green_value) * (1.0 / eps)).eval();
        let z_deriv = ((green_value_z_eps - &green_value) * (1.0 / eps)).eval();

        for target_index in 0..ntargets {
            assert_relative_eq!(
                green_value[[target_index, 0]],
                expected[[0, target_index]],
                epsilon = 1E-12
            );

            assert_relative_eq!(
                x_deriv[[target_index, 0]],
                expected[[1, target_index]],
                epsilon = 1E-5
            );
            assert_relative_eq!(
                y_deriv[[target_index, 0]],
                expected[[2, target_index]],
                epsilon = 1E-5
            );

            assert_relative_eq!(
                z_deriv[[target_index, 0]],
                expected[[3, target_index]],
                epsilon = 1E-5
            );
        }
    }

    #[test]
    fn test_assemble_laplace_3d() {
        let nsources = 3;
        let ntargets = 5;

        let sources = rlst::dense::rlst_rand_mat![f64, (nsources, 3)];
        let targets = rlst::dense::rlst_rand_mat![f64, (ntargets, 3)];
        let mut green_value = rlst::dense::rlst_mat![f64, (nsources, ntargets)];

        Laplace3dKernel::<f64>::default().assemble_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            green_value.data_mut(),
        );

        // The matrix needs to be transposed so that the first row corresponds to the first target,
        // second row to the second target and so on.

        let green_value = green_value.transpose().eval();

        for charge_index in 0..nsources {
            let mut charges = rlst::dense::rlst_col_vec![f64, nsources];
            let mut expected = rlst::dense::rlst_col_vec![f64, ntargets];
            charges[[charge_index, 0]] = 1.0;

            Laplace3dKernel::<f64>::default().evaluate_st(
                EvalType::Value,
                sources.data(),
                targets.data(),
                charges.data(),
                expected.data_mut(),
            );

            for target_index in 0..ntargets {
                assert_relative_eq!(
                    green_value[[target_index, charge_index]],
                    expected[[target_index, 0]],
                    epsilon = 1E-12
                );
            }
        }

        let mut green_value_deriv = rlst::dense::rlst_mat![f64, (nsources, 4 * ntargets)];

        Laplace3dKernel::<f64>::default().assemble_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            green_value_deriv.data_mut(),
        );

        // The matrix needs to be transposed so that the first row corresponds to the first target, etc.

        let green_value_deriv = green_value_deriv.transpose().eval();

        for charge_index in 0..nsources {
            let mut charges = rlst::dense::rlst_col_vec![f64, nsources];
            let mut expected = rlst::dense::rlst_mat![f64, (4, ntargets)];

            charges[[charge_index, 0]] = 1.0;

            Laplace3dKernel::<f64>::default().evaluate_st(
                EvalType::ValueDeriv,
                sources.data(),
                targets.data(),
                charges.data(),
                expected.data_mut(),
            );

            for deriv_index in 0..4 {
                for target_index in 0..ntargets {
                    assert_relative_eq!(
                        green_value_deriv[[4 * target_index + deriv_index, charge_index]],
                        expected[[deriv_index, target_index]],
                        epsilon = 1E-12
                    );
                }
            }
        }
    }

    #[test]
    fn test_compare_assemble_with_direct_computation() {
        let nsources = 3;
        let ntargets = 5;

        let sources = rlst::dense::rlst_rand_mat![f64, (nsources, 3)];
        let targets = rlst::dense::rlst_rand_mat![f64, (ntargets, 3)];
        let mut green_value_deriv = rlst::dense::rlst_mat![f64, (nsources, 4 * ntargets)];

        Laplace3dKernel::<f64>::default().assemble_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            green_value_deriv.data_mut(),
        );
    }
}
