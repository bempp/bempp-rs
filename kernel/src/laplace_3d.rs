//! Implementation of the Laplace kernel
use bempp_traits::
    {
        kernel::{Kernel, EvalType, KernelType},
        types::{Scalar}, fmm::Fmm
    };
use num;
use std::marker::PhantomData;

use crate::helpers::check_dimensions_evaluate;
use num::traits::FloatConst;

pub struct Laplace3dKernel<T: Scalar> {
    kernel_type: KernelType,
    _phantom_t: std::marker::PhantomData<T>,
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
        let thread_pool = bempp_tools::threads::create_pool(1);
        self.evaluate_mt(eval_type, sources, targets, charges, result, &thread_pool);
    }

    fn evaluate_mt(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as Scalar>::Real],
        targets: &[<Self::T as Scalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
        thread_pool: &rayon::ThreadPool,
    ) {
        check_dimensions_evaluate(self, eval_type, sources, targets, charges, result);
        let ntargets = targets.len() / self.space_dimension();
        let range_dim = self.range_component_count(eval_type);

        thread_pool.install(|| {
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
        })
    }

    fn range_component_count(&self, eval_type: EvalType) -> usize {
        laplace_component_count(eval_type)
    }

    fn gram(&self, eval_type: EvalType, sources: &[<Self::T as Scalar>::Real], targets: &[<Self::T as Scalar>::Real], result: &mut [Self::T]) {

        let ntargets = targets.len() / self.space_dimension();
        let nsources= sources.len() / self.space_dimension();
        for i in 0..ntargets {
            let target = [
                targets[i],
                targets[ntargets + i],
                targets[2 * ntargets + i],
            ];
            for j in 0..nsources {
                let source = [
                    sources[j],
                    sources[nsources + j],
                    sources[2 * nsources + j],
                ];
                let idx = i+ntargets*j;

                evaluate_laplace_one_target_one_source::<T>(eval_type, &target, &source, &mut result[idx..idx+1]);
            }
        }
    }

    fn scale(&self, level: u64) -> f64 {
        1. / (2f64.powf(level as f64))
    }
}

pub fn evaluate_laplace_one_target_one_source<T: Scalar> (
    eval_type: EvalType,
    target: &[<T as Scalar>::Real],
    source: &[<T as Scalar>::Real],
    result: &mut [T],
) {

    let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();
    let zero_real = <T::Real as num::Zero>::zero();
    let one_real = <T::Real as num::One>::one();

    match eval_type {
        EvalType::Value => {
            let mut my_result = T::zero();
            let diff_norm = ((target[0] - source[0]) * (target[0] - source[0])
            + (target[1] - source[1])
                * (target[1] - source[1])
            + (target[2] - source[2])
                * (target[2] - source[2]))
            .sqrt();
            let inv_diff_norm = {
                if diff_norm == zero_real {
                    zero_real
                } else {
                    one_real / diff_norm
                }
            };
            
            my_result += T::one().mul_real(inv_diff_norm);
            result[0] = my_result.mul_real(m_inv_4pi)
        }
        EvalType::ValueDeriv => {
            // Cannot simply use an array my_result as this is not
            // correctly auto-vectorized.

            let mut my_result0 = T::zero();
            let mut my_result1 = T::zero();
            let mut my_result2 = T::zero();
            let mut my_result3 = T::zero();

            let diff0 = source[0] - target[0];
            let diff1 = source[1] - target[1];
            let diff2 = source[2] - target[2];
            let diff_norm = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2).sqrt();
            let inv_diff_norm = {
                if diff_norm == zero_real {
                    zero_real
                } else {
                    one_real / diff_norm
                }
            };
            let inv_diff_norm_cubed = inv_diff_norm * inv_diff_norm * inv_diff_norm;


            result[0] = my_result0.mul_real(m_inv_4pi);
            result[1] = my_result1.mul_real(m_inv_4pi);
            result[2] = my_result2.mul_real(m_inv_4pi);
            result[3] = my_result3.mul_real(m_inv_4pi);
        }
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

    match eval_type {
        EvalType::Value => {
            let mut my_result = T::zero();
            for index in 0..nsources {
                let diff_norm = ((target[0] - sources[index]) * (target[0] - sources[index])
                    + (target[1] - sources[nsources + index])
                        * (target[1] - sources[nsources + index])
                    + (target[2] - sources[2 * nsources + index])
                        * (target[2] - sources[2 * nsources + index]))
                    .sqrt();
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
                let diff0 = sources[index] - target[0];
                let diff1 = sources[nsources + index] - target[1];
                let diff2 = sources[2 * nsources + index] - target[2];
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

fn laplace_component_count(eval_type: EvalType) -> usize {
    match eval_type {
        EvalType::Value => 1,
        EvalType::ValueDeriv => 4,
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use approx::assert_relative_eq;
    use bempp_traits::types::Scalar;
    use rlst;
    use rlst::common::tools::PrettyPrint;
    use rlst::common::traits::{Copy, Eval};
    use rlst::dense::{traits::*, rlst_pointer_mat};

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

        green_value.pretty_print();
    }

    #[test]
    fn test_gram() {
        let eps = 1E-12;

        let nsources = 5;
        let ntargets = 3;

        let sources = rlst::dense::rlst_rand_mat![f64, (nsources, 3)];
        let targets = rlst::dense::rlst_rand_mat![f64, (ntargets, 3)];

        let mut gram = rlst::dense::rlst_rand_mat![f64, (ntargets, nsources)];
        
        let kernel = Laplace3dKernel::<f64>::default();
        
        kernel.gram(
            EvalType::Value,
            sources.data(),
            targets.data(),
            gram.data_mut(),
        );

        for i in 0..ntargets {
            let target = [
                targets.data()[i],
                targets.data()[ntargets+i],
                targets.data()[ntargets * 2 + i]
            ];

            for j in 0..nsources {
                let source = [
                    sources.data()[j],
                    sources.data()[nsources + j],
                    sources.data()[nsources * 2 + j]
                ];

                let result = gram[[i, j]];
                let mut expected = vec![0f64];
                evaluate_laplace_one_target_one_source::<f64>(EvalType::Value, &target, &source, &mut expected);
                assert_relative_eq!(
                    expected[0],
                    result,
                    epsilon = eps
                );

            }
        }
    }
}
