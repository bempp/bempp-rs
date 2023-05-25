//! Implementation of the Laplace kernel
use crate::kernel::EvalType;
use rayon::prelude::*;
use std::marker::PhantomData;

use crate::kernel::{check_dimensions_assemble, check_dimensions_evaluate, Kernel, KernelType};
use bempp_traits::types::Scalar;
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

impl Kernel for Laplace3dKernel<f64> {
    type T = f64;

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
        eval_type: crate::kernel::EvalType,
        sources: &[f64],
        targets: &[f64],
        charges: &[Self::T],
        result: &mut [Self::T],
    ) {
        let thread_pool = bempp_tools::threads::create_pool(1);
        self.evaluate_mt(eval_type, sources, targets, charges, result, &thread_pool);
    }

    fn evaluate_mt(
        &self,
        eval_type: crate::kernel::EvalType,
        sources: &[f64],
        targets: &[f64],
        charges: &[Self::T],
        result: &mut [Self::T],
        thread_pool: &rayon::ThreadPool,
    ) {
        check_dimensions_evaluate(self, eval_type, sources, targets, charges, result);
        let ntargets = targets.len() % self.space_dimension();
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

    fn range_component_count(&self, eval_type: crate::kernel::EvalType) -> usize {
        laplace_component_count(eval_type)
    }
}

pub fn evaluate_laplace_one_target(
    eval_type: EvalType,
    target: &[f64],
    sources: &[f64],
    charges: &[f64],
    result: &mut [f64],
) {
    let ncharges = charges.len();
    let nsources = ncharges;
    let m_inv_4pi = 0.25 * f64::FRAC_1_PI();

    match eval_type {
        EvalType::Value => {
            let mut my_result = 0.0;
            for index in 0..sources.len() {
                let diff_norm = ((target[0] - sources[index]) * (target[0] - sources[index])
                    + (target[1] - sources[nsources + index])
                        * (target[1] - sources[nsources + index])
                    + (target[2] - sources[2 * nsources + index])
                        * (target[2] - sources[2 * nsources + index]))
                    .sqrt();
                let inv_diff_norm = {
                    if diff_norm == 0.0 {
                        0.0
                    } else {
                        1.0 / diff_norm
                    }
                };

                my_result += charges[index] * inv_diff_norm;
            }
            result[0] = m_inv_4pi * my_result;
        }
        EvalType::ValueDeriv => {
            // Cannot simply use an array my_result as this is not
            // correctly auto-vectorized.

            let mut my_result0 = 0.0;
            let mut my_result1 = 0.0;
            let mut my_result2 = 0.0;
            let mut my_result3 = 0.0;

            for index in 0..sources.len() {
                let diff0 = sources[index] - target[0];
                let diff1 = sources[nsources + index] - target[1];
                let diff2 = sources[nsources + index] - target[2];
                let diff_norm = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2).sqrt();
                let inv_diff_norm = {
                    if diff_norm == 0.0 {
                        0.0
                    } else {
                        1.0 / diff_norm
                    }
                };
                let inv_diff_norm_cubed = inv_diff_norm * inv_diff_norm * inv_diff_norm;

                my_result0 += charges[index] * inv_diff_norm;
                my_result1 += diff0 * charges[index] * inv_diff_norm_cubed;
                my_result2 += diff1 * charges[index] * inv_diff_norm_cubed;
                my_result3 += diff2 * charges[index] * inv_diff_norm_cubed;
            }

            result[0] = m_inv_4pi * my_result0;
            result[1] = m_inv_4pi * my_result1;
            result[2] = m_inv_4pi * my_result2;
            result[3] = m_inv_4pi * my_result3;
        }
    }
}

fn laplace_component_count(eval_type: crate::kernel::EvalType) -> usize {
    match eval_type {
        EvalType::Value => 1,
        EvalType::ValueDeriv => 4,
    }
}
