//! Implementation of the Helmholtz kernel
use num::{self, Zero};
use std::marker::PhantomData;

use crate::helpers::{
    check_dimensions_assemble, check_dimensions_assemble_diagonal, check_dimensions_evaluate,
};
use bempp_traits::{
    kernel::Kernel,
    types::{EvalType, RlstScalar},
};
use num::traits::FloatConst;
use rayon::prelude::*;

/// Kernel for Helmholtz in 3D
#[derive(Clone)]
pub struct Helmholtz3dKernel<T: RlstScalar> {
    wavenumber: T::Real,
    _phantom_t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> Helmholtz3dKernel<T> {
    /// Create new
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            wavenumber,
            _phantom_t: PhantomData,
        }
    }
}

impl<T: RlstScalar<Complex = T> + Send + Sync> Kernel for Helmholtz3dKernel<T>
where
    // Send and sync are defined for all the standard types that implement RlstScalar (f32, f64, c32, c64)
    <T as RlstScalar>::Complex: Send + Sync,
    <T as RlstScalar>::Real: Send + Sync,
{
    type T = T;

    fn domain_component_count(&self) -> usize {
        1
    }

    fn space_dimension(&self) -> usize {
        3
    }

    fn evaluate_st(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
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

                evaluate_helmholtz_one_target(
                    eval_type,
                    &target,
                    sources,
                    charges,
                    self.wavenumber,
                    my_chunk,
                )
            });
    }

    fn evaluate_mt(
        &self,
        eval_type: bempp_traits::types::EvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
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

                evaluate_helmholtz_one_target(
                    eval_type,
                    &target,
                    sources,
                    charges,
                    self.wavenumber,
                    my_chunk,
                )
            });
    }

    fn greens_fct(
        &self,
        eval_type: EvalType,
        source: &[<Self::T as RlstScalar>::Real],
        target: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        let zero_real = <T::Real as num::Zero>::zero();
        let one_real = <T::Real as num::One>::one();
        let m_inv_4pi = num::cast::<f64, T::Real>(0.25 * f64::FRAC_1_PI()).unwrap();
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

        let kr: T::Real = diff_norm * self.wavenumber;
        match eval_type {
            EvalType::Value => {
                result[0] = T::complex(kr.cos(), kr.sin()).mul_real(inv_diff_norm * m_inv_4pi)
            }
            EvalType::ValueDeriv => {
                let inv_diff_norm_squared = inv_diff_norm * inv_diff_norm;
                let gr = T::complex(kr.cos(), kr.sin()).mul_real(inv_diff_norm * m_inv_4pi);
                let gr_diff = gr.mul_real(inv_diff_norm_squared) * T::complex(one_real, -kr);

                result[0] = gr;
                result[1] = gr_diff.mul_real(diff0);
                result[2] = gr_diff.mul_real(diff1);
                result[3] = gr_diff.mul_real(diff2);
            }
        }
    }

    fn assemble_st(
        &self,
        eval_type: bempp_traits::types::EvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
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

                assemble_helmholtz_one_target(
                    eval_type,
                    &target,
                    sources,
                    self.wavenumber,
                    my_chunk,
                )
            });
    }

    fn assemble_mt(
        &self,
        eval_type: bempp_traits::types::EvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
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

                assemble_helmholtz_one_target(
                    eval_type,
                    &target,
                    sources,
                    self.wavenumber,
                    my_chunk,
                )
            });
    }

    fn assemble_diagonal_st(
        &self,
        eval_type: bempp_traits::types::EvalType,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        check_dimensions_assemble_diagonal(self, eval_type, sources, targets, result);
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
                let source = [
                    sources[target_index],
                    sources[ntargets + target_index],
                    sources[2 * ntargets + target_index],
                ];
                self.greens_fct(eval_type, &source, &target, my_chunk)
            });
    }

    fn range_component_count(&self, eval_type: EvalType) -> usize {
        helmholtz_component_count(eval_type)
    }
}

/// Evaluate Helmholtz kernel for one target
pub fn evaluate_helmholtz_one_target<T: RlstScalar<Complex = T>>(
    eval_type: EvalType,
    target: &[<T as RlstScalar>::Real],
    sources: &[<T as RlstScalar>::Real],
    charges: &[T],
    wavenumber: T::Real,
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
            let mut my_result_real = <<T as RlstScalar>::Real as Zero>::zero();
            let mut my_result_imag = <<T as RlstScalar>::Real as Zero>::zero();
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

                let kr = wavenumber * diff_norm;

                let g_re = <T::Real as RlstScalar>::cos(kr) * inv_diff_norm;
                let g_im = <T::Real as RlstScalar>::sin(kr) * inv_diff_norm;
                let charge_re = charges[index].re();
                let charge_im = charges[index].im();

                my_result_imag += g_re * charge_im + g_im * charge_re;
                my_result_real += g_re * charge_re - g_im * charge_im;
            }
            result[0] += <T::Complex as RlstScalar>::complex(my_result_real, my_result_imag)
                .mul_real(m_inv_4pi);
        }
        EvalType::ValueDeriv => {
            // Cannot simply use an array my_result as this is not
            // correctly auto-vectorized.

            let mut my_result0_real = <T::Real as Zero>::zero();
            let mut my_result1_real = <T::Real as Zero>::zero();
            let mut my_result2_real = <T::Real as Zero>::zero();
            let mut my_result3_real = <T::Real as Zero>::zero();

            let mut my_result0_imag = <T::Real as Zero>::zero();
            let mut my_result1_imag = <T::Real as Zero>::zero();
            let mut my_result2_imag = <T::Real as Zero>::zero();
            let mut my_result3_imag = <T::Real as Zero>::zero();

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
                let inv_diff_norm_squared = inv_diff_norm * inv_diff_norm;

                let kr = wavenumber * diff_norm;
                let g_re = <T::Real as RlstScalar>::cos(kr) * inv_diff_norm * m_inv_4pi;
                let g_im = <T::Real as RlstScalar>::sin(kr) * inv_diff_norm * m_inv_4pi;

                let g_deriv_im = (g_im - g_re * kr) * inv_diff_norm_squared;
                let g_deriv_re = (g_re + g_im * kr) * inv_diff_norm_squared;

                let charge_re = charges[index].re();
                let charge_im = charges[index].im();

                my_result0_imag += g_re * charge_im + g_im * charge_re;
                my_result0_real += g_re * charge_re - g_im * charge_im;

                let times_charge_imag = g_deriv_re * charge_im + g_deriv_im * charge_re;
                let times_charge_real = g_deriv_re * charge_re - g_deriv_im * charge_im;

                my_result1_real += times_charge_real * diff0;
                my_result1_imag += times_charge_imag * diff0;

                my_result2_real += times_charge_real * diff1;
                my_result2_imag += times_charge_imag * diff1;

                my_result3_real += times_charge_real * diff2;
                my_result3_imag += times_charge_imag * diff2;
            }

            result[0] += <T::Complex as RlstScalar>::complex(my_result0_real, my_result0_imag);
            result[1] += <T::Complex as RlstScalar>::complex(my_result1_real, my_result1_imag);
            result[2] += <T::Complex as RlstScalar>::complex(my_result2_real, my_result2_imag);
            result[3] += <T::Complex as RlstScalar>::complex(my_result3_real, my_result3_imag);
        }
    }
}

/// Assemble Helmholtz kernel for one target
pub fn assemble_helmholtz_one_target<T: RlstScalar<Complex = T>>(
    eval_type: EvalType,
    target: &[<T as RlstScalar>::Real],
    sources: &[<T as RlstScalar>::Real],
    wavenumber: T::Real,
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

                let kr = wavenumber * diff_norm;

                let g_re = <T::Real as RlstScalar>::cos(kr) * inv_diff_norm * m_inv_4pi;
                let g_im = <T::Real as RlstScalar>::sin(kr) * inv_diff_norm * m_inv_4pi;

                result[index] = <T as RlstScalar>::complex(g_re, g_im);
            }
        }
        EvalType::ValueDeriv => {
            // Cannot simply use an array my_result as this is not
            // correctly auto-vectorized.

            let mut my_result0;
            let mut my_result1;
            let mut my_result2;
            let mut my_result3;

            let mut my_result1_real: T::Real;
            let mut my_result2_real: T::Real;
            let mut my_result3_real: T::Real;

            let mut my_result1_imag: T::Real;
            let mut my_result2_imag: T::Real;
            let mut my_result3_imag: T::Real;

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
                let inv_diff_norm_squared = inv_diff_norm * inv_diff_norm;

                let kr = wavenumber * diff_norm;
                let g_re = <T::Real as RlstScalar>::cos(kr) * inv_diff_norm * m_inv_4pi;
                let g_im = <T::Real as RlstScalar>::sin(kr) * inv_diff_norm * m_inv_4pi;

                let g_deriv_im = (g_im - g_re * kr) * inv_diff_norm_squared;
                let g_deriv_re = (g_re + g_im * kr) * inv_diff_norm_squared;

                my_result1_real = g_deriv_re * diff0;
                my_result1_imag = g_deriv_im * diff0;

                my_result2_real = g_deriv_re * diff1;
                my_result2_imag = g_deriv_im * diff1;

                my_result3_real = g_deriv_re * diff2;
                my_result3_imag = g_deriv_im * diff2;

                my_result0 = <T as RlstScalar>::complex(g_re, g_im);
                my_result1 = <T as RlstScalar>::complex(my_result1_real, my_result1_imag);
                my_result2 = <T as RlstScalar>::complex(my_result2_real, my_result2_imag);
                my_result3 = <T as RlstScalar>::complex(my_result3_real, my_result3_imag);

                my_res0[index] = my_result0;
                my_res1[index] = my_result1;
                my_res2[index] = my_result2;
                my_res3[index] = my_result3;
            }
        }
    }
}

fn helmholtz_component_count(eval_type: EvalType) -> usize {
    match eval_type {
        EvalType::Value => 1,
        EvalType::ValueDeriv => 4,
    }
}

// pub fn simd_wrapper_evaluate(
//     eval_type: EvalType,
//     target: &[f32],
//     sources: &[f32],
//     charges: &[rlst_dense::types::c32],
//     wavenumber: f32,
//     result: &mut [rlst_dense::types::c32],
// ) {
//     evaluate_helmholtz_one_target::<rlst_dense::types::c32>(
//         eval_type, target, sources, charges, wavenumber, result,
//     )
// }

// pub fn simd_wrapper_assemble(
//     eval_type: EvalType,
//     target: &[f32],
//     sources: &[f32],
//     result: &mut [f32],
// ) {
//     assemble_helmholtz_one_target(eval_type, target, sources, result);
// }

#[cfg(test)]
mod test {

    use super::*;
    use approx::assert_relative_eq;
    use bempp_traits::types::RlstScalar;
    use rlst_dense::{
        array::Array,
        base_array::BaseArray,
        data_container::VectorContainer,
        rlst_dynamic_array1, rlst_dynamic_array2,
        traits::{RandomAccessByRef, RandomAccessMut, RawAccess, RawAccessMut, Shape},
    };

    use rlst_dense::types::c64;

    fn copy(
        m_in: &Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    ) -> Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> {
        let mut m = rlst_dynamic_array2!(f64, m_in.shape());
        for i in 0..m_in.shape()[0] {
            for j in 0..m_in.shape()[1] {
                *m.get_mut([i, j]).unwrap() = *m_in.get([i, j]).unwrap();
            }
        }
        m
    }

    #[test]
    fn test_helmholtz_3d() {
        let eps = 1E-8;

        let wavenumber: f64 = 1.5;

        let nsources = 5;
        let ntargets = 3;

        let mut sources = rlst_dynamic_array2!(f64, [nsources, 3]);
        let mut targets = rlst_dynamic_array2!(f64, [ntargets, 3]);
        let mut charges = rlst_dynamic_array1!(c64, [nsources]);

        sources.fill_from_seed_equally_distributed(0);
        targets.fill_from_seed_equally_distributed(1);
        charges.fill_from_seed_equally_distributed(2);

        let mut green_value = rlst_dynamic_array1!(c64, [ntargets]);

        Helmholtz3dKernel::<c64>::new(wavenumber).evaluate_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            charges.data(),
            green_value.data_mut(),
        );

        for target_index in 0..ntargets {
            let mut expected = c64::default();
            for source_index in 0..nsources {
                let dist = ((targets[[target_index, 0]] - sources[[source_index, 0]]).square()
                    + (targets[[target_index, 1]] - sources[[source_index, 1]]).square()
                    + (targets[[target_index, 2]] - sources[[source_index, 2]]).square())
                .sqrt();

                expected += charges[[source_index]]
                    * c64::exp(c64::complex(0.0, wavenumber * dist))
                    * 0.25
                    * f64::FRAC_1_PI()
                    / dist;
            }

            assert_relative_eq!(green_value[[target_index]], expected, epsilon = 1E-12);
        }

        let mut targets_x_eps = copy(&targets);
        let mut targets_y_eps = copy(&targets);
        let mut targets_z_eps = copy(&targets);

        for index in 0..ntargets {
            targets_x_eps[[index, 0]] += eps;
            targets_y_eps[[index, 1]] += eps;
            targets_z_eps[[index, 2]] += eps;
        }

        let mut expected = rlst_dynamic_array2!(c64, [4, ntargets]);

        Helmholtz3dKernel::<c64>::new(wavenumber).evaluate_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            charges.data(),
            expected.data_mut(),
        );

        let mut green_value_x_eps = rlst_dynamic_array1![c64, [ntargets]];

        Helmholtz3dKernel::<c64>::new(wavenumber).evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_x_eps.data(),
            charges.data(),
            green_value_x_eps.data_mut(),
        );

        let mut green_value_y_eps = rlst_dynamic_array1![c64, [ntargets]];

        Helmholtz3dKernel::<c64>::new(wavenumber).evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_y_eps.data(),
            charges.data(),
            green_value_y_eps.data_mut(),
        );
        let mut green_value_z_eps = rlst_dynamic_array1![c64, [ntargets]];

        Helmholtz3dKernel::<c64>::new(wavenumber).evaluate_st(
            EvalType::Value,
            sources.data(),
            targets_z_eps.data(),
            charges.data(),
            green_value_z_eps.data_mut(),
        );

        let mut x_deriv = rlst_dynamic_array1![c64, [ntargets]];
        let mut y_deriv = rlst_dynamic_array1![c64, [ntargets]];
        let mut z_deriv = rlst_dynamic_array1![c64, [ntargets]];

        x_deriv.fill_from(
            (green_value_x_eps.view() - green_value.view()).scalar_mul(c64::from_real(1.0 / eps)),
        );

        y_deriv.fill_from(
            (green_value_y_eps.view() - green_value.view()).scalar_mul(c64::from_real(1.0 / eps)),
        );
        z_deriv.fill_from(
            (green_value_z_eps.view() - green_value.view()).scalar_mul(c64::from_real(1.0 / eps)),
        );

        for target_index in 0..ntargets {
            assert_relative_eq!(
                green_value[[target_index]],
                expected[[0, target_index]],
                epsilon = 1E-12
            );

            assert_relative_eq!(
                x_deriv[[target_index]],
                expected[[1, target_index]],
                epsilon = 1E-5
            );
            assert_relative_eq!(
                y_deriv[[target_index]],
                expected[[2, target_index]],
                epsilon = 1E-5
            );

            assert_relative_eq!(
                z_deriv[[target_index]],
                expected[[3, target_index]],
                epsilon = 1E-5
            );
        }
    }

    #[test]
    fn test_assemble_helmholtz_3d() {
        let nsources = 3;
        let ntargets = 5;
        let wavenumber: f64 = 1.5;

        let mut sources = rlst_dynamic_array2!(f64, [nsources, 3]);
        let mut targets = rlst_dynamic_array2!(f64, [ntargets, 3]);

        sources.fill_from_seed_equally_distributed(1);
        targets.fill_from_seed_equally_distributed(2);

        let mut green_value_t = rlst_dynamic_array2!(c64, [nsources, ntargets]);

        Helmholtz3dKernel::<c64>::new(wavenumber).assemble_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            green_value_t.data_mut(),
        );

        // The matrix needs to be transposed so that the first row corresponds to the first target,
        // second row to the second target and so on.

        let mut green_value = rlst_dynamic_array2!(c64, [ntargets, nsources]);
        green_value.fill_from(green_value_t.transpose());

        for charge_index in 0..nsources {
            let mut charges = rlst_dynamic_array1![c64, [nsources]];
            let mut expected = rlst_dynamic_array1![c64, [ntargets]];
            charges[[charge_index]] = c64::complex(1.0, 0.0);

            Helmholtz3dKernel::<c64>::new(wavenumber).evaluate_st(
                EvalType::Value,
                sources.data(),
                targets.data(),
                charges.data(),
                expected.data_mut(),
            );

            for target_index in 0..ntargets {
                assert_relative_eq!(
                    green_value[[target_index, charge_index]],
                    expected[[target_index]],
                    epsilon = 1E-12
                );
            }
        }

        let mut green_value_deriv_t = rlst_dynamic_array2!(c64, [nsources, 4 * ntargets]);

        Helmholtz3dKernel::<c64>::new(wavenumber).assemble_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            green_value_deriv_t.data_mut(),
        );

        // The matrix needs to be transposed so that the first row corresponds to the first target, etc.

        let mut green_value_deriv = rlst_dynamic_array2!(c64, [4 * ntargets, nsources]);
        green_value_deriv.fill_from(green_value_deriv_t.transpose());

        for charge_index in 0..nsources {
            let mut charges = rlst_dynamic_array1![c64, [nsources]];
            let mut expected = rlst_dynamic_array2!(c64, [4, ntargets]);

            charges[[charge_index]] = c64::complex(1.0, 0.0);

            Helmholtz3dKernel::<c64>::new(wavenumber).evaluate_st(
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
    fn test_assemble_diag_helmholtz_3d() {
        let nsources = 5;
        let ntargets = 5;
        let wavenumber: f64 = 1.5;

        let mut sources = rlst_dynamic_array2!(f64, [nsources, 3]);
        let mut targets = rlst_dynamic_array2!(f64, [ntargets, 3]);

        sources.fill_from_seed_equally_distributed(1);
        targets.fill_from_seed_equally_distributed(2);

        let mut green_value_diag = rlst_dynamic_array1!(c64, [ntargets]);
        let mut green_value_diag_deriv = rlst_dynamic_array2!(c64, [4, ntargets]);

        Helmholtz3dKernel::<c64>::new(wavenumber).assemble_diagonal_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            green_value_diag.data_mut(),
        );
        Helmholtz3dKernel::<c64>::new(wavenumber).assemble_diagonal_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            green_value_diag_deriv.data_mut(),
        );

        let mut green_value_t = rlst_dynamic_array2!(c64, [nsources, ntargets]);

        Helmholtz3dKernel::<c64>::new(wavenumber).assemble_st(
            EvalType::Value,
            sources.data(),
            targets.data(),
            green_value_t.data_mut(),
        );

        // The matrix needs to be transposed so that the first row corresponds to the first target,
        // second row to the second target and so on.

        let mut green_value = rlst_dynamic_array2!(c64, [ntargets, nsources]);
        green_value.fill_from(green_value_t.transpose());

        let mut green_value_deriv_t = rlst_dynamic_array2!(c64, [nsources, 4 * ntargets]);

        Helmholtz3dKernel::<c64>::new(wavenumber).assemble_st(
            EvalType::ValueDeriv,
            sources.data(),
            targets.data(),
            green_value_deriv_t.data_mut(),
        );

        // The matrix needs to be transposed so that the first row corresponds to the first target, etc.

        let mut green_value_deriv = rlst_dynamic_array2!(c64, [4 * ntargets, nsources]);
        green_value_deriv.fill_from(green_value_deriv_t.transpose());

        for index in 0..nsources {
            assert_relative_eq!(
                green_value[[index, index]],
                green_value_diag[[index]],
                epsilon = 1E-12
            );

            assert_relative_eq!(
                green_value_deriv[[4 * index, index]],
                green_value_diag_deriv[[0, index]],
                epsilon = 1E-12,
            );

            assert_relative_eq!(
                green_value_deriv[[4 * index + 1, index]],
                green_value_diag_deriv[[1, index]],
                epsilon = 1E-12,
            );

            assert_relative_eq!(
                green_value_deriv[[4 * index + 2, index]],
                green_value_diag_deriv[[2, index]],
                epsilon = 1E-12,
            );

            assert_relative_eq!(
                green_value_deriv[[4 * index + 3, index]],
                green_value_diag_deriv[[3, index]],
                epsilon = 1E-12,
            );
        }
    }
}
