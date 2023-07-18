use std::{collections::HashSet, usize};

use dashmap::DashMap;
use itertools::Itertools;
use num::traits::real::Real;
use realfft::{num_complex::Complex, RealFftPlanner, num_traits::Zero};
use rustfft::{FftNum, FftPlanner};
use fftw::{plan::*, array::*, types::*};
use rayon::prelude::*;

use bempp_tools::Array3D;
use bempp_traits::arrays::Array3DAccess;
use bempp_tree::types::{domain::Domain, morton::MortonKey};

use crate::types::TransferVector;

/// Algebraically defined list of unique M2L interactions, called 'transfer vectors', for 3D FMM.
pub fn compute_transfer_vectors() -> Vec<TransferVector> {
    let point = [0.5, 0.5, 0.5];
    let domain = Domain {
        origin: [0., 0., 0.],
        diameter: [1., 1., 1.],
    };

    // Encode point in centre of domain
    let key = MortonKey::from_point(&point, &domain, 3);

    // Add neighbours, and their resp. siblings to v list.
    let mut neighbours = key.neighbors();
    let mut keys: Vec<MortonKey> = Vec::new();
    keys.push(key);
    keys.append(&mut neighbours);

    for key in neighbours.iter() {
        let mut siblings = key.siblings();
        keys.append(&mut siblings);
    }

    // Keep only unique keys
    let keys: Vec<&MortonKey> = keys.iter().unique().collect();

    let mut transfer_vectors: Vec<usize> = Vec::new();
    let mut targets: Vec<MortonKey> = Vec::new();
    let mut sources: Vec<MortonKey> = Vec::new();

    for key in keys.iter() {
        // Dense v_list
        let v_list = key
            .parent()
            .neighbors()
            .iter()
            .flat_map(|pn| pn.children())
            .filter(|pnc| !key.is_adjacent(pnc))
            .collect_vec();

        // Find transfer vectors for everything in dense v list of each key
        let tmp: Vec<usize> = v_list
            .iter()
            .map(|v| key.find_transfer_vector(v))
            .collect_vec();

        transfer_vectors.extend(&mut tmp.iter().cloned());
        sources.extend(&mut v_list.iter().cloned());

        let tmp_targets = vec![**key; tmp.len()];
        targets.extend(&mut tmp_targets.iter().cloned());
    }

    let mut unique_transfer_vectors = Vec::new();
    let mut unique_indices = HashSet::new();

    for (i, vec) in transfer_vectors.iter().enumerate() {
        if !unique_transfer_vectors.contains(vec) {
            unique_transfer_vectors.push(*vec);
            unique_indices.insert(i);
        }
    }

    let unique_sources: Vec<MortonKey> = sources
        .iter()
        .enumerate()
        .filter(|(i, _)| unique_indices.contains(i))
        .map(|(_, x)| *x)
        .collect_vec();

    let unique_targets: Vec<MortonKey> = targets
        .iter()
        .enumerate()
        .filter(|(i, _)| unique_indices.contains(i))
        .map(|(_, x)| *x)
        .collect_vec();

    let mut result = Vec::<TransferVector>::new();

    for ((t, s), v) in unique_targets
        .into_iter()
        .zip(unique_sources)
        .zip(unique_transfer_vectors)
    {
        result.push(TransferVector {
            vector: v,
            target: t,
            source: s,
        })
    }

    result
}

pub fn pad3<T>(arr: &Array3D<T>, pad_size: (usize, usize, usize), pad_index: (usize, usize, usize)) -> Array3D<T>
where
    T: Clone+FftNum
{
    let &(m, n, o) = arr.shape();

    let (x, y, z) = pad_index;
    let (p, q, r) = pad_size;


    // Check that there is enough space for pad
    assert!(x+p <= m+p && y+q <= n+q && z+r <= o+r);

    let mut padded = Array3D::new((p+m, q+n, r+o));

    for i in 0..m {
        for j in 0..n {
            for k in 0..o {
                *padded.get_mut(x+i, y+j, z+k).unwrap() = *arr.get(i, j, k).unwrap();
            }
        }
    }

    padded
}

pub fn flip3<T>(arr: &Array3D<T>) -> Array3D<T>
where
    T: Clone+FftNum 
{
    let mut flipped = Array3D::new(*arr.shape());
    
    let &(m, n, o) = arr.shape();

    for i in 0..m {
        for j in 0..n {
            for k in 0..o {
                *flipped.get_mut(i, j, k).unwrap() = *arr.get(m-i-1, n-j-1, o-k-1).unwrap();
            }
        }
    } 

    flipped
}

pub fn rfft3_fftw(mut input: &mut [f64], mut output: &mut[c64], shape: &[usize]) {

    let mut plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();

    plan.r2c(input, output);
}

pub fn rfft3_fftw_par_dm(
    mut input: &DashMap<MortonKey, Array3D<f64>>,
    mut output: &DashMap<MortonKey, Array3D<c64>>,
    shape: &[usize],
    targets: &[MortonKey]
) {
    let size: usize = shape.iter().product();
    let size_d = shape.last().unwrap();
    let size_real = (size / size_d) * (size_d / 2 + 1);
    
    let mut plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();

    targets.into_par_iter().for_each(|key| {
        plan.r2c(
            input.get_mut(key).unwrap().get_data_mut(),
            output.get_mut(key).unwrap().get_data_mut()
        );
    });
}

pub fn irfft3_fftw(mut input: &mut [c64], mut output: &mut[f64], shape: &[usize]) {
    let size: usize = shape.iter().product(); 
    let mut plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();
    plan.c2r(input, output);
    // Normalise
    output 
        .iter_mut()
        .for_each(|value| *value *= 1.0 / (size as f64));
}


pub fn rfft3<T>(input_arr: &Array3D<T>) -> Array3D<Complex<T>>
where
    T: Clone + FftNum,
{
    let &(m, n, o) = input_arr.shape();

    let m_ = m / 2 + 1;
    let mut transformed = Array3D::<Complex<T>>::new((m_, n, o));

    let mut real_planner = RealFftPlanner::<T>::new();
    let real_fft = real_planner.plan_fft_forward(m);
    let mut planner = FftPlanner::<T>::new();
    let fftn = planner.plan_fft_forward(n);
    let ffto = planner.plan_fft_forward(o);
    let mut scratch: Vec<Complex<T>> = vec![Complex::zero(); m];

    // X dimension
    for j in 0..n {
        for k in 0..o {
            // Form slices
            let mut input = Vec::new();
            for i in 0..m {
                input.push(*input_arr.get(i, j, k).unwrap())
            }

            let mut output = vec![Complex::zero(); m_];
            let _ = real_fft.process_with_scratch(&mut input, &mut output, &mut scratch);

            for i in 0..m_ {
                *transformed.get_mut(i, j, k).unwrap() = output[i];
            }
        }
    }

    // Y dimension
    for i in 0..m_ {
        for k in 0..o {
            // Form slices
            let mut data = Vec::new();
            for j in 0..n {
                data.push(*transformed.get(i, j, k).unwrap())
            }
            let _ = fftn.process_with_scratch(&mut data, &mut scratch);
            for j in 0..n {
                *transformed.get_mut(i, j, k).unwrap() = data[j];
            }
        }
    }

    // Z dimension
    for i in 0..m_ {
        for j in 0..n {
            let mut data = Vec::new();
            for k in 0..o {
                data.push(*transformed.get(i, j, k).unwrap())
            }
            let _ = ffto.process_with_scratch(&mut data, &mut scratch);
            for k in 0..o {
                *transformed.get_mut(i, j, k).unwrap() = data[k];
            }
        }
    }

    transformed
}

pub fn irfft3<T>(input_arr: &Array3D<Complex<T>>, real_dim: usize) -> Array3D<T>
where
    T: FftNum + Clone,
{
    let &(m, n, o) = input_arr.shape();

    let mut transformed = Array3D::<Complex<T>>::new((real_dim, n, o));
    let mut result = Array3D::new((real_dim, n, o));
    let mut scratch = vec![Complex::zero(); o];

    let mut real_planner = RealFftPlanner::<T>::new();
    let real_fft = real_planner.plan_fft_inverse(real_dim);
    let mut planner = FftPlanner::<T>::new();
    let fftn = planner.plan_fft_inverse(n);
    let ffto = planner.plan_fft_inverse(o);

    // Z axis
    for i in 0..m {
        for j in 0..n {
            let mut data = Vec::new();
            for k in 0..o {
                data.push(*input_arr.get(i, j, k).unwrap())
            }
            let _ = ffto.process_with_scratch(&mut data, &mut scratch);
            let norm = T::one() / T::from_usize(data.len()).unwrap();
            for k in 0..o {
                *transformed.get_mut(i, j, k).unwrap() = data[k] * norm;
            }
        }
    }

    // Y axis
    for i in 0..m {
        for k in 0..o {
            let mut data = Vec::new();
            for j in 0..n {
                data.push(*transformed.get_mut(i, j, k).unwrap());
            }
            let _ = fftn.process_with_scratch(&mut data, &mut scratch);
            let norm = T::one() / T::from_usize(data.len()).unwrap();
            for j in 0..n {
                *transformed.get_mut(i, j, k).unwrap() = data[j] * norm;
            }
        }
    }

    // X axis
    for j in 0..n {
        for k in 0..o {
            let mut input = Vec::new();
            for i in 0..m {
                input.push(*transformed.get_mut(i, j, k).unwrap());
            }
            let mut output = vec![T::zero(); real_dim];

            let _ = real_fft.process_with_scratch(&mut input, &mut output, &mut scratch);
            let norm = T::one() / T::from_usize(real_dim).unwrap();

            for i in 0..real_dim {
                *transformed.get_mut(i, j, k).unwrap() = Complex::from(output[i] * norm);
            }
        }
    }

    for i in 0..real_dim {
        for j in 0..n {
            for k in 0..o {
                *result.get_mut(i, j, k).unwrap() = T::from(transformed.get(i, j, k).unwrap().re);
            }
        }
    }

    result
}

#[cfg(test)]
mod test {

    use super::*;

    use approx_eq::assert_approx_eq;

    #[test]
    fn test_rfft3() {
        let mut input = Array3D::new((3, 3, 3));

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    *input.get_mut(i, j, k).unwrap() = (i + j * 3 + k * 3 * 3 + 1) as f64
                }
            }
        }

        let transformed = rfft3(&input);

        let result = irfft3(&transformed, 3);

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    assert_approx_eq!(*result.get(i, j, k).unwrap(), *input.get(i, j, k).unwrap());
                }
            }
        }
    }

    #[test]
    fn test_pad3() {
        let dim = 3;
        // Initialise input data
        let mut input = Array3D::new((dim, dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    *input.get_mut(i, j, k).unwrap() = (i + j * dim + k * dim * dim + 1) as f64
                }
            }
        }

        // Test padding at edge of each axis
        let pad_size = (2,3,4);
        let pad_index = (0, 0, 0);
        let padded = pad3(&input, pad_size, pad_index);

        let &(m, n, o) = padded.shape();

        // Check dimension
        assert_eq!(m, dim+pad_size.0);
        assert_eq!(n, dim+pad_size.1);
        assert_eq!(o, dim+pad_size.2);

        // Check that padding has been correctly applied
        for i in dim..m {
            for j in dim..n {
                for k in dim.. o {
                    assert_eq!(*padded.get(i, j, k).unwrap(), 0f64)
                }
            }
        }

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    assert_eq!(*padded.get(i, j, k).unwrap(), *input.get(i, j, k).unwrap())
                }
            }
        }

        // Test padding at the start of each axis
        let pad_index = (2,2,2);
      
        let padded = pad3(&input, pad_size, pad_index);

        // Check that padding has been correctly applied
        for i in 0..pad_index.0 {
            for j in 0..pad_index.1 {
                for k in 0.. pad_index.2 {
                    assert_eq!(*padded.get(i, j, k).unwrap(), 0f64)
                }
            }
        }

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    assert_eq!(*padded.get(i+pad_index.0, j+pad_index.1, k+pad_index.2).unwrap(), *input.get(i, j, k).unwrap());
                }
            }
        } 
    }
}