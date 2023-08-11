use std::{collections::{HashSet, HashMap}, usize, sync::{Arc, RwLock}, env::args, hash::Hash};

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
// pub fn compute_transfer_vectors() -> Vec<TransferVector> {
//     let point = [0.5, 0.5, 0.5];
//     let domain = Domain {
//         origin: [0., 0., 0.],
//         diameter: [1., 1., 1.],
//     };

//     // Encode point in centre of domain
//     let key = MortonKey::from_point(&point, &domain, 3);

//     // Add neighbours, and their resp. siblings to v list.
//     let mut neighbours = key.neighbors();
//     let mut keys: Vec<MortonKey> = Vec::new();
//     keys.push(key);
//     keys.append(&mut neighbours);

//     for key in neighbours.iter() {
//         let mut siblings = key.siblings();
//         keys.append(&mut siblings);
//     }

//     // Keep only unique keys
//     let keys: Vec<&MortonKey> = keys.iter().unique().collect();

//     let mut transfer_vectors: Vec<usize> = Vec::new();
//     let mut targets: Vec<MortonKey> = Vec::new();
//     let mut sources: Vec<MortonKey> = Vec::new();

//     for key in keys.iter() {
//         // Dense v_list
//         let v_list = key
//             .parent()
//             .neighbors()
//             .iter()
//             .flat_map(|pn| pn.children())
//             .filter(|pnc| !key.is_adjacent(pnc))
//             .collect_vec();

//         // Find transfer vectors for everything in dense v list of each key
//         let tmp: Vec<usize> = v_list
//             .iter()
//             .map(|v| key.find_transfer_vector(v))
//             .collect_vec();

//         transfer_vectors.extend(&mut tmp.iter().cloned());
//         sources.extend(&mut v_list.iter().cloned());

//         let tmp_targets = vec![**key; tmp.len()];
//         targets.extend(&mut tmp_targets.iter().cloned());
//     }

//     let mut unique_transfer_vectors = Vec::new();
//     let mut unique_indices = HashSet::new();

//     for (i, vec) in transfer_vectors.iter().enumerate() {
//         if !unique_transfer_vectors.contains(vec) {
//             unique_transfer_vectors.push(*vec);
//             unique_indices.insert(i);
//         }
//     }

//     let unique_sources: Vec<MortonKey> = sources
//         .iter()
//         .enumerate()
//         .filter(|(i, _)| unique_indices.contains(i))
//         .map(|(_, x)| *x)
//         .collect_vec();

//     let unique_targets: Vec<MortonKey> = targets
//         .iter()
//         .enumerate()
//         .filter(|(i, _)| unique_indices.contains(i))
//         .map(|(_, x)| *x)
//         .collect_vec();

//     let mut result = Vec::<TransferVector>::new();

//     for ((t, s), v) in unique_targets
//         .into_iter()
//         .zip(unique_sources)
//         .zip(unique_transfer_vectors)
//     {
//         result.push(TransferVector {
//             vector: v,
//             target: t,
//             source: s,
//         })
//     }

//     result
// }

/// Algebraically defined list of unique M2L interactions, called 'transfer vectors', for 3D FMM, remove redundant
/// vectors through axial and diagonal reflections.
pub fn compute_transfer_vectors_unique() -> (
    Vec<TransferVector>,
    HashMap<usize, usize>
 ) {
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

    let mut transfer_vectors_component: Vec<[i64; 3]> = Vec::new();

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
        let tmp: Vec<[i64; 3]> = v_list
            .iter()
            .map(|v| key.find_transfer_vector_components(v))
            .collect_vec();

        transfer_vectors_component.extend(&mut tmp.iter().cloned()); 

        sources.extend(&mut v_list.iter().cloned());

        let tmp_targets = vec![**key; tmp.len()];
        targets.extend(&mut tmp_targets.iter().cloned());
    }

    // Find unique transfer vectors (316 in total hom. smooth kernels)
    let mut unique_transfer_vectors = Vec::new();
    let mut unique_transfer_vectors_component = Vec::new();
    let mut unique_indices = HashSet::new();

    let mut transfer_vectors = transfer_vectors_component.iter().map(|c| MortonKey::find_transfer_vector_from_components(c)).collect_vec();
    
    for (i, (vec, comp)) in transfer_vectors.iter().zip(transfer_vectors_component).enumerate() {
        if !unique_transfer_vectors.contains(vec) {
            unique_transfer_vectors.push(*vec);
            unique_transfer_vectors_component.push(comp);
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

    // Now remove axial/diagonal redundancies as well resulting in 16 transfer vectors in total
    let mut reflected_transfer_vectors = Vec::new();

    // Create a map between the original transfer vectors and the reduced set.
    let mut axial_diag_map = HashMap::new();
    let mut axial_diag_map_component = HashMap::new();
    let mut unique_reflected_transfer_vectors = HashSet::new();

    for t in unique_transfer_vectors_component.iter() {

        let t_refl = reflect_transfer_vector(t);
        // Get into checksum for ease of lookup
        let t_checksum = MortonKey::find_transfer_vector_from_components(t);
        let t_rot_checksum = MortonKey::find_transfer_vector_from_components(&t_refl[..]);

        axial_diag_map.insert(t_checksum, t_rot_checksum.clone());
        axial_diag_map_component.insert(t, t_refl.clone());
        unique_reflected_transfer_vectors.insert(t_rot_checksum.clone());

        reflected_transfer_vectors.push(t_refl);
    }

    // For each unique transfer vector find representative source/target pair for calculating
    // FMM matrices

    let mut result = Vec::<TransferVector>::new();

    let mut checked = HashSet::new();

    for ((((&t, &s)), &h), &c) in unique_targets
        .iter()
        .zip(unique_sources.iter())
        .zip(unique_transfer_vectors.iter())
        .zip(unique_transfer_vectors_component.iter())
    {
        
        let h_refl = axial_diag_map.get(&h).unwrap();
        let c_refl = axial_diag_map_component.get(&c).unwrap();
        
        if !checked.contains(h_refl) {
            result.push(TransferVector {
                components: [c_refl[0], c_refl[1], c_refl[2]],
                hash: *h_refl,
                target: t,
                source: s,
            });
            checked.insert(h_refl);
        } 
    }

    // Also need to store mappings between surface/conv grid multi-indices
    // for the reduced set of sources and targets I've taken that correspond
    // to unique transfer vectors.



    (result, axial_diag_map)
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

use rlst::dense::{
    base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix, traits::*, Dynamic,
};

pub type FftMatrixf64 =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

pub type FftMatrixc64 =
    Matrix<c64, BaseMatrix<c64, VectorContainer<c64>, Dynamic, Dynamic>, Dynamic, Dynamic>;


pub fn rfft3_fftw_par_vec(
    mut input: &mut FftMatrixf64,
    mut output: &mut FftMatrixc64,
    shape: &[usize],
) {
    assert!(shape.len() == 3);

    let size: usize = shape.iter().product();
    let size_d = shape.last().unwrap();
    let size_real = (size / size_d) * (size_d / 2 + 1);

    let mut plan: R2CPlan64 = R2CPlan::aligned(shape, Flag::MEASURE).unwrap();
    let it_inp = input.data_mut().par_chunks_exact_mut(size).into_par_iter();
    let it_out = output.data_mut().par_chunks_exact_mut(size_real).into_par_iter();

    it_inp.zip(it_out).for_each(|(inp, out)| {
        plan.r2c(inp, out);
    });
}


pub fn irfft3_fftw_par_vec(
    mut input: &mut FftMatrixc64,
    mut output: &mut FftMatrixf64,
    shape: &[usize]
) {
    let size: usize = shape.iter().product(); 
    let size_d = shape.last().unwrap();
    let size_real = (size / size_d) * (size_d / 2 + 1);
    let mut plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();

    let it_inp= input.data_mut().par_chunks_exact_mut(size_real).into_par_iter();
    let it_out= output.data_mut().par_chunks_exact_mut(size).into_par_iter();

    it_inp.zip(it_out).for_each(|(inp, out)| {
        plan.c2r(inp, out);
        // Normalise output
        out
            .iter_mut()
            .for_each(|value| *value *= 1.0/(size as f64));
    })
}


pub fn irfft3_fftw_par_slice(
    mut input: &mut [Complex<f64>],
    mut output: &mut [f64],
    shape: &[usize]
) {
    let size: usize = shape.iter().product(); 
    let size_d = shape.last().unwrap();
    let size_real = (size / size_d) * (size_d / 2 + 1);
    let mut plan: C2RPlan64 = C2RPlan::aligned(shape, Flag::MEASURE).unwrap();

    let it_inp = input.par_chunks_exact_mut(size_real).into_par_iter();
    let it_out = output.par_chunks_exact_mut(size).into_par_iter();

    it_inp.zip(it_out).for_each(|(inp, out)| {
        plan.c2r(inp, out);
        // Normalise output
        out
            .iter_mut()
            .for_each(|value| *value *= 1.0/(size as f64));
    })
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

pub fn argsort<T: Ord>(arr: &[T]) -> Vec<usize> {
    arr.iter()
        .enumerate()
        .sorted_by(|a, b| a.1.cmp(b.1))
        .map(|(idx, _)| idx)
        .collect()
}

/// Reflect a vector corresponding to a position on the convolution grid
///  into the reference octant based on a translation vector.
pub fn reflect_transfer_vector(t: &[i64]) -> Vec<i64> {

    // Axial reflection
    let axial = |c: &i64| {
        if *c >= 0 {
            *c
        } else {
            -*c
        }
    };

    let axial = t.iter().map(axial).collect_vec();

    // Diagonal reflection
    let idxs = argsort(&axial);

    let axial_diag = idxs
        .iter()
        .map(|&i| axial[i].clone())
        .collect_vec();

    axial_diag
}

pub fn reflect_transfer_vector_axial(t: &[i64]) -> Vec<i64> {
    // Axial reflection
    let axial = |c: &i64| {
        if *c >= 0 {
            *c
        } else {
            -*c
        }
    };

    let axial = t.iter().map(axial).collect_vec();
    axial
}

pub fn reflect_transfer_vector_diagonal(t: &[i64]) -> Vec<i64> {
       // Diagonal reflection
       let idxs = argsort(t);

       let axial_diag = idxs
           .iter()
           .map(|&i| t[i].clone())
           .collect_vec();
   
       axial_diag 
}

/// Perform an axial reflection of a surface multi index to get it into the reference octant.
pub fn axial_reflection_surface(multi_index: &[usize], transfer_vector: &[i64], order: usize) -> Vec<usize> {

    fn helper(m: usize, t: i64, order: usize) -> usize {
        if t >= 0 {
            return m
        } else {
            return order - (m-1)
        }
    }

    let res = multi_index.iter().enumerate().map(|(i, &m)| helper(m, transfer_vector[i], order)).collect_vec();

    res
}

pub fn axial_reflection_convolution(multi_index: &[usize], transfer_vector: &[i64], order: usize) -> Vec<usize> {

    fn helper(m: usize, t: i64, order: usize) -> usize {
        if t >= 0 {
            return m
        } else {
            return 2*(order-1) - (m-1)
        }
    }

    let res = multi_index.iter().enumerate().map(|(i, &m)| helper(m, transfer_vector[i], order)).collect_vec();

    res 
}

/// Perform a diagonal reflection to get surface multi-index into reference cone. At this point
/// the transfer vectors are assumed to correspond to translations already in the reference cone.
pub fn diagonal_reflection(multi_index: &[usize], transfer_vector: &[i64]) -> Vec<usize> {

    let idxs = argsort(transfer_vector);

    let res = idxs.iter().map(|&i| multi_index[i]).collect_vec();

    res
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