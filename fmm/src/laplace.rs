//! Laplace kernel implementation.
use bempp_traits::{kernel::Kernel, types::Result};

pub struct LaplaceKernel {
    pub dim: usize,
    pub is_singular: bool,
    pub value_dimension: usize,
}

impl LaplaceKernel {
    fn new(dim: usize, is_singular: bool, value_dimension: usize) -> LaplaceKernel {
        LaplaceKernel {
            dim,
            is_singular,
            value_dimension,
        }
    }

    fn gradient_kernel_3D(source: &[f64], target: &[f64], c: usize) -> f64 {
        let num = source[c] - target[c];
        let invdiff: f64 = source
            .iter()
            .zip(target.iter())
            .map(|(s, t)| (s - t).powf(2.0))
            .sum::<f64>()
            .sqrt()
            * std::f64::consts::PI
            * 4.0;

        let mut tmp = invdiff.recip();
        tmp *= invdiff;
        tmp *= invdiff;
        tmp *= num;

        if tmp.is_finite() {
            tmp
        } else {
            0.
        }
    }

    fn potential_kernel_3D(&self, source: &[f64], target: &[f64]) -> f64 {
        let mut tmp = source
            .iter()
            .zip(target.iter())
            .map(|(s, t)| (s - t).powf(2.0))
            .sum::<f64>()
            .powf(0.5)
            * std::f64::consts::PI
            * 4.0;

        tmp = tmp.recip();

        if tmp.is_finite() {
            tmp
        } else {
            0.
        }
    }
}

impl Kernel for LaplaceKernel {
    type PotentialData = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn is_singular(&self) -> bool {
        self.is_singular
    }

    fn value_dimension(&self) -> usize {
        self.value_dimension
    }

    fn potential(&self, sources: &[f64], charges: &[f64], targets: &[f64], potentials: &mut [f64]) {
        for (i, j) in (0..targets.len()).step_by(self.dim()).enumerate() {
            let mut potential = 0.0;
            let target = &targets[j..(j + self.dim())];

            for (k, l) in (0..sources.len()).step_by(self.dim()).enumerate() {
                let source = &sources[l..(l + self.dim())];
                let tmp;
                if self.dim() == 3 {
                    tmp = self.potential_kernel_3D(source, target);
                } else {
                    panic!("Kernel not implemented for dimension={:?}!", self.dim())
                }

                potential += charges[k] * tmp;
            }
            potentials[i] = potential
        }
    }

    fn gram(
        &self,
        sources: &[f64],
        targets: &[f64],
    ) -> bempp_traits::types::Result<Self::PotentialData> {
        let mut result: Vec<f64> = Vec::new();

        for i in (0..targets.len()).step_by(self.dim()) {
            let target = &targets[i..(i + self.dim())];
            let mut row: Vec<f64> = Vec::new();

            for j in (0..sources.len()).step_by(self.dim()) {
                let source = &sources[j..(j + self.dim())];

                let tmp;
                if self.dim() == 3 {
                    tmp = self.potential_kernel_3D(source, target);
                } else {
                    panic!("Gram not implemented for dimension={:?}!", self.dim())
                }

                row.push(tmp);
            }
            result.append(&mut row);
        }
        Result::Ok(result)
    }

    fn scale(&self, level: u64) -> f64 {
        1. / (2f64.powf(level as f64))
    }
}

pub mod tests {

    use std::vec;

    use ndarray_linalg::assert;
    use rand::prelude::*;
    use rand::SeedableRng;

    use super::*;

    #[allow(dead_code)]
    fn points_fixture(npoints: usize, dim: usize) -> Vec<f64> {
        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0..1.0);
        let mut points = Vec::new();

        for _ in 0..npoints {
            for _ in 0..dim {
                points.push(between.sample(&mut range))
            }
        }

        points
    }

    #[test]
    #[should_panic(expected = "Kernel not implemented for dimension=2!")]
    pub fn test_potential_panics() {
        let dim = 2;
        let npoints = 100;
        let sources = points_fixture(npoints, dim);
        let targets = points_fixture(npoints, dim);
        let charges = vec![1.0; npoints];
        let mut potentials = vec![0.; npoints];

        let kernel = LaplaceKernel::new(dim, false, dim);
        kernel.potential(
            &sources[..],
            &charges[..],
            &targets[..],
            &mut potentials[..],
        );
    }

   
    #[test]
    #[should_panic(expected = "Gram not implemented for dimension=2!")]
    pub fn test_gram_panics() {
        let dim = 2;
        let npoints = 100;
        let sources = points_fixture(npoints, dim);
        let targets = points_fixture(npoints, dim);

        let kernel = LaplaceKernel::new(dim, false, dim);
        kernel.gram(&sources[..], &targets[..]).unwrap();
    }

    #[test]
    pub fn test_gram() {
        let dim = 3;
        let nsources = 100;
        let ntargets = 200;
        let sources = points_fixture(nsources, dim);
        let targets = points_fixture(ntargets, dim);

        let kernel = LaplaceKernel::new(dim, false, dim);
        let gram = kernel.gram(&sources[..], &targets[..]).unwrap();

        // Test dimension of output
        assert_eq!(gram.len(), ntargets * nsources);
    }
}
