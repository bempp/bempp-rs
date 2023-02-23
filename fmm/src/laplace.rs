/// Laplace kernel
use solvers_traits::{kernel::Kernel, types::Result};

pub struct LaplaceKernel {
    pub dim: usize,
    pub is_singular: bool,
    pub value_dimension: usize,
}

impl LaplaceKernel {
    fn potential_kernel(source: &[f64; 3], target: &[f64; 3]) -> f64 {
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

    fn gradient_kernel(source: &[f64; 3], target: &[f64; 3], c: usize) -> f64 {
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
}

impl Kernel for LaplaceKernel {
    type PotentialData = Vec<f64>;
    type GradientData = Vec<[f64; 3]>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn is_singular(&self) -> bool {
        self.is_singular
    }

    fn value_dimension(&self) -> usize {
        self.value_dimension
    }

    fn potential(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
        potentials: &mut [f64],
    ) {
        // TODO: Implement multithreaded
        for (i, target) in targets.iter().enumerate() {
            let mut potential = 0.0;

            for (source, charge) in sources.iter().zip(charges) {
                potential += charge * LaplaceKernel::potential_kernel(source, target);
            }
            potentials[i] = potential
        }
    }

    fn gradient(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
        gradients: &mut [[f64; 3]],
    ) {
        // TODO: Implement multithreaded
        for (i, target) in targets.iter().enumerate() {
            for (source, charge) in sources.iter().zip(charges) {
                gradients[i][0] -= charge * LaplaceKernel::gradient_kernel(source, target, 0);
                gradients[i][1] -= charge * LaplaceKernel::gradient_kernel(source, target, 1);
                gradients[i][2] -= charge * LaplaceKernel::gradient_kernel(source, target, 2);
            }
        }
    }

    fn gram(
        &self,
        sources: &[[f64; 3]],
        targets: &[[f64; 3]],
    ) -> solvers_traits::types::Result<Self::PotentialData> {
        let mut result: Vec<f64> = Vec::new();

        for target in targets.iter() {
            let mut row: Vec<f64> = Vec::new();
            for source in sources.iter() {
                row.push(LaplaceKernel::potential_kernel(source, target));
            }
            result.append(&mut row);
        }
        Result::Ok(result)
    }

    fn scale(&self, level: u64) -> f64 {
        1. / (2f64.powf(level as f64))
    }
}
