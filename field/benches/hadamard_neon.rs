use criterion::{black_box, criterion_group, criterion_main, Criterion};

use cauchy::c64;

use rlst::dense::{rlst_mat, rlst_rand_mat, RawAccess, RawAccessMut};

use bempp_field::hadamard::aarch64::hadamard_product_simd_neon;

fn bench_hadamard_product_neon(order: usize) {
    let nsiblings = 8;
    let nconvolutions = 16;

    let n = 2 * order - 1;
    let &(m, n, o) = &(n, n, n);

    let p = m + 1;
    let q = n + 1;
    let r = o + 1;
    let size_real = p * q * (r / 2 + 1);

    let sibling_coefficients = rlst_rand_mat![c64, (nsiblings * size_real, 1)];
    let kernel_evaluations = rlst_rand_mat![c64, (nconvolutions * size_real, 1)];
    let mut result = rlst_mat![c64, (nconvolutions * nsiblings * size_real, 1)];

    hadamard_product_simd_neon(
        order,
        sibling_coefficients.data(),
        kernel_evaluations.data(),
        result.data_mut(),
    );
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("hadamard_product_simd_neon(order=9)", |b| {
        b.iter(|| bench_hadamard_product_neon(black_box(9)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
