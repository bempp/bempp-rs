use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;

use rlst::{
    common::traits::ColumnMajorIterator,
    dense::{rlst_rand_mat, RawAccess},
};

use bempp_field::types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm};
use bempp_kernel::laplace_3d::{evaluate_laplace_one_target, Laplace3dKernel};
use bempp_traits::{fmm::FmmLoop, tree::Tree};
use bempp_tree::{implementations::helpers::points_fixture, types::single_node::SingleNodeTree};

use bempp_fmm::{
    charge::build_charge_dict,
    types::{FmmData, KiFmm},
};

fn fft_fmm(npoints: usize, order: usize, depth: u64) {
    let points = points_fixture(npoints, None, None);
    let global_idxs = (0..npoints).collect_vec();
    let charges = vec![1.0; npoints];

    let alpha_inner = 1.05;
    let alpha_outer = 2.9;
    let adaptive = false;
    let ncrit = 150;
    let kernel = Laplace3dKernel::<f64>::default();

    let tree = SingleNodeTree::new(
        points.data(),
        adaptive,
        Some(ncrit),
        Some(depth),
        &global_idxs[..],
    );

    let m2l_data_fft = FftFieldTranslationKiFmm::new(
        kernel.clone(),
        order,
        tree.get_domain().clone(),
        alpha_inner,
    );

    let fmm = KiFmm::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data_fft);

    // Form charge dict, matching charges with their associated global indices
    let mut charge_dict = build_charge_dict(&global_idxs[..], &charges[..]);

    let datatree = FmmData::new(fmm, &charge_dict);

    let times = datatree.run(Some(true));
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fft laplace order=9 depth=5 npoints=1e6", |b| {
        b.iter(|| fft_fmm(1000000, 9, 5))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(3);
    targets = criterion_benchmark
}

criterion_main!(benches);
