use std::env;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use rand::prelude::*;
use rand::SeedableRng;

use bempp_fmm::{
    charge::Charges,
    fmm::{FmmData, KiFmm},
    laplace::LaplaceKernel,
};
use bempp_traits::{fmm::FmmLoop, tree::Tree};
use bempp_tree::types::{
    morton::MortonKey,
    point::{Point, PointType},
    single_node::SingleNodeTree,
};

fn points_fixture(npoints: usize) -> Vec<Point> {
    let mut range = StdRng::seed_from_u64(0);
    let between = rand::distributions::Uniform::from(0.0..1.0);
    let mut points: Vec<[PointType; 3]> = Vec::new();

    for _ in 0..npoints {
        points.push([
            between.sample(&mut range),
            between.sample(&mut range),
            between.sample(&mut range),
        ])
    }

    let points = points
        .iter()
        .enumerate()
        .map(|(i, p)| Point {
            coordinate: *p,
            global_idx: i,
            base_key: MortonKey::default(),
            encoded_key: MortonKey::default(),
        })
        .collect_vec();
    points
}

fn upward_pass(n: usize) {
    let points = points_fixture(n);
    let depth = 3;
    let n_crit = 150;

    let order = 7;
    let alpha_inner = 1.05;
    let alpha_outer = 1.95;
    let adaptive = true;

    let kernel = LaplaceKernel::new(3, false, 3);

    let tree = SingleNodeTree::new(&points, adaptive, Some(n_crit), Some(depth));

    let fmm = KiFmm::new(order, alpha_inner, alpha_outer, kernel, tree);

    let charges = Charges::new();

    let datatree = FmmData::new(fmm, charges);

    datatree.upward_pass();
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("FMM");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("Uniform 1000000", |b| {
        b.iter(|| upward_pass(black_box(1000000)))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
