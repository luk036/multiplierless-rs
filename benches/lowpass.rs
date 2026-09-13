use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim};
use ellalgo_rs::ell::Ell;
use multiplierless_rs::{create_lowpass_case, FilterDesignConstruct, LowpassOracle};

const N: usize = 32;

fn converged_r() -> Arr {
    let r0 = Arr::new(N);
    let mut ellip = Ell::new_with_scalar(40.0, r0);
    let (mut omega, mut spsq) = create_lowpass_case(N);
    let options = Options::new(50000, 1e-14);
    cutting_plane_optim(&mut omega, &mut ellip, &mut spsq, &options)
        .0
        .expect("converged")
}

fn bench_assess_optim(c: &mut Criterion) {
    let x = converged_r();
    let (mut omega, _) = create_lowpass_case(N);
    c.bench_function("assess_optim (feasible)", |b| {
        b.iter(|| {
            let mut spsq = 0.01;
            let _ = omega.assess_optim(black_box(&x), &mut spsq);
        });
    });
}

fn bench_full(c: &mut Criterion) {
    let fdc = FilterDesignConstruct::new_default(N);
    let spsq0 = fdc.spsq;
    c.bench_function("cutting_plane_optim N=32", |b| {
        b.iter(|| {
            let mut omega = LowpassOracle::new(fdc.clone());
            let mut spsq = spsq0;
            let r0 = Arr::new(N);
            let mut ellip = Ell::new_with_scalar(40.0, r0);
            let options = Options::new(50000, 1e-14);
            let _ = cutting_plane_optim(&mut omega, &mut ellip, &mut spsq, &options);
        });
    });
}

criterion_group!(benches, bench_assess_optim, bench_full);
criterion_main!(benches);
