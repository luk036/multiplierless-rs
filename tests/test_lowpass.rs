use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options};
use ellalgo_rs::ell::Ell;
use multiplierless_rs::create_lowpass_case;

fn run_lowpass(use_parallel_cut: bool) -> (bool, usize) {
    let n = 32;
    let r0 = Arr::new(n);
    let mut ellip = Ell::new_with_scalar(40.0, r0);
    ellip.no_defer_trick = !use_parallel_cut;

    let (mut omega, mut spsq) = create_lowpass_case(n);
    let options = Options::new(50000, 1e-14);

    let (r, num_iters) = cutting_plane_optim(&mut omega, &mut ellip, &mut spsq, &options);
    (r.is_some(), num_iters)
}

#[test]
fn test_lowpass_with_parallel_cut() {
    let (feasible, num_iters) = run_lowpass(true);
    assert!(feasible);
    // The exact iteration count depends on numerical precision
    // but should converge within reasonable bounds
    assert!(num_iters <= 22000, "Too many iterations: {num_iters}");
}
