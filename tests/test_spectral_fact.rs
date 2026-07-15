use ellalgo_rs::arr::Arr;
use multiplierless_rs::{inverse_spectral_fact, spectral_fact, spectral_fact_root};

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

#[test]
fn test_inverse_spectral_fact_energy() {
    let mut h = Arr::new(7);
    h[0] = 0.76006445;
    h[1] = 0.54101887;
    h[2] = 0.42012073;
    h[3] = 0.3157191;
    h[4] = 0.10665804;
    h[5] = 0.04326203;
    h[6] = 0.01315678;

    let r = inverse_spectral_fact(&h);
    let energy: f64 = h.iter().map(|&x| x * x).sum();
    assert!(approx_eq(r[0], energy, 1e-6));
    // r[0] should be ~1.16
    assert!(r[0] > 1.0 && r[0] < 1.3);
}

#[test]
fn test_spectral_fact_root_energy_preserving() {
    let mut r = Arr::new(7);
    let ref_vals = [1.16, 0.81, 0.55, 0.32, 0.11, 0.04, 0.01];
    for (i, &v) in ref_vals.iter().enumerate() {
        r[i] = v;
    }
    let h = spectral_fact_root(&r, 1e-8);
    let energy: f64 = h.iter().map(|&x| x * x).sum();
    assert!(approx_eq(energy, r[0], 0.05));
}

#[test]
fn test_spectral_fact_fft_energy_preserving() {
    let mut r = Arr::new(7);
    let ref_vals = [1.16, 0.81, 0.55, 0.32, 0.11, 0.04, 0.01];
    for (i, &v) in ref_vals.iter().enumerate() {
        r[i] = v;
    }
    let h = spectral_fact(&r);
    let energy: f64 = h.iter().map(|&x| x * x).sum();
    eprintln!("FFT result h = {:?}", h.iter().collect::<Vec<_>>());
    eprintln!("Energy = {energy}, expected = {}", r[0]);
    assert!(approx_eq(energy, r[0], 0.05));
}
