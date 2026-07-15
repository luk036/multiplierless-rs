use ellalgo_rs::arr::Arr;
use ginger::aberth::{aberth_autocorr, initial_aberth_autocorr, poly_from_roots};
use ginger::rootfinding::Options as GingerOptions;
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::TAU;


/// Spectral factorization via FFT (Kolmogorov 1939).
pub fn spectral_fact_fft(r: &Arr) -> Arr {
    let n = r.len();
    let mult_factor = 100;
    let m = mult_factor * n;

    // Build A matrix: A[i,j] = 1 for j=0, else 2·cos(ω_i·j)
    // Use DFT frequency grid: ω_k = 2π·k/m for k = 0…m-1 (endpoint=False)
    let step = TAU / m as f64;
    let mut a = Arr::zeros(m, n);
    for i in 0..m {
        let wi = i as f64 * step;
        a.set(i, 0, 1.0);
        for j in 1..n {
            a.set(i, j, 2.0 * (wi * j as f64).cos());
        }
    }

    // R = A @ r  (matrix-vector multiply)
    let mut r_vals = Arr::new(m);
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a.get(i, j) * r[j];
        }
        r_vals[i] = sum;
    }

    // Clip near-zero / negative values
    let min_val = r_vals.iter().copied().fold(f64::INFINITY, f64::min);
    if min_val <= 0.0 {
        for i in 0..m {
            if r_vals[i] <= 0.0 {
                r_vals[i] = 1e-10;
            }
        }
    }

    // α = 0.5 · ln(|R|)
    let mut alpha = Arr::new(m);
    for i in 0..m {
        alpha[i] = 0.5 * r_vals[i].abs().ln();
    }

    // Hilbert transform via FFT
    let mut planner = FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(m);
    let fft_inv = planner.plan_fft_inverse(m);

    let mut alphatmp: Vec<Complex<f64>> = (0..m).map(|i| Complex::new(alpha[i], 0.0)).collect();
    fft_fwd.process(&mut alphatmp);

    // Flip negative frequencies, zero DC and Nyquist
    let half = m / 2;
    for i in half..m {
        alphatmp[i] = -alphatmp[i];
    }
    alphatmp[0] = Complex::new(0.0, 0.0);
    alphatmp[half] = Complex::new(0.0, 0.0);

    // Multiply by j (rotate by 90°): (a + jb) * j = -b + ja
    for i in 0..m {
        alphatmp[i] = Complex::new(-alphatmp[i].im, alphatmp[i].re);
    }
    fft_inv.process(&mut alphatmp);

    let inv_m = 1.0 / m as f64;
    let mut phi = Arr::new(m);
    for i in 0..m {
        phi[i] = alphatmp[i].re * inv_m;
    }

    // Downsample by mult_factor
    let out_len = n; // m / mult_factor = n
    let mut alpha1 = Arr::new(out_len);
    let mut phi1 = Arr::new(out_len);
    for i in 0..out_len {
        alpha1[i] = alpha[i * mult_factor];
        phi1[i] = phi[i * mult_factor];
    }

    // exp(α₁ + j·φ₁)
    let mut complex_out: Vec<Complex<f64>> = (0..out_len)
        .map(|i| {
            let mag = alpha1[i].exp();
            Complex::new(mag * phi1[i].cos(), mag * phi1[i].sin())
        })
        .collect();

    // IFFT → impulse response
    let fft_inv2 = planner.plan_fft_inverse(out_len);
    fft_inv2.process(&mut complex_out);

    let inv_n = 1.0 / out_len as f64;
    let mut h = Arr::new(out_len);
    for i in 0..out_len {
        h[i] = complex_out[i].re * inv_n;
    }
    h
}

/// Spectral factorization via Aberth-Ehrlich root-finding.
pub fn spectral_fact_root(r: &Arr, tolerance: f64) -> Arr {
    let n = r.len();
    let deg = 2 * n - 2;

    let mut coeffs = vec![0.0; deg + 1];
    coeffs[0] = r[n - 1];
    for i in 0..n - 1 {
        coeffs[i + 1] = 2.0 * r[n - 2 - i];
    }
    for i in 0..n - 2 {
        coeffs[deg - i - 1] = 2.0 * r[n - 2 - i];
    }
    coeffs[n - 1] = 2.0 * r[0];
    coeffs[deg] = r[n - 1];
    coeffs.reverse();

    let mut zs = initial_aberth_autocorr(&coeffs);
    let opts = GingerOptions {
        tolerance,
        max_iters: 500,
        tol_ind: 1e-15,
    };
    aberth_autocorr(&coeffs, &mut zs, &opts);

    let inside: Vec<Complex<f64>> = zs
        .iter()
        .map(|z| {
            if z.norm() < 1.0 {
                *z
            } else {
                Complex::new(1.0, 0.0) / z
            }
        })
        .collect();

    let hc = poly_from_roots(&inside);
    let energy_h: f64 = hc.iter().map(|c| c * c).sum();
    let norm = (r[0] / energy_h).sqrt();

    let mut h = Arr::new(n);
    for i in 0..n.min(hc.len()) {
        h[i] = hc[i] * norm;
    }
    h
}

/// Spectral factorization (convenience, delegates to FFT variant).
pub fn spectral_fact(r: &Arr) -> Arr {
    spectral_fact_fft(r)
}

/// Inverse spectral factorization — autocorrelation from impulse response.
pub fn inverse_spectral_fact(h: &Arr) -> Arr {
    let n = h.len();
    let mut r = Arr::new(n);
    for t in 0..n {
        let mut sum = 0.0;
        for i in 0..(n - t) {
            sum += h[i + t] * h[i];
        }
        r[t] = sum;
    }
    r
}
