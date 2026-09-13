use ellalgo_rs::arr::{linspace, Arr};
use ellalgo_rs::cutting_plane::{OracleOptim, ParallelCut};
use ellalgo_rs::round_robin::RoundRobin;
use std::f64::consts::PI;

/// Dot product of one matrix row with `x`, split into four independent partial
/// sums.
///
/// A plain `.iter().zip().map().sum()` is a single dependent f64 chain that
/// LLVM cannot reassociate (no fast-math) and therefore cannot vectorize. Four
/// independent accumulators expose instruction-level parallelism and are ~2.3x
/// faster on the 32-wide rows used here.
#[inline]
fn dot_row(row: &[f64], x: &[f64]) -> f64 {
    let mut a0 = 0.0;
    let mut a1 = 0.0;
    let mut a2 = 0.0;
    let mut a3 = 0.0;
    let mut r = row.chunks_exact(4);
    let mut xc = x.chunks_exact(4);
    for (r4, x4) in r.by_ref().zip(xc.by_ref()) {
        a0 += r4[0] * x4[0];
        a1 += r4[1] * x4[1];
        a2 += r4[2] * x4[2];
        a3 += r4[3] * x4[3];
    }
    let mut s = (a0 + a1) + (a2 + a3);
    for (a, b) in r.remainder().iter().zip(xc.remainder()) {
        s += a * b;
    }
    s
}

/// Scan `count` rows of `mat` in round-robin order and return the first
/// violating cut reported by `check`, or None if none of the rows violate.
///
/// This is the shared Template-Method skeleton for the passband, stopband, and
/// non-redundant constraint scans (mirrors `scan_constraints` in
/// multiplierless-cpp and `_scan_constraints` in the Python port).
fn scan_constraints(
    mat: &Arr,
    rr: &mut RoundRobin,
    count: usize,
    x: &Arr,
    check: impl FnMut(usize, f64) -> Option<(Arr, ParallelCut)>,
) -> Option<(Arr, ParallelCut)> {
    let mut check = check;
    let cols = mat.cols();
    let md = mat.data();
    let xd = x.data();
    for _ in 0..count {
        let k = rr.advance() as usize;
        let v = dot_row(&md[k * cols..(k + 1) * cols], xd);
        if let Some(cut) = check(k, v) {
            return Some(cut);
        }
    }
    None
}

/// Filter design construct containing all parameters for lowpass filter design.
#[derive(Clone)]
pub struct FilterDesignConstruct {
    pub n: usize,
    pub ap: Arr,
    pub as_: Arr,
    pub anr: Arr,
    pub lpsq: f64,
    pub upsq: f64,
    pub spsq: f64,
}

impl FilterDesignConstruct {
    #[inline]
    pub fn new_default(n: usize) -> Self {
        Self::new(n, 0.12, 0.20, 0.125, 0.125, 15)
    }

    pub fn new(
        n: usize,
        wpass_norm: f64,
        wstop_norm: f64,
        passband_ripple: f64,
        stopband_attn: f64,
        discretization_factor: usize,
    ) -> Self {
        let wpass = wpass_norm * PI;
        let wstop = wstop_norm * PI;
        let delta = 20.0 * (1.0 + passband_ripple).log10();
        let delta2 = 20.0 * stopband_attn.log10();
        let m = discretization_factor * n;
        let w = linspace(0.0, PI, m);

        let mut a = Arr::zeros(m, n);
        for i in 0..m {
            a.set(i, 0, 1.0);
            for j in 1..n {
                a.set(i, j, 2.0 * (w[i] * j as f64).cos());
            }
        }

        let ind_p: Vec<usize> = (0..m).filter(|&i| w[i] <= wpass).collect();
        let ind_s: Vec<usize> = (0..m).filter(|&i| w[i] >= wstop).collect();

        let lp = 10.0_f64.powf(-delta / 20.0);
        let up = 10.0_f64.powf(delta / 20.0);
        let sp = 10.0_f64.powf(delta2 / 20.0);

        let ap_rows = ind_p.len();
        let mut ap = Arr::zeros(ap_rows, n);
        for (i, &p) in ind_p.iter().enumerate().take(ap_rows) {
            for j in 0..n {
                ap.set(i, j, a.get(p, j));
            }
        }

        let as_rows = ind_s.len();
        let mut as_ = Arr::zeros(as_rows, n);
        for (i, &s) in ind_s.iter().enumerate().take(as_rows) {
            for j in 0..n {
                as_.set(i, j, a.get(s, j));
            }
        }

        let p_last = ind_p[ind_p.len() - 1];
        let s_start = ind_s[0];
        let anr_len = s_start.saturating_sub(p_last + 1);
        let mut anr = Arr::zeros(anr_len, n);
        for i in 0..anr_len {
            for j in 0..n {
                anr.set(i, j, a.get(p_last + 1 + i, j));
            }
        }

        Self {
            n,
            ap,
            as_,
            anr,
            lpsq: lp * lp,
            upsq: up * up,
            spsq: sp * sp,
        }
    }
}

pub struct LowpassOracle {
    pub(crate) fdc: FilterDesignConstruct,
    rr_ap: RoundRobin,  // passband scan: [0, ap.rows())
    rr_as: RoundRobin,  // stopband scan: [0, as_.rows())
    rr_anr: RoundRobin, // non-redundant scan: [0, anr.rows())
    g_buf: Arr,         // pre-allocated gradient buffer
}

impl LowpassOracle {
    pub fn new(fdc: FilterDesignConstruct) -> Self {
        let n = fdc.n;
        let n_ap = fdc.ap.rows() as i32;
        let n_as = fdc.as_.rows() as i32;
        let n_anr = fdc.anr.rows() as i32;
        Self {
            fdc,
            rr_ap: RoundRobin::new(n_ap),
            rr_as: RoundRobin::new(n_as),
            rr_anr: RoundRobin::new(n_anr),
            g_buf: Arr::new(n),
        }
    }

    /// Fill a pre-allocated gradient buffer from a matrix row (memcpy).
    /// Returns a clone of the buffer (avoids per-allocation Vec::new overhead).
    #[inline]
    fn fill_grad(g_buf: &mut Arr, mat: &Arr, row: usize, sign: f64) -> Arr {
        let n = mat.cols();
        let start = row * n;
        let row_data = &mat.data()[start..start + n];
        // SAFETY: g_buf is sized to mat.cols() and row_data is mat.cols()
        unsafe {
            std::ptr::copy_nonoverlapping(row_data.as_ptr(), g_buf.data_mut().as_mut_ptr(), n);
        }
        let mut g = g_buf.clone();
        if sign < 0.0 {
            for v in g.data_mut() {
                *v = -*v;
            }
        }
        g
    }
}

impl OracleOptim<Arr> for LowpassOracle {
    type CutChoice = ParallelCut;

    fn assess_optim(&mut self, x: &Arr, spsq: &mut f64) -> ((Arr, ParallelCut), bool) {
        if x[0] < 0.0 {
            let mut g = Arr::new(x.len());
            g[0] = -1.0;
            return ((g, ParallelCut(-x[0], None)), false);
        }

        // Passband constraints: lp_sq <= v <= up_sq.
        let n_ap = self.fdc.ap.rows();
        if let Some((g, cut)) = scan_constraints(&self.fdc.ap, &mut self.rr_ap, n_ap, x, |k, v| {
            if v > self.fdc.upsq {
                Some((
                    Self::fill_grad(&mut self.g_buf, &self.fdc.ap, k, 1.0),
                    ParallelCut(v - self.fdc.upsq, Some(v - self.fdc.lpsq)),
                ))
            } else if v < self.fdc.lpsq {
                Some((
                    Self::fill_grad(&mut self.g_buf, &self.fdc.ap, k, -1.0),
                    ParallelCut(-v + self.fdc.lpsq, Some(-v + self.fdc.upsq)),
                ))
            } else {
                None
            }
        }) {
            return ((g, cut), false);
        }

        // Stopband constraint: 0 <= v <= spsq, tracking the maximum row.
        let n_as = self.fdc.as_.rows();
        let mut fmax = -1e100;
        let mut imax = 0;
        if let Some((g, cut)) = scan_constraints(&self.fdc.as_, &mut self.rr_as, n_as, x, |k, v| {
            if v > *spsq {
                Some((
                    Self::fill_grad(&mut self.g_buf, &self.fdc.as_, k, 1.0),
                    ParallelCut(v - *spsq, Some(v)),
                ))
            } else if v < 0.0 {
                Some((
                    Self::fill_grad(&mut self.g_buf, &self.fdc.as_, k, -1.0),
                    ParallelCut(-v, Some(-v + *spsq)),
                ))
            } else {
                if v > fmax {
                    fmax = v;
                    imax = k;
                }
                None
            }
        }) {
            return ((g, cut), false);
        }

        // Non-redundant constraint: v >= 0.
        let n_anr = self.fdc.anr.rows();
        if let Some((g, cut)) =
            scan_constraints(&self.fdc.anr, &mut self.rr_anr, n_anr, x, |k, v| {
                if v < 0.0 {
                    Some((
                        Self::fill_grad(&mut self.g_buf, &self.fdc.anr, k, -1.0),
                        ParallelCut(-v, None),
                    ))
                } else {
                    None
                }
            })
        {
            return ((g, cut), false);
        }

        *spsq = fmax;
        let g = Self::fill_grad(&mut self.g_buf, &self.fdc.as_, imax, 1.0);
        let cut = ParallelCut(0.0, Some(fmax));
        ((g, cut), true)
    }
}

pub fn create_lowpass_case(n: usize) -> (LowpassOracle, f64) {
    let fdc = FilterDesignConstruct::new_default(n);
    let spsq = fdc.spsq;
    let omega = LowpassOracle::new(fdc);
    (omega, spsq)
}
