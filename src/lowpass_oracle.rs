use ellalgo_rs::arr::{linspace, Arr};
use ellalgo_rs::cutting_plane::{OracleOptim, ParallelCut};
use std::f64::consts::PI;

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
        for i in 0..ap_rows {
            for j in 0..n {
                ap.set(i, j, a.get(ind_p[i], j));
            }
        }

        let as_rows = ind_s.len();
        let mut as_ = Arr::zeros(as_rows, n);
        for i in 0..as_rows {
            for j in 0..n {
                as_.set(i, j, a.get(ind_s[i], j));
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
    i_ap: usize,
    i_as: usize,
    i_anr: usize,
}

impl LowpassOracle {
    pub fn new(fdc: FilterDesignConstruct) -> Self {
        Self {
            fdc,
            i_ap: 0,
            i_as: 0,
            i_anr: 0,
        }
    }

    fn dot_row(&self, mat: &Arr, row: usize, x: &Arr) -> f64 {
        let mut sum = 0.0;
        for j in 0..x.len() {
            sum += mat.get(row, j) * x[j];
        }
        sum
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

        let n_ap = self.fdc.ap.rows();
        for _ in 0..n_ap {
            if self.i_ap == n_ap {
                self.i_ap = 0;
            }
            let v = self.dot_row(&self.fdc.ap, self.i_ap, x);
            if v > self.fdc.upsq {
                let mut g = Arr::new(self.fdc.ap.cols());
                for j in 0..self.fdc.ap.cols() {
                    g[j] = self.fdc.ap.get(self.i_ap, j);
                }
                let cut = ParallelCut(v - self.fdc.upsq, Some(v - self.fdc.lpsq));
                self.i_ap += 1;
                return ((g, cut), false);
            }
            if v < self.fdc.lpsq {
                let mut g = Arr::new(self.fdc.ap.cols());
                for j in 0..self.fdc.ap.cols() {
                    g[j] = -self.fdc.ap.get(self.i_ap, j);
                }
                let cut = ParallelCut(-v + self.fdc.lpsq, Some(-v + self.fdc.upsq));
                self.i_ap += 1;
                return ((g, cut), false);
            }
            self.i_ap += 1;
        }

        let n_as = self.fdc.as_.rows();
        let mut fmax = -1e100;
        let mut imax = 0;
        for _ in 0..n_as {
            if self.i_as == n_as {
                self.i_as = 0;
            }
            let v = self.dot_row(&self.fdc.as_, self.i_as, x);
            if v > *spsq {
                let mut g = Arr::new(self.fdc.as_.cols());
                for j in 0..self.fdc.as_.cols() {
                    g[j] = self.fdc.as_.get(self.i_as, j);
                }
                let cut = ParallelCut(v - *spsq, Some(v));
                self.i_as += 1;
                return ((g, cut), false);
            }
            if v < 0.0 {
                let mut g = Arr::new(self.fdc.as_.cols());
                for j in 0..self.fdc.as_.cols() {
                    g[j] = -self.fdc.as_.get(self.i_as, j);
                }
                let cut = ParallelCut(-v, Some(-v + *spsq));
                self.i_as += 1;
                return ((g, cut), false);
            }
            if v > fmax {
                fmax = v;
                imax = self.i_as;
            }
            self.i_as += 1;
        }

        let n_anr = self.fdc.anr.rows();
        for _ in 0..n_anr {
            if self.i_anr == n_anr {
                self.i_anr = 0;
            }
            let v = self.dot_row(&self.fdc.anr, self.i_anr, x);
            if v < 0.0 {
                let mut g = Arr::new(self.fdc.anr.cols());
                for j in 0..self.fdc.anr.cols() {
                    g[j] = -self.fdc.anr.get(self.i_anr, j);
                }
                let cut = ParallelCut(-v, None);
                self.i_anr += 1;
                return ((g, cut), false);
            }
            self.i_anr += 1;
        }

        *spsq = fmax;
        let mut g = Arr::new(self.fdc.as_.cols());
        for j in 0..self.fdc.as_.cols() {
            g[j] = self.fdc.as_.get(imax, j);
        }
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
