use crate::lowpass_oracle::LowpassOracle;
use crate::spectral_fact::{inverse_spectral_fact, spectral_fact};
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{OracleOptim, OracleOptimQ, ParallelCut};

/// Direct double → double CSD quantization (matches C++ `csd_quantize`).
fn csd_quantize_direct(mut num: f64, mut nnz: u32) -> f64 {
    if num == 0.0 || nnz == 0 {
        return 0.0;
    }
    let exp = (num.abs() * 1.5).log2().ceil() as i32 - 1;
    let mut bit_val = 2.0_f64.powi(exp);
    let mut result = 0.0;
    while nnz > 0 && num.abs() > 1e-100 {
        if (1.5 * num).abs() > bit_val {
            let sgn = if num > 0.0 { 1.0 } else { -1.0 };
            result += sgn * bit_val;
            num -= sgn * bit_val;
            nnz -= 1;
        }
        bit_val *= 0.5;
    }
    result
}

pub struct LowpassOracleQ {
    nnz: u32,
    lowpass: LowpassOracle,
    rcsd: Arr,
    num_retries: u32,
}

impl LowpassOracleQ {
    pub fn new(nnz: u32, lowpass: LowpassOracle) -> Self {
        let n = lowpass.fdc.n;
        Self {
            nnz,
            lowpass,
            rcsd: Arr::new(n),
            num_retries: 0,
        }
    }
}

impl OracleOptimQ<Arr> for LowpassOracleQ {
    type CutChoice = ParallelCut;

    fn assess_optim_q(
        &mut self,
        r: &Arr,
        spsq: &mut f64,
        retry: bool,
    ) -> ((Arr, ParallelCut), bool, Arr, bool) {
        if !retry {
            let (cut, shrunk) = self.lowpass.assess_optim(r, spsq);
            if !shrunk {
                return (cut, shrunk, r.clone(), true);
            }
            let h = spectral_fact(r);
            let n = h.len();
            let mut hcsd = Arr::new(n);
            for i in 0..n {
                hcsd[i] = csd_quantize_direct(h[i], self.nnz);
            }
            self.rcsd = inverse_spectral_fact(&hcsd);
            self.num_retries = 0;
        } else {
            self.num_retries += 1;
        }

        let ((gc, hc), shrunk) = self.lowpass.assess_optim(&self.rcsd, spsq);
        let dot_val = {
            let mut d = 0.0;
            for i in 0..gc.len() {
                d += gc[i] * (self.rcsd[i] - r[i]);
            }
            d
        };
        let hc_adj = match hc {
            ParallelCut(b0, b1) => ParallelCut(b0 + dot_val, b1.map(|b| b + dot_val)),
        };
        (
            (gc, hc_adj),
            shrunk,
            self.rcsd.clone(),
            self.num_retries < 15,
        )
    }
}
