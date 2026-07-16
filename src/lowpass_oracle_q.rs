use crate::lowpass_oracle::LowpassOracle;
use crate::spectral_fact::{inverse_spectral_fact, spectral_fact};
use csd::to_csdnnz;
use csd::to_decimal;
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{OracleOptim, OracleOptimQ, ParallelCut};

fn csd_quantize(num: f64, nnz: u32) -> f64 {
    if num == 0.0 {
        return 0.0;
    }
    let csd_str = to_csdnnz(num, nnz);
    to_decimal(&csd_str)
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
                hcsd[i] = csd_quantize(h[i], self.nnz);
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
