use clap::Parser;
use csd::to_csdnnz;
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim_q, Options};
use ellalgo_rs::ell::Ell;
use multiplierless_rs::lowpass_oracle::{FilterDesignConstruct, LowpassOracle};
use multiplierless_rs::lowpass_oracle_q::LowpassOracleQ;
use multiplierless_rs::spectral_fact::{spectral_fact_fft, spectral_fact_root};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "multiplierless")]
struct Cli {
    spec_file: PathBuf,
}

#[derive(Deserialize)]
struct Spec {
    #[serde(default = "default_filter_order")]
    filter_order: usize,
    #[serde(default = "default_passband_edge")]
    passband_edge: f64,
    #[serde(default = "default_stopband_edge")]
    stopband_edge: f64,
    #[serde(default = "default_passband_ripple")]
    passband_ripple: f64,
    #[serde(default = "default_stopband_attenuation")]
    stopband_attenuation: f64,
    #[serde(default = "default_csd_nnz")]
    csd_nnz: u32,
    #[serde(default = "default_discretization_factor")]
    discretization_factor: usize,
    #[serde(default = "default_max_iters")]
    max_iters: usize,
    #[serde(default = "default_tolerance")]
    tolerance: f64,
    #[serde(default = "default_ellipsoid_radius")]
    ellipsoid_radius: f64,
    #[serde(default = "default_parallel_cut")]
    parallel_cut: bool,
    #[serde(default = "default_spectral_method")]
    spectral_method: String,
    #[serde(default = "default_root_tolerance")]
    root_tolerance: f64,
    #[serde(default)]
    verilog: Option<VerilogSpec>,
}

#[derive(Deserialize, Default)]
struct VerilogSpec {
    #[serde(default = "default_input_width")]
    input_width: i32,
    #[serde(default = "default_module_name")]
    module_name: String,
    #[serde(default = "default_verilog_form")]
    #[allow(dead_code)]
    form: String,
}

fn default_filter_order() -> usize {
    32
}
fn default_passband_edge() -> f64 {
    0.12
}
fn default_stopband_edge() -> f64 {
    0.20
}
fn default_passband_ripple() -> f64 {
    0.125
}
fn default_stopband_attenuation() -> f64 {
    0.125
}
fn default_csd_nnz() -> u32 {
    7
}
fn default_discretization_factor() -> usize {
    15
}
fn default_max_iters() -> usize {
    50000
}
fn default_tolerance() -> f64 {
    1e-14
}
fn default_ellipsoid_radius() -> f64 {
    40.0
}
fn default_parallel_cut() -> bool {
    true
}
fn default_spectral_method() -> String {
    "fft".to_string()
}
fn default_root_tolerance() -> f64 {
    1e-8
}
fn default_input_width() -> i32 {
    16
}
fn default_module_name() -> String {
    "fir_filter".to_string()
}
fn default_verilog_form() -> String {
    "transpose".to_string()
}

#[derive(Serialize)]
struct Output {
    filter_order: usize,
    csd_nnz: u32,
    iterations: usize,
    spectral_method: String,
    coefficients: Vec<CoeffOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verilog: Option<String>,
}

#[derive(Serialize)]
struct CoeffOutput {
    index: usize,
    value: f64,
    csd: String,
}

fn build_range_expr(csd: &str, start: usize, length: usize, max_power: i32) -> String {
    let mut expr = String::new();
    let mut first = true;
    let end = start + length.min(csd.len());
    for i in start..end {
        let power = max_power - i as i32;
        match csd.as_bytes()[i] {
            b'+' => {
                if first {
                    expr += &format!("x_shift{power}");
                    first = false;
                } else {
                    expr += &format!(" + x_shift{power}");
                }
            }
            b'-' => {
                if first {
                    expr += &format!("-x_shift{power}");
                    first = false;
                } else {
                    expr += &format!(" - x_shift{power}");
                }
            }
            _ => {}
        }
    }
    expr
}

fn generate_transpose_verilog(
    csd_strings: &[String],
    input_width: i32,
    module_name: &str,
) -> String {
    let n = csd_strings.len();
    let max_len = csd_strings.iter().map(|s| s.len()).max().unwrap_or(0);
    let max_power = max_len as i32 - 1;
    let output_width = input_width + max_power;

    let mut v = String::new();
    v += &format!("\nmodule {module_name} (");
    v += "\n    input clk,";
    v += "\n    input rst_n,";
    v += &format!("\n    input signed [{}:0] x,", input_width - 1);
    v += &format!("\n    output signed [{}:0] y", output_width - 1);
    v += "\n);";

    v += "\n\n    // Transpose-form pipeline registers";
    for idx in 0..n {
        v += &format!("\n    reg signed [{}:0] sum{idx};", output_width - 1);
    }

    v += "\n\n    always @(posedge clk or negedge rst_n) begin";
    v += "\n        if (!rst_n) begin";
    for idx in 0..n {
        v += &format!("\n            sum{idx} <= 0;");
    }
    v += "\n        end else begin";

    for idx in 0..n {
        let coeff_idx = n - 1 - idx;
        let csd = &csd_strings[coeff_idx];
        let mut raw = csd.replace('.', "");
        while raw.len() < max_len {
            raw.insert(0, '0');
        }
        let expr = build_range_expr(&raw, 0, raw.len(), max_power);

        if idx == 0 {
            if expr.is_empty() {
                v += "\n            sum0 <= 0;";
            } else {
                v += &format!("\n            sum0 <= {expr};");
            }
        } else if expr.is_empty() {
            v += &format!("\n            sum{idx} <= sum{};", idx - 1);
        } else {
            v += &format!("\n            sum{idx} <= sum{} + {expr};", idx - 1);
        }
    }

    v += "\n        end";
    v += "\n    end";
    v += &format!("\n\n    assign y = sum{};", n - 1);
    v += "\nendmodule\n";
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let json_str = fs::read_to_string(&cli.spec_file)?;
    let spec: Spec = serde_json::from_str(&json_str)?;

    let n = spec.filter_order;
    let nnz = spec.csd_nnz;

    let fdc = FilterDesignConstruct::new(
        n,
        spec.passband_edge,
        spec.stopband_edge,
        spec.passband_ripple,
        spec.stopband_attenuation,
        spec.discretization_factor,
    );
    let mut spsq = fdc.spsq;

    let lowpass = LowpassOracle::new(fdc);
    let mut omega = LowpassOracleQ::new(nnz, lowpass);

    let r0 = Arr::new(n);
    let mut ellip = Ell::new_with_scalar(spec.ellipsoid_radius, r0);
    ellip.no_defer_trick = !spec.parallel_cut;

    let options = Options::new(spec.max_iters, spec.tolerance);

    let (r_opt, num_iters) = cutting_plane_optim_q(&mut omega, &mut ellip, &mut spsq, &options);

    let r = match r_opt {
        Some(r) => r,
        None => {
            eprintln!("Optimization failed — no feasible solution after {num_iters} iterations.");
            std::process::exit(1);
        }
    };

    let h = if spec.spectral_method == "fft" {
        spectral_fact_fft(&r)
    } else {
        spectral_fact_root(&r, spec.root_tolerance)
    };

    let coefficients: Vec<CoeffOutput> = h
        .iter()
        .enumerate()
        .map(|(i, &val)| CoeffOutput {
            index: i,
            value: val,
            csd: to_csdnnz(val, nnz),
        })
        .collect();

    let verilog = spec.verilog.map(|vl| {
        let csd_strings: Vec<String> = coefficients.iter().map(|c| c.csd.clone()).collect();
        generate_transpose_verilog(&csd_strings, vl.input_width, &vl.module_name)
    });

    let output = Output {
        filter_order: n,
        csd_nnz: nnz,
        iterations: num_iters,
        spectral_method: spec.spectral_method,
        coefficients,
        verilog,
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
