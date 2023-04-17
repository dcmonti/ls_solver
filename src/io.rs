use core::panic;

use crate::api::Method;
use clap::Parser;
use nalgebra::DVector;
use nalgebra_sparse as nasp;
use nasp::{CscMatrix, CooMatrix, coo};

#[derive(Parser, Debug)]
#[clap(
    author = "Davide Monti <d.monti11@campus.unimib.it>",
    version,
    about = "ls_solver",
    long_about = "Simple tool and library for linear system solution"
)]
struct Args {
    #[clap(help = "Input matrix (in .mtx format)", required = true)]
    matrix_path: String,

    #[clap(help = "Input vector", default_value = "None")]
    vector_path: String,

    // Iterative method
    #[clap(
        help_heading = "Iterative Method",
        short = 'm',
        long = "method",
        default_value_t = 0,
        help = "0: Jacobi, 1: Gauß-Seidel, 2: gradient, 3: conjugate gradient"
    )]
    method: i32,

    // Tolerance
    #[clap(
        help_heading = "Tolerance",
        short = 't',
        long = "tolerance",
        default_value_t = 4,
        help = "Set tolerance as desired negative exponent (e.g. 4 is 0.0001)"
    )]
    tolerance: i32,

    // Max Steps
    #[clap(
        help_heading = "Max iteration",
        short = 'i',
        long = "max_iter",
        default_value_t = 20000,
        help = "Set max number of iterations for the routine"
    )]
    max_iter: i32,

    // Omega
    #[clap(
        help_heading = "Relax factor",
        short = 'o',
        long = "omega",
        default_value_t = 1.0,
        help = "Set relax factor with float in (0,1]"
    )]
    omega: f64,

    // Settings
    #[clap(
        help_heading = "Set running mode",
        short = 's',
        long = "setting",
        default_value_t = 0,
        help = "0: consider vector as b and solve the system Ax=b\n1: consider vector as x and evaluate method precision"
    )]
    setting: i32,

}

fn get_matrix_path() -> String {
    let args = Args::parse();
    args.matrix_path
}

fn get_vector_path() -> String {
    let args = Args::parse();
    args.vector_path
}

pub fn read_matrix() -> CscMatrix<f64> {
    let file_path = get_matrix_path();
    let sparse_matrix = nasp::io::load_coo_from_matrix_market_file(file_path).unwrap();
    let csc_matrix = CscMatrix::from(&sparse_matrix);
    csc_matrix
}

pub fn read_vector() -> DVector<f64> {
    let file_path = get_vector_path();
    let coo_matrix:CooMatrix<f64> = nasp::io::load_coo_from_matrix_market_file(file_path).unwrap();

    let mut row_m = false;
    let mut vector = match (coo_matrix.ncols(), coo_matrix.nrows()) {
        (1, _) => {
            row_m = true;
            DVector::from_element(coo_matrix.nrows(), 0.0)
        }
        (_, 1) => {
            DVector::from_element(coo_matrix.ncols(), 0.0)
        }
        _ => {panic!("Vector file wrong format")}
    };
    
    // TODO: better impl, check dimension, implement setting, in solver check dimension correct for A
    if row_m {
        for (row, _, val) in coo_matrix.triplet_iter() {
            vector[row] = *val;
        }
    } else {
        for (_, col, val) in coo_matrix.triplet_iter() {
            vector[col] = *val;
        }
    }
    vector
}

pub fn get_method() -> Method {
    let args = Args::parse();
    match args.method {
        0 => Method::JA,
        1 => Method::GS,
        2 => Method::GR,
        3 => Method::CG,
        _ => panic!("Value must be between 0 and 3, try --help for more information"),
    }
}

pub fn get_tol() -> f64 {
    let args = Args::parse();
    let tol_exp = -args.tolerance;
    10.0_f64.powi(tol_exp)
}

pub fn get_max_iter() -> i32 {
    let args = Args::parse();
    args.max_iter
}

pub fn get_omega() -> f64 {
    let args = Args::parse();
    let mut omega = args.omega;
    if omega < 0.0 || omega > 1.0 {
        omega = 1.0
    }
    omega
}

fn get_setting() -> i32 {
    let args = Args::parse();
    match args.setting {
        0 | 1 => args.setting,
        _ => panic!("-s must be 0 or 1, try --help for more info")
    }
}
