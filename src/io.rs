use clap::Parser;
use nalgebra_sparse as nasp;
use nasp::CsrMatrix;

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
}

fn get_matrix_path() -> String {
    let args = Args::parse();
    args.matrix_path
}

pub fn read_matrix() -> CsrMatrix<f64> {
    let file_path = get_matrix_path();
    let sparse_matrix = nasp::io::load_coo_from_matrix_market_file(file_path).unwrap();
    let csr_matrix = CsrMatrix::from(&sparse_matrix);
    csr_matrix
}

pub fn get_method() -> i32 {
    let args = Args::parse();
    args.method
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
