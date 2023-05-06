use core::panic;
use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
};

use crate::{api::Method, utility::Setting};
use clap::Parser;
use nalgebra::DVector;
use nalgebra_sparse as nasp;
use nasp::{CooMatrix, CsrMatrix};

#[derive(Parser, Debug)]
#[clap(
    author = "Davide Monti <d.monti11@campus.unimib.it>\nSamuele Campanella <s.campanella3@campus.unimib.it>",
    version,
    about = "ls_solver",
    long_about = "Simple tool and library for linear system solution"
)]
struct Args {
    #[clap(
        help_heading = "I/O",
        help = "Input matrix (in .mtx format)",
        required = true
    )]
    matrix_path: String,

    #[clap(
        help_heading = "I/O",
        help = "Input vector (in .mtx format).\nIf not specified x := [1 1 ... 1] and b := ax",
        default_value = "None"
    )]
    vector_path: String,

    #[clap(
        help_heading = "I/O",
        short = 'o',
        long = "output",
        default_value = "None",
        help = "Output file with approximate solution, if None solution will not be printed"
    )]
    output_path: String,

    // Iterative method
    #[clap(
        help_heading = "Settings",
        short = 'm',
        long = "method",
        default_value_t = 0,
        help = "0: Jacobi\n1: Gauß-Seidel\n2: gradient\n3: conjugate gradient\n4: Jacobi-preconditioned gradient (only if matrix is SPD)"
    )]
    method: i32,

    // Tolerance
    #[clap(
        help_heading = "Settings",
        short = 't',
        long = "tolerance",
        default_value_t = 4,
        help = "Set tolerance as desired negative exponent (e.g. 4 is 0.0001)"
    )]
    tolerance: i32,

    // Max Steps
    #[clap(
        help_heading = "Settings",
        short = 'i',
        long = "max_iter",
        default_value_t = 20000,
        help = "Set max number of iterations for the routine"
    )]
    max_iter: i32,

    // Omega
    #[clap(
        help_heading = "Settings",
        short = 'O',
        long = "omega",
        default_value_t = 1.0,
        help = "Set relax factor with float desired\nUsed only if method is 0 or 1"
    )]
    omega: f64,

    // Settings
    #[clap(
        help_heading = "Settings",
        short = 's',
        long = "set-mode",
        default_value_t = 0,
        help = "Used only if [VECTOR_PATH] is specified\n0: consider vector as b and solve the system Ax=b\n1: consider vector as x and evaluate method precision"
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

pub fn read_matrix() -> CsrMatrix<f64> {
    let file_path = get_matrix_path();
    let sparse_matrix = nasp::io::load_coo_from_matrix_market_file(file_path).unwrap();

    CsrMatrix::from(&sparse_matrix)
}
pub fn read_vector() -> (DVector<f64>, Setting) {
    let file_path = get_vector_path();
    if file_path.eq(&"None") {
        (DVector::from_element(1, 0.0), Setting::Default)
    } else {
        (parse_vector(&file_path), get_setting())
    }
}

pub fn parse_vector(file_path: &String) -> DVector<f64> {
    let extension = file_path[file_path.len() - 3..].to_string();
    if extension.eq(&String::from("mtx")) {
        parse_mtx_vector(file_path)
    } else {
        parse_standard_vector(file_path)
    }
}

fn parse_standard_vector(file_path: &String) -> DVector<f64> {
    let file = File::open(file_path).unwrap();
    let lines = BufReader::new(file).lines();

    let mut vector = Vec::new();
    for line in lines {
        let f = line.unwrap().trim().parse::<f64>().unwrap();
        vector.push(f)
    }
    DVector::from(vector)
}
fn parse_mtx_vector(file_path: &String) -> DVector<f64> {
    let coo_matrix: CooMatrix<f64> = nasp::io::load_coo_from_matrix_market_file(file_path).unwrap();

    match (coo_matrix.ncols(), coo_matrix.nrows()) {
        (1, _) => {
            let mut tmp_vector = DVector::from_element(coo_matrix.nrows(), 0.0);
            for (row, _, val) in coo_matrix.triplet_iter() {
                tmp_vector[row] = *val;
            }
            tmp_vector
        }
        (_, 1) => {
            let mut tmp_vector = DVector::from_element(coo_matrix.ncols(), 0.0);
            for (_, col, val) in coo_matrix.triplet_iter() {
                tmp_vector[col] = *val;
            }
            tmp_vector
        }
        _ => {
            panic!("Vector file wrong format")
        }
    }
}

pub fn get_method() -> Method {
    let args = Args::parse();
    match args.method {
        0 => Method::JA,
        1 => Method::GS,
        2 => Method::GR,
        3 => Method::CG,
        4 => Method::PG,
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
    if !(0.0..=1.0).contains(&omega) {
        omega = 1.0
    }
    omega
}

fn get_setting() -> Setting {
    let args = Args::parse();
    match args.setting {
        0 => Setting::Solve,
        1 => Setting::Precision,
        _ => panic!("-s must be 0 or 1, try --help for more info"),
    }
}

fn get_output_path() -> Result<String, ()> {
    let args = Args::parse();
    if args.output_path.eq(&"None") {
        Err(())
    } else {
        Ok(args.output_path)
    }
}

pub fn write_to_output_path(result: &[f64]) {
    if let Ok(path) = get_output_path() {
        let output_data: Vec<String> = result.iter().map(|n| n.to_string()).collect();
        let mut file = File::create(path).unwrap();
        writeln!(file, "{}", output_data.join("\n")).unwrap();
        }
}
