use nalgebra::DVector;
use nalgebra_sparse::{io::load_coo_from_matrix_market_file, CscMatrix};

use crate::solver;

pub enum Method {
    JA,
    GS,
    GR,
    CG,
}

pub fn read_matrix_from_matrix_market_file(file_path: &String) -> CscMatrix<f64> {
    let coo_matrix = load_coo_from_matrix_market_file(file_path).unwrap();
    CscMatrix::from(&coo_matrix)
}
pub fn solve_linear_system(
    a: &CscMatrix<f64>,
    b: &DVector<f64>,
    method: Method,
    tol: f64,
    max_iter: i32,
    omega: f64,
) {
    match method {
        Method::JA => solver::exec(&a, &b, Method::JA, tol, max_iter, omega),
        Method::GS => solver::exec(&a, &b, Method::GS, tol, max_iter, omega),
        Method::GR => solver::exec(&a, &b, Method::GR, tol, max_iter, omega),
        Method::CG => solver::exec(&a, &b, Method::CG, tol, max_iter, omega),
    }
}
