use nalgebra::DVector;
use nalgebra_sparse::{io::load_coo_from_matrix_market_file, CscMatrix};

use crate::solver;

#[derive(Debug)]
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
    solver::exec(&a, &b, method, tol, max_iter, omega);
}
