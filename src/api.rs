use nalgebra::DVector;
use nalgebra_sparse::{io::load_coo_from_matrix_market_file, CscMatrix};

use crate::{
    solver,
    utility::{compute_rel_err, init_b, Stat},
};

#[derive(Debug)]
pub enum Method {
    JA,
    GS,
    GR,
    CG,
}
pub struct Performance {
    pub rel_err: f64,
    pub time: u128,
    pub iter: u32,
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
) -> Stat {
    solver::exec(&a, &b, method, tol, max_iter, omega)
}

pub fn compute_performance(
    a: &CscMatrix<f64>,
    solution: &DVector<f64>,
    method: Method,
    tol: f64,
    max_iter: i32,
    omega: f64,
) -> Performance {
    let b = init_b(solution, a);
    let result = solver::exec(&a, &b, method, tol, max_iter, omega);
    let rel_err = compute_rel_err(solution, result.get_solution());
    Performance {
        rel_err: rel_err,
        time: result.get_time(),
        iter: result.get_iterations(),
    }
}