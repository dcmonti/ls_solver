use nalgebra::{Const, DVector};
use nalgebra_sparse::{io::load_coo_from_matrix_market_file, CsrMatrix};

use crate::{
    io, solver,
    utility::{compute_rel_err, init_b, Stat},
};

#[derive(Debug)]
pub enum Method {
    JA,
    GS,
    GR,
    CG,
    PG,
}

impl Method {
    pub fn copy(&self) -> Method {
        match self {
            Method::JA => Method::JA,
            Method::GS => Method::GS,
            Method::CG => Method::CG,
            Method::GR => Method::GR,
            Method::PG => Method::PG,
        }
    }
}
pub struct Performance {
    pub rel_err: f64,
    pub time: u128,
    pub iter: u32,
}

pub fn read_matrix_from_matrix_market_file(file_path: &String) -> CsrMatrix<f64> {
    let coo_matrix = load_coo_from_matrix_market_file(file_path).unwrap();
    CsrMatrix::from(&coo_matrix)
}

pub fn read_vector_from_file(file_path: &String) -> DVector<f64> {
    io::parse_vector(file_path)
}
pub fn init_solution(size: usize, value: f64) -> DVector<f64> {
    DVector::from_element(size, value)
}
pub fn init_random_vector(size: usize) -> DVector<f64> {
    DVector::new_random_generic(nalgebra::Dyn(size), Const::<1>)
}

pub fn solve_linear_system(
    a: &CsrMatrix<f64>,
    b: &DVector<f64>,
    method: Method,
    tol: f64,
    max_iter: i32,
    omega: f64,
) -> Stat {
    solver::exec(a, b, method, tol, max_iter, omega)
}

pub fn compute_performance(
    a: &CsrMatrix<f64>,
    solution: &DVector<f64>,
    method: Method,
    tol: f64,
    max_iter: i32,
    omega: f64,
) -> Performance {
    let b = init_b(solution, a);
    let result = solver::exec(a, &b, method, tol, max_iter, omega);
    let rel_err = compute_rel_err(solution, result.get_solution());
    Performance {
        rel_err,
        time: result.get_time(),
        iter: result.get_iterations(),
    }
}
