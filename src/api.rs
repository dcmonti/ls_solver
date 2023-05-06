use std::fmt;

use nalgebra::{Const, DVector};
use nalgebra_sparse::{io::load_coo_from_matrix_market_file, CsrMatrix};

use crate::{
    io, solver,
    utility::{compute_rel_err, init_b},
};

/// Method is used to set the method used by the routine. 
/// 
/// With:
/// * **Method::JA** = Jacobi
/// * **Method::GS** = Gauss-Seidel
/// * **Method::GR** = gradient
/// * **Method::CG** = conjugate gradient
/// * **Method::PG** = gradient with Jacobi preconditioning
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
/// Performace is a struct returned by [compute_precision].
/// 
/// It contains three fields:
/// * **rel_err**: the relative error of the computed result against the correct one
/// * **time**: the time the routine take to solve the system
/// * **iter**: the number of iteration the routine take to solve the system
pub struct Performance {
    pub rel_err: f64,
    pub time: u128,
    pub iter: u32,
}

/// Stat is a struct returned by [solve_linear_system].
/// 
/// It contains three fields:
/// * **solution**: the solution computed by the routine
/// * **time**: the time the routine take to solve the system
/// * **iter**: the number of iteration the routine take to solve the system
#[derive(Debug)]
pub struct Stat {
    solution: DVector<f64>,
    time: u128,
    iter: u32,
}

impl Stat {
    pub fn new(solution: DVector<f64>, time: u128, iter: u32) -> Stat {
        Stat {
            solution,
            time,
            iter,
        }
    }
    pub fn get_solution(&self) -> &DVector<f64> {
        &self.solution
    }

    pub fn get_time(&self) -> u128 {
        self.time
    }

    pub fn get_iterations(&self) -> u32 {
        self.iter
    }
}
impl fmt::Display for Stat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Result:\n{:?}\nMethod converged in \t{} iterations \t({} ms)",self.solution ,self.iter, self.time)
    }
}

/// Parse a matrix market file (.mtx) and return the csr matrix representation of it.
/// Panic if matrix is not sparse
pub fn read_matrix_from_matrix_market_file(file_path: &String) -> CsrMatrix<f64> {
    let coo_matrix = load_coo_from_matrix_market_file(file_path).unwrap();
    CsrMatrix::from(&coo_matrix)
}

/// Parse a vector from a matrix market file or a file with each row representing the entry of the vector.
pub fn read_vector_from_file(file_path: &String) -> DVector<f64> {
    io::parse_vector(file_path)
}

/// Initialize a vector with dimension = \[size\] and each entry = value
pub fn init_solution(size: usize, value: f64) -> DVector<f64> {
    DVector::from_element(size, value)
}

/// Initialize a vector with dimension = \[size\] and each entry a random value between 0 and 1
pub fn init_random_vector(size: usize) -> DVector<f64> {
    DVector::new_random_generic(nalgebra::Dyn(size), Const::<1>)
}

/// Solves the linear system ax=b and returns a [Stat] instance:
/// 
/// ### Arguments:
/// * **a**: matrix
/// * **b**: vector of constant terms
/// * **method**: an istance of the enum Method
/// * **tol**: the tolerance required to stop the routine
/// * **max_iter**: the maximum number of iteration after wich the routine stops
/// * **omega**: the relaxation factor, used only if method is either Jacobi or Gauss-Seidel
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

/// Determines the accuracy of x (solution computed by the routine) against the correct one and returns an istance of [Performance].
/// 
/// Where x is the solution of the system ax = b, with b := a*solution.
/// 
/// ### Arguments:
/// * **a**: matrix 
/// * **solution**: the given solution of the system
/// * **method**: an istance of the enum Method
/// * **tol**: the tolerance required to stop the routine
/// * **max_iter**: the maximum number of iteration after wich the routine stops
/// * **omega**: the relaxation factor, used only if method is either Jacobi or Gauss-Seidel
pub fn compute_precision(
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
