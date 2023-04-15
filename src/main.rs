use ls_solver::{
    io, solver,
    utility::{self, Method},
};
use nalgebra::{self, DVector};

fn main() {
    let a = io::read_matrix();
    let tol = io::get_tol();
    let max_iter = io::get_max_iter();
    let omega = io::get_omega();

    let solution = DVector::from_element(a.ncols(), 1.0);
    let b = utility::init_b(&solution, &a);

    let method = io::get_method();
    match method {
        Method::JA => solver::exec(&a, &b, Method::JA, tol, max_iter, omega),
        Method::GS => solver::exec(&a, &b, Method::GS, tol, max_iter, omega),
        Method::GR => solver::exec(&a, &b, Method::GR, tol, max_iter, omega),
        _ => {}
    }
}
