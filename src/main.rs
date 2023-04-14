use ls_solver::{io, jacobi, utility};
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
        0 => jacobi::jacobi_solve(&a, omega, &b, tol, max_iter),
        _ => {}
    }
}
