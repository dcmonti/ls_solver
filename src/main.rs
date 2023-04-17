use ls_solver::{
    io, solver,
    utility::{self, size_are_compatible, Setting},
};
use nalgebra::{self, DVector};

fn main() {
    let a = io::read_matrix();
    let (vector, setting) = io::read_vector();
    size_are_compatible(&a, &vector, &setting);

    let method = io::get_method();
    let tol = io::get_tol();
    let max_iter = io::get_max_iter();
    let omega = io::get_omega();

    match setting {
        Setting::Default => {
            let solution = DVector::from_element(a.ncols(), 1.0);
            let b = utility::init_b(&solution, &a);
            solver::exec(&a, &b, method, tol, max_iter, omega)
        }
        Setting::Precision => {
            // solution is vector
            let b = utility::init_b(&vector, &a);
            solver::exec(&a, &b, method, tol, max_iter, omega)
        }
        Setting::Solve => solver::exec(&a, &vector, method, tol, max_iter, omega),
    }
}
