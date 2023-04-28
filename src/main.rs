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
            let result = solver::exec(&a, &b, method, tol, max_iter, omega);
            println!(
                "Relative Error: {}\nIteration: {}\nTime: {} ms",
                utility::compute_rel_err(&solution, result.get_solution()),
                result.get_iterations(),
                result.get_time()
            )
        }
        Setting::Precision => {
            // solution is vector
            let b = utility::init_b(&vector, &a);
            let result = solver::exec(&a, &b, method, tol, max_iter, omega);
            println!(
                "Relative Error: {}\nIteration: {}\nTime: {} ms",
                utility::compute_rel_err(&vector, result.get_solution()),
                result.get_iterations(),
                result.get_time()
            )
        }
        Setting::Solve => {
            let result = solver::exec(&a, &vector, method, tol, max_iter, omega);
            println!("{}", result.to_string())
        }
    }
    
}
