use crate::{
    api::Method,
    cg_solve, gradient_solve,
    gs_solve::{get_gs_p, get_gs_update},
    jacobi_solve::{get_jacobi_p, get_jacobi_update},
    utility,
};
use nalgebra::{self, DVector};
use nalgebra_sparse::CscMatrix;

pub fn exec(
    a: &CscMatrix<f64>,
    b: &DVector<f64>,
    method: Method,
    tol: f64,
    max_iter: i32,
    omega: f64,
) {
    let size = a.ncols();
    let b_norm = b.norm();

    let p = match method {
        Method::JA => Some(get_jacobi_p(a, omega)),
        Method::GS => Some(get_gs_p(a, omega)),
        _ => None,
    };

    let mut x = DVector::from_element(size, 0.0);
    let mut residue = utility::compute_residue(a, &x, b, size);
    let mut d = match method {
        Method::CG => Some(utility::compute_residue(a, &x, b, size)),
        _ => None,
    };

    let mut count = 0;
    let mut tol_reached = false;

    while count < max_iter {
        let residue_norm = residue.norm();
        if utility::tolerance_reached(tol, residue_norm, b_norm) {
            println!("{:?}", x);
            tol_reached = true;
            break;
        }

        match method {
            Method::JA => {
                residue = utility::compute_residue(a, &x, b, size);
                let update = get_jacobi_update(&p.as_ref().unwrap(), &residue);
                x += update;
            }
            Method::GS => {
                residue = utility::compute_residue(a, &x, b, size);
                let update = get_gs_update(&p.as_ref().unwrap(), &residue);
                x += update
            }
            Method::GR => {
                residue = utility::compute_residue(a, &x, b, size);
                let alpha = gradient_solve::get_alpha_k(a, &residue);
                x += alpha * &residue;
            }
            Method::CG => {
                // compute alpha and update x
                let alpha = cg_solve::compute_alpha(&d.as_ref().unwrap(), &residue, a);
                x += alpha * d.as_ref().unwrap();

                // compute residue with updated x
                residue = utility::compute_residue(a, &x, b, size);

                // compute beta and update d
                let beta = cg_solve::compute_beta(&d.as_ref().unwrap(), &residue, a);
                d = Some(&residue - beta * d.as_ref().unwrap());
            }
        };

        count += 1;
    }
    if !tol_reached {
        panic!("Method didn't converged")
    }
    println!("ITERATIONS: {count}");
}
