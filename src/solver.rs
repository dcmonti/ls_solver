use crate::{
    gradient_solve,
    gs_solve::{get_gs_p, get_gs_update},
    jacobi_solve::{get_jacobi_p, get_jacobi_update},
    utility::{self, Method},
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
        Method::GR => None,
        Method::CG => {
            todo!()
        }
    };

    let mut x = DVector::from_element(size, 0.0);
    let mut count = 0;
    let mut tol_reached = false;

    while count < max_iter {
        let residue = utility::compute_residue(a, &x, b, size);

        let residue_norm = residue.norm();
        if utility::tolerance_reached(tol, residue_norm, b_norm) {
            println!("{:?}", x);
            tol_reached = true;
            break;
        }

        match method {
            Method::JA => {
                let update = get_jacobi_update(&p.as_ref().unwrap(), &residue);
                x += update;
            }
            Method::GS => {
                let update = get_gs_update(&p.as_ref().unwrap(), &residue);
                x += update
            }
            Method::GR => {
                let alpha = gradient_solve::get_alpha_k(a, &residue);
                x += alpha * residue;
            }
            Method::CG => {
                todo!()
            }
        };

        count += 1;
    }
    if !tol_reached {
        panic!("Method didn't converged")
    }
    println!("ITERATIONS: {count}");
}
