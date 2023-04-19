use std::time::Instant;

use crate::{
    api::Method,
    cg_solve, gradient_solve,
    gs_solve::{get_gs_p, get_gs_update},
    jacobi_solve::{get_jacobi_p, get_jacobi_update},
    utility::{self, Stat},
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
    let start = Instant::now();
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
        // d only used for cg
        Method::CG => utility::compute_residue(a, &x, b, size),
        _ => DVector::from(vec![0.0]),
    };

    let mut count = 0;
    let mut tol_reached = false;

    while count < max_iter {
        let residue_norm = residue.norm();
        if utility::tolerance_reached(tol, residue_norm, b_norm) {
            // TODO: definitive output
            let duration = start.elapsed().as_millis();
            let sol: Vec<f64> = x.into_iter().map(|val| *val).collect();
            let statistics = Stat::new(sol, duration, count as u32);
            println!("{}", statistics.to_string());
            tol_reached = true;
            break;
        }

        // compute update
        match method {
            Method::JA => {
                let update = get_jacobi_update(&p.as_ref().unwrap(), &residue);
                x += update;
                residue = utility::compute_residue(a, &x, b, size);
            }
            Method::GS => {
                let update = get_gs_update(&p.as_ref().unwrap(), &residue);
                x += update;
                residue = utility::compute_residue(a, &x, b, size);
            }
            Method::GR => {
                let alpha = gradient_solve::get_alpha_k(a, &residue);
                x += alpha * &residue;
                residue = utility::compute_residue(a, &x, b, size);
            }
            Method::CG => {
                // compute alpha and update x
                let alpha = cg_solve::compute_alpha(&d, &residue, a);
                x.axpy(alpha, &d, 1.0);
                

                // compute residue with updated x
                residue = utility::compute_residue(a, &x, b, size);

                // compute beta and update d
                let beta = cg_solve::compute_beta(&d, &residue, a);
                d.axpy(1.0, &residue, -beta);
                
            }
        };

        count += 1;
    }
    if !tol_reached {
        panic!("Method didn't converged")
    }
}
