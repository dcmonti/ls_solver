use std::time::Instant;

use crate::{
    api::Method,
    cg_solve, gradient_solve, gs_solve, jacobi_solve, pgr_solve,
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
) -> Stat {
    let start = Instant::now();
    let size = a.ncols();
    let b_norm = b.norm();

    let p = match method {
        Method::JA | Method::PG => Some(jacobi_solve::get_jacobi_p(a, omega)),
        Method::GS => Some(gs_solve::get_gs_p(a, omega)),
        _ => None,
    };

    let mut x = DVector::from_element(size, 0.0);
    let mut residue = DVector::from_element(size, 0.0);
    utility::compute_residue(a, &x, b, size, &mut residue);

    let mut d = match method {
        // d only used for cg
        Method::CG => {
            let mut tmp_d = DVector::from_element(size, 0.0);
            utility::compute_residue(a, &x, b, size, &mut tmp_d);
            tmp_d
        },
        Method::GR => DVector::from_element(size, 0.0),
        _ => DVector::from(vec![0.0]),
    };

    let mut count = 0;

    while count < max_iter {
        let residue_norm = residue.norm();
        if utility::tolerance_reached(tol, residue_norm, b_norm) {
            let duration = start.elapsed().as_millis();
            let statistics = Stat::new(x, duration, count as u32);
            return statistics;
        }
        // TODO: change update method, using only reference if possible (ALSO FOR RESIDUE)
        // compute update
        match method {
            Method::JA => {
                jacobi_solve::compute_jacobi_update(&mut x, &p.as_ref().unwrap(), &residue);
                utility::compute_residue(a, &x, b, size, &mut residue);
            }
            Method::GS => {
                gs_solve::compute_gs_update(&mut x, &p.as_ref().unwrap(), &residue);
                utility::compute_residue(a, &x, b, size, &mut residue);
            }
            Method::GR => {
                gradient_solve::compute_gr_update(a, &residue, &mut x,&mut d);
                utility::compute_residue(a, &x, b, size, &mut residue);
            }
            Method::CG => {
                // compute alpha and update x
                let alpha = cg_solve::compute_alpha(&d, &residue, a);
                x.axpy(alpha, &d, 1.0);

                // compute residue with updated x
                utility::compute_residue(a, &x, b, size, &mut residue);

                // compute beta and update d
                let beta = cg_solve::compute_beta(&d, &residue, a);
                d.axpy(1.0, &residue, -beta);
            }
            Method::PG => {
                // compute  z update
                let mut z = DVector::from_element(size, 0.0);
                pgr_solve::compute_z(&mut z, p.as_ref().unwrap(), &residue);

                // compute alpha and x update
                let alpha = pgr_solve::get_precond_alpha_k(a, &residue, &z);
                x.axpy(alpha, &z, 1.0);

                // compute residue update
                pgr_solve::compute_precond_residue(&mut residue, a, &z, alpha);
            }
        };

        count += 1;
    }
    panic!("Method didn't converged");
}
