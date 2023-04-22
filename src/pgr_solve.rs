use std::time::Instant;

use crate::{
    api::Method,
    gs_solve::{get_gs_p},
    jacobi_solve::{get_jacobi_p},
    utility::{self, Stat},
};
use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csc_dense, Op},
    CscMatrix,
};

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

    let _p = match method {
        Method::JA => Some(get_jacobi_p(a, omega)),
        Method::GS => Some(get_gs_p(a, omega)),
        _ => None,
    };
    let d = get_jacobi_p(&a, 1.0);

    let mut x = DVector::from_element(size, 0.0);

    let mut residue = utility::compute_residue(a, &x, b, size);

    let mut count = 0;

    while count < max_iter {
        let residue_norm = residue.norm();
        if utility::tolerance_reached(tol, residue_norm, b_norm) {
            // TODO: definitive output
            let duration = start.elapsed().as_millis();
            let statistics = Stat::new(x, duration, count as u32);
            return statistics;
        }
        // compute  z update
        let mut z = DVector::from_element(size, 0.0);
        spmm_csc_dense(1.0, &mut z, 1.0, Op::NoOp(&d), Op::NoOp(&residue));

        // compute alpha and x update
        let alpha = get_precond_alpha_k(a, &residue, &z);
        x.axpy(alpha, &z, 1.0);

        // compute residue update
        let mut a_z = DVector::from_element(size, 0.0);
        spmm_csc_dense(1.0, &mut a_z, 1.0, Op::NoOp(&a), Op::NoOp(&z));
        residue.axpy(-alpha, &a_z, 1.0);

        count += 1;
    }
    panic!("Method didn't converged");
}

#[inline]
pub fn get_precond_alpha_k(a: &CscMatrix<f64>, residue: &DVector<f64>, z: &DVector<f64>) -> f64 {
    let num = z.dot(residue);

    let mut a_z = DVector::from_element(residue.nrows(), 0.0);
    spmm_csc_dense(1.0, &mut a_z, 1.0, Op::NoOp(a), Op::NoOp(z));
    let den = z.dot(&a_z);
    num / den
}

#[inline]
pub fn compute_z(z: &mut DVector<f64>, p: &CscMatrix<f64>, residue: &DVector<f64>) {
    spmm_csc_dense(1.0, z, 1.0, Op::NoOp(p), Op::NoOp(residue));
}
#[inline]
pub fn compute_precond_residue(
    residue: &mut DVector<f64>,
    a: &CscMatrix<f64>,
    z: &DVector<f64>,
    alpha: f64,
) {
    let mut a_z = DVector::from_element(a.nrows(), 0.0);
    spmm_csc_dense(1.0, &mut a_z, 1.0, Op::NoOp(a), Op::NoOp(z));
    residue.axpy(-alpha, &a_z, 1.0);
}
