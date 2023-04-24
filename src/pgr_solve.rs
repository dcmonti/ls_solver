


use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csc_dense, Op},
    CscMatrix,
};

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
pub fn compute_precond_residue(residue: &mut DVector<f64>, a: &CscMatrix<f64>, z: &DVector<f64>, alpha: f64) {
    let mut a_z = DVector::from_element(a.nrows(), 0.0);
    spmm_csc_dense(1.0, &mut a_z, 1.0, Op::NoOp(a), Op::NoOp(z));
    residue.axpy(-alpha, &a_z, 1.0);
}

                