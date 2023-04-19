use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csc_dense, Op},
    CscMatrix,
};

#[inline]
pub fn compute_alpha(d: &DVector<f64>, residue: &DVector<f64>, a: &CscMatrix<f64>) -> f64 {
    let num = d.dot(&residue);
    let mut a_d = DVector::from_element(residue.nrows(), 0.0);
    spmm_csc_dense(1.0, &mut a_d, 1.0, Op::NoOp(a), Op::NoOp(d));
    let den = d.dot(&a_d);
    num / den
}

#[inline]
pub fn compute_beta(d: &DVector<f64>, residue: &DVector<f64>, a: &CscMatrix<f64>) -> f64 {
    let mut a_r = DVector::from_element(residue.nrows(), 0.0);
    spmm_csc_dense(1.0, &mut a_r, 1.0, Op::NoOp(a), Op::NoOp(residue));
    let num = d.dot(&a_r);

    let mut a_d = DVector::from_element(residue.nrows(), 0.0);
    spmm_csc_dense(1.0, &mut a_d, 1.0, Op::NoOp(a), Op::NoOp(d));
    let den = d.dot(&a_d);
    num / den
}
