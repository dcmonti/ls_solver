use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csc_dense, Op},
    CscMatrix,
};

pub fn compute_alpha(d: &DVector<f64>, residue: &DVector<f64>, a: &CscMatrix<f64>) -> f64 {
    let num = (d.transpose() * residue)[(0, 0)];
    let mut a_d = DVector::from_element(residue.nrows(), 0.0);
    spmm_csc_dense(1.0, &mut a_d, 1.0, Op::NoOp(a), Op::NoOp(d));
    let den = (d.transpose() * a_d)[(0, 0)];
    num / den
}

pub fn compute_beta(d: &DVector<f64>, residue: &DVector<f64>, a: &CscMatrix<f64>) -> f64 {
    let mut a_r = DVector::from_element(residue.nrows(), 0.0);
    spmm_csc_dense(1.0, &mut a_r, 1.0, Op::NoOp(a), Op::NoOp(residue));
    let num = (d.transpose() * a_r)[(0, 0)];

    let mut a_d = DVector::from_element(residue.nrows(), 0.0);
    spmm_csc_dense(1.0, &mut a_d, 1.0, Op::NoOp(a), Op::NoOp(d));
    let den = (d.transpose() * a_d)[(0, 0)];
    num / den
}
