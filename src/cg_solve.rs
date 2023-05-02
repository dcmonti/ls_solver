use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csr_dense, Op},
    CsrMatrix,
};

#[inline]
pub fn compute_alpha(
    d: &DVector<f64>,
    residue: &DVector<f64>,
    a: &CsrMatrix<f64>,
    support: &mut DVector<f64>,
) -> f64 {
    let num = d.dot(residue);
    spmm_csr_dense(0_f64, &mut *support, 1.0, Op::NoOp(a), Op::NoOp(d));
    let den = d.dot(support);
    num / den
}

#[inline]
pub fn compute_beta(
    d: &DVector<f64>,
    residue: &DVector<f64>,
    a: &CsrMatrix<f64>,
    support: &mut DVector<f64>,
) -> f64 {
    let den = d.dot(support);

    spmm_csr_dense(0_f64, &mut *support, 1.0, Op::NoOp(a), Op::NoOp(residue));
    let num = d.dot(support);
    num / den
}
