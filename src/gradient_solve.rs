use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csc_dense, Op},
    CscMatrix,
};


#[inline]
pub fn compute_gr_update(a: &CscMatrix<f64>, residue: &DVector<f64>, x: &mut DVector<f64>, a_r: &mut DVector<f64>) {
    let num = residue.dot(residue);  
    spmm_csc_dense(0.0, &mut *a_r, 1.0, Op::NoOp(a), Op::NoOp(residue));
    let den = residue.dot(a_r);

    let alpha = num / den;
    x.axpy(alpha, &residue, 1.0);
}   
