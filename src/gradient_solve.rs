use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csc_dense, Op}, CscMatrix,
};

pub fn get_alpha_k(a: &CscMatrix<f64>, residue: &DVector<f64>) -> f64 {
    let num = (residue.transpose() * residue)[(0,0)];
    let mut a_r = DVector::from_element(residue.nrows(), 0.0);
    spmm_csc_dense(1.0, &mut a_r, 1.0, Op::NoOp(a), Op::NoOp(residue));
    let den = (residue.transpose() * a_r)[(0,0)];
    num/den
}
