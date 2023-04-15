use crate::utility;
use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csc_dense, Op},
    CooMatrix, CscMatrix,
};

pub fn get_jacobi_p(a: &CscMatrix<f64>, omega: f64) -> CscMatrix<f64> {
    let diag = Vec::from(a.diagonal_as_csc().values());
    let mut p_inv_coo = CooMatrix::<f64>::new(a.nrows(), a.ncols());
    for (i, val) in diag.into_iter().enumerate() {
        p_inv_coo.push(i, i, omega / val);
    }
    CscMatrix::from(&p_inv_coo)
}

pub fn get_jacobi_update(p: &CscMatrix<f64>, residue: &DVector<f64>) -> DVector<f64> {
    let mut update = DVector::from_element(p.nrows(), 0.0);
    spmm_csc_dense(1.0, &mut update, 1.0, Op::NoOp(p), Op::NoOp(residue));
    update
}