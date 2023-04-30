use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csr_dense, Op},
    CooMatrix, CsrMatrix,
};

#[inline]
pub fn get_jacobi_p(a: &CsrMatrix<f64>, omega: f64) -> CsrMatrix<f64> {
    let diag = Vec::from(a.diagonal_as_csr().values());
    let mut p_inv_coo = CooMatrix::<f64>::new(a.nrows(), a.ncols());
    for (i, val) in diag.into_iter().enumerate() {
        p_inv_coo.push(i, i, omega / val);
    }
    CsrMatrix::from(&p_inv_coo)
}

#[inline]
pub fn compute_jacobi_update(
    x: &mut DVector<f64>,
    p: &CsrMatrix<f64>,
    residue: &DVector<f64>,
    support: &mut DVector<f64>,
) {
    spmm_csr_dense(0_f64, &mut *support, 1.0, Op::NoOp(p), Op::NoOp(residue));
    x.axpy(1.0, support, 1.0);
}
