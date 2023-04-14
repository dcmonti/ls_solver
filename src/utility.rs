use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csr_dense, Op},
    CsrMatrix,
};

pub fn tolerance_reached(tol: f64, residue_norm: f64, b_norm: f64) -> bool {
    residue_norm / b_norm < tol
}

pub fn init_b(solution: &DVector<f64>, a: &CsrMatrix<f64>) -> DVector<f64> {
    let mut b = DVector::from_element(a.ncols(), 0.0);
    spmm_csr_dense(1.0, &mut b, 1.0, Op::NoOp(a), Op::NoOp(solution));
    b
}
