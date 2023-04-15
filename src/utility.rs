use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csc_dense, Op},
    CscMatrix,
};

pub fn tolerance_reached(tol: f64, residue_norm: f64, b_norm: f64) -> bool {
    residue_norm / b_norm < tol
}

pub fn init_b(solution: &DVector<f64>, a: &CscMatrix<f64>) -> DVector<f64> {
    let mut b = DVector::from_element(a.ncols(), 0.0);
    spmm_csc_dense(1.0, &mut b, 1.0, Op::NoOp(a), Op::NoOp(solution));
    b
}

pub fn compute_residue(
    a: &CscMatrix<f64>,
    x: &DVector<f64>,
    b: &DVector<f64>,
    size: usize,
) -> DVector<f64> {
    let mut residue = DVector::from_element(size, 0.0);
    spmm_csc_dense(1.0, &mut residue, 1.0, Op::NoOp(a), Op::NoOp(x));

    let mut residue_update = DVector::from_element(size, 0.0);
    b.sub_to(&residue, &mut residue_update);
    residue_update
}

pub enum Method {
    JA,
    GS,
    GR,
    CG,
}
