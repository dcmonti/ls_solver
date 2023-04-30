use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spsolve_csc_lower_triangular, Op},
    CscMatrix, CsrMatrix,
};

#[inline]
pub fn get_gs_p(a: &CsrMatrix<f64>, omega: f64) -> CscMatrix<f64> {
    let p = a.lower_triangle() / omega;
    CscMatrix::from(&p)
}

#[inline]
pub fn compute_gs_update(x: &mut DVector<f64>, p_lt: &CscMatrix<f64>, residue: &mut DVector<f64>) {
    match spsolve_csc_lower_triangular(Op::NoOp(p_lt), &mut *residue) {
        Ok(_) => x.axpy(1.0, residue, 1.0),
        Err(_) => {
            panic!("Gauss Seidel didn't converged, try with another method")
        }
    }
}
