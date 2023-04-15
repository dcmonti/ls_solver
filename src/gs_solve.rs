use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spsolve_csc_lower_triangular, Op},
    CscMatrix,
};

pub fn get_gs_p(a: &CscMatrix<f64>, omega: f64) -> CscMatrix<f64> {
    let lt = a.lower_triangle();
    lt * omega
}

pub fn get_gs_update(p_lt: &CscMatrix<f64>, residue: &DVector<f64>) -> DVector<f64> {
    let mut update = DVector::from_element(residue.nrows(), 0.0);
    update += residue;
    match spsolve_csc_lower_triangular(Op::NoOp(&p_lt), &mut update) {
        Ok(_) => update,
        Err(_) => {
            panic!("Gauss Seidel didn't converged, try with another method")
        }
    }
}
