use crate::utility;
use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csr_dense, Op},
    CooMatrix, CsrMatrix,
};

pub fn jacobi_solve(a: &CsrMatrix<f64>, omega: f64, b: &DVector<f64>, tol: f64, max_iter: i32) {
    let size = a.ncols();
    let b_norm = b.norm();
    let p_inv = get_inverse_diag(&a, omega);

    let mut x = DVector::from_element(size, 0.0);
    let mut count = 0;
    let mut tol_reached = false;

    while count < max_iter {
        let mut residue = DVector::from_element(size, 0.0);
        spmm_csr_dense(1.0, &mut residue, 1.0, Op::NoOp(&a), Op::NoOp(&x));

        let mut residue_update = DVector::from_element(size, 0.0);
        b.sub_to(&residue, &mut residue_update);

        let residue_norm = residue_update.norm();
        if utility::tolerance_reached(tol, residue_norm, b_norm) {
            println!("{:?}", x);
            tol_reached = true;
            break;
        }

        let mut update = DVector::from_element(size, 0.0);
        spmm_csr_dense(
            1.0,
            &mut update,
            1.0,
            Op::NoOp(&p_inv),
            Op::NoOp(&mut residue_update),
        );

        x += update;
        count += 1;
    }
    if !tol_reached {
        panic!("Method didn't converged")
    }
    println!("ITERATIONS: {count}");
}

fn get_inverse_diag(a: &CsrMatrix<f64>, omega: f64) -> CsrMatrix<f64> {
    let diag = Vec::from(a.diagonal_as_csr().values());
    let mut p_inv_coo = CooMatrix::<f64>::new(a.nrows(), a.ncols());
    for (i, val) in diag.into_iter().enumerate() {
        p_inv_coo.push(i, i, omega / val);
    }
    CsrMatrix::from(&p_inv_coo)
}
