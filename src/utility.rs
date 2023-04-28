use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csr_dense, Op},
    CsrMatrix,
};

#[derive(Debug)]
pub struct Stat {
    solution: DVector<f64>,
    time: u128,
    iter: u32,
}

impl Stat {
    pub fn new(solution: DVector<f64>, time: u128, iter: u32) -> Stat {
        Stat {
            solution,
            time,
            iter,
        }
    }
    pub fn get_solution(&self) -> &DVector<f64> {
        &self.solution
    }

    pub fn get_time(&self) -> u128 {
        self.time
    }

    pub fn get_iterations(&self) -> u32 {
        self.iter
    }

    pub fn to_string(&self) -> String {
        format!(
            "Result:\n{:?}\nMethod converged in \t{} iterations \t({} ms)",
            self.get_solution(),
            self.get_iterations(),
            self.get_time()
        )
    }
}

#[derive(Debug)]
pub enum Setting {
    Solve,
    Precision,
    Default,
}
pub fn tolerance_reached(tol: f64, residue_norm: f64, b_norm: f64) -> bool {
    residue_norm / b_norm < tol
}

pub fn init_b(solution: &DVector<f64>, a: &CsrMatrix<f64>) -> DVector<f64> {
    let mut b = DVector::from_element(a.ncols(), 0.0);
    spmm_csr_dense(1.0, &mut b, 1.0, Op::NoOp(a), Op::NoOp(solution));
    b
}

#[inline]
pub fn compute_residue(
    a: &CsrMatrix<f64>,
    x: &DVector<f64>,
    b: &DVector<f64>,
    residue: &mut DVector<f64>,
) {
    spmm_csr_dense(0_f64, &mut *residue, 1.0, Op::NoOp(a), Op::NoOp(x));
    residue.axpy(1.0, b, -1.0);

}

pub fn size_are_compatible(a: &CsrMatrix<f64>, vector: &DVector<f64>, setting: &Setting) {
    match setting {
        Setting::Precision | Setting::Solve => {
            if a.ncols() != vector.nrows() {
                panic!("Vector has wrong size for matrix")
            }
        }
        _ => {}
    }
}

pub fn compute_rel_err(solution: &DVector<f64>, x: &DVector<f64>) -> f64 {
    let diff = solution - x;
    diff.norm() / solution.norm()
}
