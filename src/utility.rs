use nalgebra::{self, DVector};
use nalgebra_sparse::{
    ops::{serial::spmm_csc_dense, Op},
    CscMatrix,
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

pub fn size_are_compatible(a: &CscMatrix<f64>, vector: &DVector<f64>, setting: &Setting) {
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
    diff.norm()/solution.norm()

}
