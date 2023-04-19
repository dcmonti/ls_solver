use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ls_solver::{api::{solve_linear_system, Method}, io::read_matrix, utility::init_b};
use nalgebra::DVector;
use nalgebra_sparse::{io::load_coo_from_matrix_market_file, CscMatrix};

pub fn jacobi_benchmark(c: &mut Criterion) {
    let vem1: CscMatrix<f64> = CscMatrix::from(&load_coo_from_matrix_market_file("benches/test_matrices/vem1.mtx").unwrap());
    let vem2: CscMatrix<f64> = CscMatrix::from(&load_coo_from_matrix_market_file("benches/test_matrices/vem2.mtx").unwrap());
    let spa1: CscMatrix<f64> = CscMatrix::from(&load_coo_from_matrix_market_file("benches/test_matrices/spa1.mtx").unwrap());
    let spa2: CscMatrix<f64> = CscMatrix::from(&load_coo_from_matrix_market_file("benches/test_matrices/spa2.mtx").unwrap());

    let vem1_solution = DVector::from_element(vem1.nrows(), 1.0);
    let vem2_solution = DVector::from_element(vem2.nrows(), 1.0);
    let spa1_solution = DVector::from_element(spa1.nrows(), 1.0);
    let spa2_solution = DVector::from_element(spa2.nrows(), 1.0);

    let vem1_b = init_b(&vem1_solution, &vem1);
    let vem2_b = init_b(&vem2_solution, &vem2);
    let spa1_b = init_b(&spa1_solution, &spa1);
    let spa2_b = init_b(&spa2_solution, &spa2);
    
    let mut group = c.benchmark_group("vem1");
    group.measurement_time(Duration::from_secs(60));
    for tol in [ -4, -6, -8, -10] {
        group.bench_with_input(BenchmarkId::new("ja", tol), &tol,|b, _tol| {
            b.iter( ||
                solve_linear_system(&vem1, &vem1_b, Method::JA, 10.0_f64.powi(tol), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("gs", tol), &tol,|b, _tol| {
            b.iter( ||
                solve_linear_system(&vem1, &vem1_b, Method::GS, 10.0_f64.powi(tol), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("gr", tol), &tol,|b, _tol| {
            b.iter( ||
                solve_linear_system(&vem1, &vem1_b, Method::GR, 10.0_f64.powi(tol), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("cg", tol), &tol,|b, _tol| {
            b.iter( ||
                solve_linear_system(&vem1, &vem1_b, Method::CG, 10.0_f64.powi(tol), 20000, 1.0)
            )
        });
    }   
}

criterion_group!(benches, jacobi_benchmark);
criterion_main!(benches);
