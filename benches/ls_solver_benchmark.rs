use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, SamplingMode};
use ls_solver::{api::{solve_linear_system, Method}, io::read_matrix, utility::init_b};
use nalgebra::DVector;
use nalgebra_sparse::{io::load_coo_from_matrix_market_file, CscMatrix};

pub fn vem1_benchmark(c: &mut Criterion) {
    let matr: CscMatrix<f64> = CscMatrix::from(&load_coo_from_matrix_market_file("benches/test_matrices/vem1.mtx").unwrap());
    
    let matr_solution = DVector::from_element(matr.nrows(), 1.0);

    let matr_b = init_b(&matr_solution, &matr);
    
    let mut group = c.benchmark_group("vem1");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));

    for met in [Method::JA, Method::GS, Method::GR, Method::CG] {
        group.bench_with_input(BenchmarkId::new("4", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("6", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("8", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("10", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            )
        });
    }   
}

pub fn vem2_benchmark(c: &mut Criterion) {
    let matr: CscMatrix<f64> = CscMatrix::from(&load_coo_from_matrix_market_file("benches/test_matrices/vem2.mtx").unwrap());
    
    let matr_solution = DVector::from_element(matr.nrows(), 1.0);

    let matr_b = init_b(&matr_solution, &matr);
    
    let mut group = c.benchmark_group("vem2");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));

    for met in [Method::JA, Method::GS, Method::GR, Method::CG] {
        group.bench_with_input(BenchmarkId::new("4", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("6", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("8", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("10", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            )
        });
    }     
}

pub fn spa1_benchmark(c: &mut Criterion) {
    let matr: CscMatrix<f64> = CscMatrix::from(&load_coo_from_matrix_market_file("benches/test_matrices/spa1.mtx").unwrap());
    
    let matr_solution = DVector::from_element(matr.nrows(), 1.0);

    let matr_b = init_b(&matr_solution, &matr);
    
    let mut group = c.benchmark_group("spa1");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));

    for met in [Method::JA, Method::GS, Method::GR, Method::CG] {
        group.bench_with_input(BenchmarkId::new("4", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("6", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("8", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("10", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            )
        });
    }   
}

pub fn spa2_benchmark(c: &mut Criterion) {
    let matr: CscMatrix<f64> = CscMatrix::from(&load_coo_from_matrix_market_file("benches/test_matrices/spa2.mtx").unwrap());
    
    let matr_solution = DVector::from_element(matr.nrows(), 1.0);

    let matr_b = init_b(&matr_solution, &matr);
    
    let mut group = c.benchmark_group("spa2");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));

    for met in [Method::JA, Method::GS, Method::GR, Method::CG] {
        group.bench_with_input(BenchmarkId::new("4", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("6", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("8", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            )
        });
        group.bench_with_input(BenchmarkId::new("10", met_to_str(&met)), &met,|b, met| {
            b.iter( ||
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            )
        });
    }   
}

    pub fn met_to_str(met: &Method) -> String {
        match met {
            Method::JA => String::from("ja"),
            Method::GS => String::from("gs"),
            Method::GR => String::from("gr"),
            Method::CG => String::from("cg"),
        }
    }


criterion_group!(benches, vem1_benchmark, vem2_benchmark, spa1_benchmark, spa2_benchmark);
criterion_main!(benches);
