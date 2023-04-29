use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use ls_solver::{
    api::{solve_linear_system, Method},
    utility::init_b,
};
use nalgebra::DVector;
use nalgebra_sparse::{io::load_coo_from_matrix_market_file, CsrMatrix};

pub fn vem1_benchmark(c: &mut Criterion) {
    let matr: CsrMatrix<f64> = CsrMatrix::from(
        &load_coo_from_matrix_market_file("benches/test_matrices/vem1.mtx").unwrap(),
    );

    let matr_solution = DVector::from_element(matr.nrows(), 1.0);

    let matr_b = init_b(&matr_solution, &matr);

    let mut group = c.benchmark_group("vem1");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));

    for met in [Method::JA, Method::GS, Method::GR, Method::CG] {
        group.bench_with_input(BenchmarkId::new("4", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("6", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("8", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("10", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-10), 20000, 1.0)
            })
        });
    }
}

pub fn vem2_benchmark(c: &mut Criterion) {
    let matr: CsrMatrix<f64> = CsrMatrix::from(
        &load_coo_from_matrix_market_file("benches/test_matrices/vem2.mtx").unwrap(),
    );

    let matr_solution = DVector::from_element(matr.nrows(), 1.0);

    let matr_b = init_b(&matr_solution, &matr);

    let mut group = c.benchmark_group("vem2");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));

    for met in [Method::JA, Method::GS, Method::GR, Method::CG] {
        group.bench_with_input(BenchmarkId::new("4", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("6", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("8", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("10", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-10), 20000, 1.0)
            })
        });
    }
}

pub fn spa1_benchmark(c: &mut Criterion) {
    let matr: CsrMatrix<f64> = CsrMatrix::from(
        &load_coo_from_matrix_market_file("benches/test_matrices/spa1.mtx").unwrap(),
    );

    let matr_solution = DVector::from_element(matr.nrows(), 1.0);

    let matr_b = init_b(&matr_solution, &matr);

    let mut group = c.benchmark_group("spa1");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));

    for met in [Method::JA, Method::GS, Method::CG] {
        group.bench_with_input(BenchmarkId::new("4", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("6", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("8", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("10", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-10), 20000, 1.0)
            })
        });
    }
}

pub fn spa2_benchmark(c: &mut Criterion) {
    let matr: CsrMatrix<f64> = CsrMatrix::from(
        &load_coo_from_matrix_market_file("benches/test_matrices/spa2.mtx").unwrap(),
    );

    let matr_solution = DVector::from_element(matr.nrows(), 1.0);

    let matr_b = init_b(&matr_solution, &matr);

    let mut group = c.benchmark_group("spa2");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));
    for met in [Method::JA, Method::GS, Method::CG] {
        group.bench_with_input(BenchmarkId::new("4", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("6", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("8", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("10", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-10), 20000, 1.0)
            })
        });
    }
}

fn precond_vs_gradient_spa1(c: &mut Criterion) {
    let spa1: CsrMatrix<f64> = CsrMatrix::from(
        &load_coo_from_matrix_market_file("benches/test_matrices/spa1.mtx").unwrap(),
    );

    let spa1_solution = DVector::from_element(spa1.nrows(), 1.0);

    let spa1_b = init_b(&spa1_solution, &spa1);

    let mut group = c.benchmark_group("p_gra-gra(spa1)");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));
    group.warm_up_time(Duration::from_secs(100));
    for met in [Method::GR, Method::PG] {
        group.bench_with_input(BenchmarkId::new("4", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&spa1, &spa1_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("6", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&spa1, &spa1_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("8", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&spa1, &spa1_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("10", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&spa1, &spa1_b, met.copy(), 10.0_f64.powi(-10), 20000, 1.0)
            })
        });
    }
}

fn precond_vs_gradient_spa2(c: &mut Criterion) {
    let spa1: CsrMatrix<f64> = CsrMatrix::from(
        &load_coo_from_matrix_market_file("benches/test_matrices/spa2.mtx").unwrap(),
    );

    let spa1_solution = DVector::from_element(spa1.nrows(), 1.0);

    let spa1_b = init_b(&spa1_solution, &spa1);

    let mut group = c.benchmark_group("p_gra-gra(spa2)");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));
    group.warm_up_time(Duration::from_secs(100));

    for met in [Method::GR, Method::PG] {
        group.bench_with_input(BenchmarkId::new("4", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&spa1, &spa1_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("6", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&spa1, &spa1_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("8", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&spa1, &spa1_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("10", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&spa1, &spa1_b, met.copy(), 10.0_f64.powi(-10), 20000, 1.0)
            })
        });
    }
}

pub fn met_to_str(met: &Method) -> String {
    match met {
        Method::JA => String::from("ja"),
        Method::GS => String::from("gs"),
        Method::GR => String::from("gr"),
        Method::CG => String::from("cg"),
        Method::PG => String::from("pg"),
    }
}

criterion_group!(
    benches,
    vem1_benchmark,
    vem2_benchmark,
    spa1_benchmark,
    spa2_benchmark, 
    precond_vs_gradient_spa1,
    precond_vs_gradient_spa2
);
criterion_main!(benches);
