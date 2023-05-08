use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    SamplingMode,
};
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
    group.warm_up_time(Duration::from_secs(100));
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for met in [Method::JA, Method::GS, Method::GR, Method::CG] {
        group.bench_with_input(BenchmarkId::new("-4", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-6", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-8", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-10", met_to_str(&met)), &met, |b, met| {
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
    group.warm_up_time(Duration::from_secs(100));
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for met in [Method::JA, Method::GS, Method::GR, Method::CG] {
        group.bench_with_input(BenchmarkId::new("-4", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-6", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-8", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-10", met_to_str(&met)), &met, |b, met| {
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
    group.warm_up_time(Duration::from_secs(100));
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for met in [Method::JA, Method::GS, Method::GR, Method::CG, Method::PG] {
        group.bench_with_input(BenchmarkId::new("-4", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-6", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-8", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-10", met_to_str(&met)), &met, |b, met| {
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
    group.warm_up_time(Duration::from_secs(100));
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(100));
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    for met in [Method::JA, Method::GS, Method::GR, Method::CG, Method::PG] {
        group.bench_with_input(BenchmarkId::new("-4", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-4), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-6", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-6), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-8", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-8), 20000, 1.0)
            })
        });
        group.bench_with_input(BenchmarkId::new("-10", met_to_str(&met)), &met, |b, met| {
            b.iter(|| {
                solve_linear_system(&matr, &matr_b, met.copy(), 10.0_f64.powi(-10), 20000, 1.0)
            })
        });
    }
}

pub fn met_to_str(met: &Method) -> String {
    match met {
        Method::JA => String::from("JA"),
        Method::GS => String::from("GS"),
        Method::GR => String::from("GR"),
        Method::CG => String::from("CG"),
        Method::PG => String::from("PG"),
    }
}

criterion_group!(
    benches,
    vem1_benchmark,
    vem2_benchmark,
    spa1_benchmark,
    spa2_benchmark,
);
criterion_main!(benches);