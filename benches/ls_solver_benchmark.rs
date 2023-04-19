use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ls_solver::api::solve_linear_system;

pub fn jacobi_benchmark(c: &mut Criterion) {

    //c.bench_function("Jacobi", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
