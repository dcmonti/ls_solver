fn main() {
    cc::Build::new()
        .file("src/sparse_matrix_product.c")
        //.cpp_set_stdlib("stdc++")
        .compile("sparse_matrix_product");
}