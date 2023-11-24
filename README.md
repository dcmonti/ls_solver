# ls_solver
A simple tool and library for linear system solution



## Introduction
The goal of this project is to provide a purely Rust implementation [rust-lang](https://www.rust-lang.org) of a tool for solving linear systems through iterative methods. The resulting tool, `ls_solver`, can take a matrix A and an optional vector b as input and then compute the vector of solutions for the system Ax=b.

To better adapt to user needs, `ls_solver` has a dual interface, making it usable both as a stand-alone command-line tool and as a library within larger projects. In addition, there are several optional parameters easily configurable by the user based on their needs. A more detailed description of these features will be presented in the following sections.

This report will outline the implementation features of `ls_solver`, as well as the results obtained from tests and provide a brief explanation of its functionality.

All materials in this document, such as diagrams or tables of results, as well as `ls_solver` itself, are available at [https://github.com/dcmonti/ls_solver](https://github.com/dcmonti/ls_solver).

## Ls_solver
The only requirement for using `ls_solver` is to have Rust version 2021 (or later) and its package manager Cargo (available for download [here](https://www.rust-lang.org/tools/install)).

Once this requirement is met, simply clone the repository and proceed with compilation using the following commands:
```bash
git clone https://github.com/dcmonti/ls_solver
cd ls_solver
cargo build --release
```

For maximizing the performance of `ls_solver` and if portability is not an issue, you can use:
```bash
RUSTFLAGS="-C target-cpu=native"
```

The executable can now be found in `target/release`.

### Operation
Since `ls_solver` can be used both as a library and as a stand-alone tool, both modes of usage will be explained.

#### Command-line Usage
To use `ls_solver` from the command line, simply run from the project directory:
```bash
target/release/ls_solver [MATRIX_PATH] <OPT_ARGS>
```
Here, `[MATRIX_PATH]` is the path to the file in `.mtx` format containing the matrix. The matrix must adhere to the coordinate format of the matrix market (consultable [here](https://math.nist.gov/MatrixMarket/formats.html)).

`<OPT_ARGS>` are the parameters that the user can specify according to their needs. For example, to calculate the precision of the Jacobi method in solving the matrix in the file `example.mtx` with a tolerance of $10^{-6}$, the command would be:
```bash
target/release/ls_solver /path/to/example.mtx -m 0 -t 6
```

For a detailed description of all options and possible values, execute:
```bash
target/release/ls_solver --help
```
This will display the following screen:

```markdown
Usage: ls_solver [OPTIONS] <MATRIX_PATH> [VECTOR_PATH]

I/O:
  -o, --output <OUTPUT_PATH>
          Output file with approximate solution,
          if None solution will not be printed
          [default: None]

  <MATRIX_PATH>
          Input matrix (in .mtx format)

  [VECTOR_PATH]
          Input vector (in .mtx format).
          If not specified x := [1 1 ... 1] and b := ax
          [default: None]

Settings:
  -m, --method <METHOD>
          0: Jacobi
          1: Gauß-Seidel
          2: gradient
          3: conjugate gradient
          4: Jacobi-preconditioned gradient (only if matrix is SPD)
          [default: 0]

  -t, --tolerance <TOLERANCE>
          Set tolerance as desired negative exponent (e.g. 4 is 0.0001)
          [default: 4]

  -i, --max_iter <MAX_ITER>
          Set max number of iterations for the routine
          [default: 20000]

  -O, --omega <OMEGA>
          Set relax factor with float desired
          Used only if method is 0 or 1
          [default: 1]

  -s, --set-mode <SETTING>
          Used only if [VECTOR_PATH] is specified
          0: consider vector as b and solve the system Ax=b
          1: consider vector as x and evaluate method precision
          [default: 0]
```

#### Library Usage
`ls_solver` can be used as a library in other Rust projects by adding the following dependency to the `Cargo.toml` file:
```toml
[dependencies]
ls_solver = { git = "https://github.com/dcmonti/ls_solver" }
```

Then, `ls_solver` can be easily used within the project by adding the following line to the files where the library is needed:
```rust
use ls_solver::api::*;
```

Detailed explanations of these functions are available in the documentation, which can be found directly in the project folder (`doc/ls_solver/api/index.html`).

## Acknowledgments
 [Nalgebra](https://github.com/dimforge/nalgebra)  
 [Clap](https://github.com/clap-rs/clap)

## Contributors
Davide Cesare Monti ([https://github.com/dcmonti](https://github.com/dcmonti))  
Samuele Campanella ([https://github.com/kmp222](https://github.com/kmp222))
