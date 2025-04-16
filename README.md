# Compiler (series of labs for CSE141)

This is a collection of a series of labs that I have done for my CSE141 class.

If you want to use this as the basis for teaching a class, feel free to do so. I would appreciate it if you could let me know if you do, but it's not required.

I'd recommend leaving the code in `src/generic` as is, and having students re-implement parts of the modules in `src/langage` as they go through the labs.

## Usage

> Note:
> All of these commands are to be run from the root of the project (the same directory as this file).

### Prerequisites

In order to compile this, you'll first need to install rust. You can do this by following the instructions on the [official rust website](https://www.rust-lang.org/tools/install).

If you're on a unix-like system, you can use the following command to install the rust toolchain:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After installing rust, you'll need the `cargo` package manager, but this should be installed automatically with rust. Just run the following to make sure everything is up to date:

```bash
rustup update
```

After installing, you can check that everything is installed correctly by running:

```bash
rustc --version
cargo --version
```

### Compiling

To compile the code, simply run the following command (from the root of the project (same directory as this file)):

```bash
cargo build --release
```

This will compile the scanner, parser, and other binaries and place them in `target/release`.

### Running

#### Scanner (Lab 1)

To run the scanner:

```bash
cargo run --bin scanner --release -- <input_file>
```

Where `<input_file>` is the path to the file you want to scan.

output will be the same code, with "cse141" prepended to all identifiers

#### Parser (Lab 2)

To run the parser:

```bash
cargo run --bin parser --release -- <input_file>
```

Where `<input_file>` is the path to the file you want to parse.

Output will report the number of variables, functions, and statements in the file.

#### AST Printer (Lab 3)

To run the AST printer:

```bash
cargo run --bin gee --release -- <input_file>
```

Where `<input_file>` is the path to the `.gee` file you want to parse

### Running tests

To run the tests, you can use the following command:

```bash
cargo test
```

This will run a suite of unit and integration tests that test the scanner against the provided test files.
