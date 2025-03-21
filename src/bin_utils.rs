//! Utility functions used for both the binaries, and some tests.

use std::path::Path;

use crate::{
    generic::Scanner,
    language::{CTokenType, RULES},
};

/// Convenience functon that:
/// - takes as input the contents of the file to scan,
/// - runs that input through the scanner,
/// - emits tokens to the output string,
/// - and will check for errors and include an error message in the output string if any are found.
#[must_use]
pub fn scan(input: &str) -> String {
    let scanner = Scanner::new(input, &RULES);
    let mut output = String::new();

    let tokens: Vec<_> = scanner.iter().collect();

    for token in &tokens {
        match token {
            Ok(token) => {
                // add the string "cse141" to the beginning of every <identifier> token except the name of function "main"
                if token.kind == CTokenType::Identifier && token.text != "main" {
                    output += "cse141";
                }
                output += token.text;
            }
            Err(_) => {
                output += "The input program contains errors for scanning.";
                break;
            }
        }
    }

    output
}

/// Convenience function that is basically all the functionality of
/// the main function, but without the command line argument parsing.
/// This is useful for testing.
///
/// # Errors
///
/// Returns an error if there is an issue reading the input file or writing the output file.
pub fn run_scanner(input_file: impl AsRef<Path>) -> Result<(), std::io::Error> {
    let input_file = input_file.as_ref();
    // read the file into a string
    let input = std::fs::read_to_string(input_file)?;

    // send that string to the scanner and get the output
    let output = scan(&input);

    // save the output to a file (same name as input, with `_gen` appended)
    let mut output_path = input_file.to_path_buf();
    output_path.set_file_name(
        input_file
            .file_stem()
            .ok_or(std::io::ErrorKind::InvalidInput)?
            .to_str()
            .ok_or(std::io::ErrorKind::InvalidInput)?
            .to_owned()
            + "_gen.c",
    );
    std::fs::write(&output_path, output)?;

    Ok(())
}
