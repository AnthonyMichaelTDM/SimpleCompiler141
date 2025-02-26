use std::path::Path;

use compiler::{
    generic::{Scanner, TokenSpan},
    language::{CTokenType, RULES},
};

mod automaton;
mod binrep;
mod branch;
mod error_string;
mod expression;
mod fibonacci;
mod formulae;
mod funcall;
mod funcall2;
mod illegal;
mod loop_;
mod mandel;
mod max;
mod parameter;
mod parse;
mod parse2;
mod polymorphism;
mod recursion;
mod sort;
mod square;
mod tax;
mod times8;
mod trivial;
mod validate_program_outputs;

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
        // add the string "cse141" to the beginning of every <identifier> token except the name of function "main"
        if token.token_type == CTokenType::Identifier && token.text != "main" {
            output += "cse141";
        }
        output += token.text;
    }

    // emit an error message if we didn't match the entire input
    if !matches!(tokens.last(), Some(TokenSpan { end, .. }) if *end == input.len()) {
        output += "The input program contains errors for scanning.";
    }
    // assert that the tokens are contiguous and cover the entire input
    debug_assert!(tokens.windows(2).all(|w| w[0].end == w[1].start));

    output
}

/// Convenience function that is basically all the functionality of
/// the main function, but without the command line argument parsing.
/// This is useful for testing.
///
/// # Errors
///
/// Returns an error if there is an issue reading the input file or writing the output file.
pub fn run(input_file: impl AsRef<Path>) -> Result<(), std::io::Error> {
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
