//! Utility functions used for both the binaries, and some tests.

use std::path::Path;

use crate::{
    generic::{Error, LL1Parser, ParseTree, ParseTreeNode, Parser, Scanner},
    language::{c_grammar, CNonTerminal, CTerminal, CTokenType, RULES},
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
/// the scanner's main function, but without the command line argument parsing.
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

#[derive(Debug, PartialEq)]
pub struct ParserOutput {
    pub variable_count: usize,
    pub function_count: usize,
    pub statement_count: usize,
}

/// This function will parse the tokens emitted from the given scanner, and return a count of
/// - how many variables are declared
/// - how many functions are declared
/// - how many statements are in the code
///
/// # Errors
///
/// Returns an error if there is an issue parsing the token stream.
pub fn parse<'a>(
    scanner: Scanner<'a, CTokenType>,
    with_ll1_parser: bool,
) -> Result<ParserOutput, Error<CNonTerminal, CTerminal, CTokenType>> {
    let parse_tree = if with_ll1_parser {
        print!("Constructing grammar... ");
        let grammar = c_grammar();
        let terminating = grammar.check_terminating()?;
        let ready = terminating.generate_sets();
        let ll1 = ready.check_ll1()?;
        println!("done");

        print!("Generating parse tables... ");
        let parse_table = ll1.generate_parse_table();
        let parser = LL1Parser::new(parse_table);
        println!("done");

        print!("Parsing... ");
        let parse_tree: ParseTree<'a, CNonTerminal, CTerminal, CTokenType> =
            parser.parse(scanner)?;
        println!("done");
        parse_tree
    } else {
        print!("Constructing grammar... ");
        let grammar = c_grammar();
        let terminating = grammar.check_terminating()?;
        let ready = terminating.generate_sets();
        println!("done");

        print!("Generating parse tables... ");
        let parse_table = ready.generate_parse_table();
        let parser = Parser::new(parse_table);
        println!("done");

        print!("Parsing... ");
        let parse_tree: ParseTree<'a, CNonTerminal, CTerminal, CTokenType> =
            parser.parse(scanner)?;
        println!("done");
        parse_tree
    };

    print!("Traversing parse tree... ");
    let mut variable_count = 0;
    let mut function_count = 0;
    let mut statement_count = 0;
    parse_tree.visit_pre_order(&mut |node| {
        match node {
            ParseTree::Node(ParseTreeNode { non_terminal, .. }) => {
                match non_terminal {
                    // variable
                    CNonTerminal::Id0 => {
                        variable_count += 1;
                    }
                    // function
                    CNonTerminal::Func2 | CNonTerminal::Func5 => {
                        function_count += 1;
                    }
                    // statement
                    CNonTerminal::Statement => {
                        statement_count += 1;
                    }
                    _ => {}
                }
            }
            // we don't care about other nodes
            _ => {}
        }
    });
    println!("done");

    Ok(ParserOutput {
        variable_count,
        function_count,
        statement_count,
    })
}

/// Convenience function that is basically all the functionality of
/// the parsers's main function, but without the command line argument parsing.
/// This is useful for testing.
///
/// # Errors
///
/// Returns an error if there is an issue parsing the input file.
pub fn run_parser(
    input_file: impl AsRef<Path>,
) -> Result<ParserOutput, Error<CNonTerminal, CTerminal, CTokenType>> {
    print!("Scanning file... ");
    let input = std::fs::read_to_string(&input_file)?;
    let scanner = Scanner::new(&input, &RULES);
    println!("done");

    parse(scanner, true)
}
