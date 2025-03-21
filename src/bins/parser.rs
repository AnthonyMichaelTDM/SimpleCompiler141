//! this binary will parse the provided code, and print out:
//! - whether parsing was successful (and what went wrong if it wasnt')
//! - how many variables are declared
//! - how many functions are declared
//! - how many statements are in the code

use compiler::bin_utils::{run_parser, ParserOutput};

fn main() -> Result<(), String> {
    // get the target file path from args
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: parser <file>");
        return Err("Invalid args".to_string());
    }
    let file_path = &args[1];

    // parse it into a PathBuf
    let file_path = std::path::PathBuf::from(file_path);

    // ensure the file exists
    if !file_path.exists() {
        eprintln!("File not found");
        return Err("File not found".to_string());
    }
    // and that it's a file
    if !file_path.is_file() {
        eprintln!("Not a file");
        return Err("Not a file".to_string());
    }

    let ParserOutput {
        function_count,
        variable_count,
        statement_count,
    } = run_parser(&file_path).map_err(|e| e.to_string())?;

    println!(
        "Parsing successful! Variables: {}, Functions: {}, Statements: {}",
        variable_count, function_count, statement_count
    );

    Ok(())
}
