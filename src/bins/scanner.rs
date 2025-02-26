//! all this does is get the target file path from args, read the file into a string,
//! send that string to the scanner, and save the output to a file (same name as input, with `_gen` appended)

use compiler::bin_utils::run_scanner;

fn main() -> Result<(), String> {
    // get the target file path from args
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: scanner <file>");
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

    // run the scanner
    run_scanner(&file_path).map_err(|e| e.to_string())
}
