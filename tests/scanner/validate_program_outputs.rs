//! All the other tests just validate the output of the scanner,
//! These tests ensure that the file created by the scanner can be compiled and that it's output matches the output of the original file.
//!
//! We can only test binaries that don't expect to read from stdin, as we can't provide input to the generated binaries.

use super::run;
use pretty_assertions::assert_str_eq;
use rstest::rstest;

#[rstest]
#[case::fibonacci("fibonacci")]
#[case::funcall("funcall")]
#[case::funcall2("funcall2")]
#[case::illegal("illegal")]
#[case::mandel("mandel")]
#[case::parse("parse")]
#[case::sort("sort")]
fn test_binary(#[case] file: &str) {
    let path = format!("test_files/{file}.c");
    let path = std::path::Path::new(&path);
    // run the scanner on the file to create a new file
    run(path).expect("Failed to run scanner");

    // compile the test file
    let make_output = std::process::Command::new("make")
        .current_dir("./test_files")
        .arg(file)
        .output()
        .expect("Failed to run make");
    assert!(make_output.status.success());
    // compile the generated file
    let make_output = std::process::Command::new("make")
        .current_dir("./test_files")
        .arg(format!("{file}_gen"))
        .output()
        .expect("Failed to run make");
    assert!(make_output.status.success());

    // run the compiled test file
    let expected = std::process::Command::new(format!("test_files/{file}"))
        .output()
        .expect("Failed to run test file");
    let expected = String::from_utf8(expected.stdout).expect("Failed to convert output to string");

    // run the compiled generated file
    let output = std::process::Command::new(format!("test_files/{file}_gen"))
        .output()
        .expect("Failed to run generated file");
    let output = String::from_utf8(output.stdout).expect("Failed to convert output to string");

    // // clean up the generated files
    std::fs::remove_file(format!("test_files/{file}")).expect("Failed to remove test file binary");
    std::fs::remove_file(format!("test_files/{file}_gen"))
        .expect("Failed to remove generated file binary");
    std::fs::remove_file(format!("test_files/{file}_gen.c"))
        .expect("Failed to remove generated file");

    assert_str_eq!(expected, output);
}
