use std::path::{Path, PathBuf};

use rstest::rstest;

use compiler::bin_utils::{parse, ParserOutput};
use compiler::generic::Scanner;
use compiler::language::RULES;

#[rstest]
#[case::automaton("automaton.c", Ok((5, 6, 48)))]
#[case::binrep("binrep.c", Ok((2, 2, 21)))]
#[case::branch("branch.c", Ok((2, 1, 6)))]
#[case::error_string("errorString.c", Err("Scanner error: Illegal character '%' found at position 1".to_string()))]
#[case::expression("expression.c", Ok((2, 1, 3)))]
#[case::fibonacci("fibonacci.c", Ok((5, 3, 17)))]
#[case::formulae("formulae.c", Ok((12, 2, 31)))]
#[case::funcall("funcall.c", Ok((1, 8, 10)))]
#[case::funcall2("funcall2.c", Ok((1, 5, 7)))]
#[case::illegal("illegal.c", Ok((4, 2, 15)))]
#[case::loop_("loop.c", Ok((2, 1, 6)))]
#[case::mandel("mandel.c", Ok((8, 6, 34)))]
#[case::max("max.c", Ok((2, 2, 6)))]
#[case::parameter("parameter.c", Ok((1, 2, 5)))]
#[case::parse("parse.c", Err("Parse error: No production found for non-terminal Statement0 and lookahead Some(OwnedTokenSpan { kind: Symbol, start: 348, end: 350, text: \"==\", is_eof: false, is_whitespace: false }) (which is a DoubleEqual)".to_string()))]
#[case::parse2("parse2.c", Err("Scanner error: Illegal character '@' found at position 120".to_string()))]
#[case::polymorphism("polymorphism.c", Ok((2, 2, 6)))]
#[case::recursion("recursion.c", Ok((1, 2, 5)))]
#[case::sort("sort.c", Ok((9, 3, 127)))]
#[case::square("square.c", Ok((2, 2, 6)))]
#[case::tax("tax.c", Ok((24, 2, 87)))]
#[case::times8("times8.c", Ok((2, 3, 4)))]
#[case::trivial("trivial.c", Ok((1, 2, 3)))]
fn test_parser(
    #[case] file: PathBuf,
    #[case] expected: Result<(usize, usize, usize), String>,
    #[values(true, false)] with_ll1_parser: bool,
) {
    let path = Path::new("test_files").join(file);
    let input = std::fs::read_to_string(path).expect("Failed to read file");
    let scanner = Scanner::new(&input, &RULES);
    let parser_output = parse(scanner, with_ll1_parser);

    match expected {
        Ok((expected_variable_count, expected_function_count, expected_statement_count)) => {
            let expected = ParserOutput {
                variable_count: expected_variable_count,
                function_count: expected_function_count,
                statement_count: expected_statement_count,
            };

            assert_eq!(parser_output, Ok(expected));
        }
        Err(e) => {
            assert_eq!(parser_output.map_err(|e| e.to_string()), Err(e));
        }
    }
}
