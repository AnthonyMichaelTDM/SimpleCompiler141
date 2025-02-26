//! This module contains things specific to the token language we're implementing
use crate::generic::TokenRule;
use core::ops::Range;

/// Types of tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTokenType {
    /// A comment or macro statement
    MetaStatement,
    /// A reserved word
    ReservedWord,
    /// An identifier
    Identifier,
    /// A number
    Number,
    /// A symbol
    Symbol,
    /// A string
    String,
    /// Whitespace
    Space,
}

// Types for each token rule
#[derive(Debug)]
pub struct MacroRule;
#[derive(Debug)]
pub struct CommentRule;
#[derive(Debug)]
pub struct ReservedWordRule;
#[derive(Debug)]
pub struct IdentifierRule;
#[derive(Debug)]
pub struct NumberRule;
#[derive(Debug)]
pub struct SymbolRule;
#[derive(Debug)]
pub struct StringRule;
#[derive(Debug)]
pub struct SpaceRule;

/// The rules for tokenizing the input, ordered by precedence
pub static RULES: [&dyn TokenRule<TokenType = CTokenType>; 8] = [
    &MacroRule,
    &CommentRule,
    &ReservedWordRule,
    &IdentifierRule,
    &NumberRule,
    &SymbolRule,
    &StringRule,
    &SpaceRule,
];

/// Reserved words in the language
pub const RESERVED_WORDS: [&str; 12] = [
    "int", "void", "if", "while", "return", "read", "write", "print", "continue", "break",
    "binary", "decimal",
];

/// Symbols in the language
pub const SYMBOLS: [&str; 22] = [
    "(", ")", "{", "}", "[", "]", ",", ";", "+", "-", "*", "/", "==", "!=", ">", ">=", "<", "<=",
    "=", "&&", "||", "'",
];
/// A list of exceptions for two character symbols (or other things that might be mistaken for symbols).
///
/// If the first character(s) of the two character symbol is matched, and the next character(s) would complete the symbol,
/// we should not match the first character(s).
///
/// I say character(s) because the implementation supports exceptions longer than 2 characters, even if none are currently defined.
pub const SYMBOL_EXCEPTIONS: [&str; 8] = [
    "==", "!=", ">=", "<=", "&&", "||", // exceptions for two character symbols
    "/*", "//", // exceptions for comments
];

/////////////////////////////////////////////////
// Implementations
/////////////////////////////////////////////////

impl TokenRule for MacroRule {
    type TokenType = CTokenType;
    /// a macro statement starts with a '#' and ends with a newline
    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        input.starts_with('#') && input.ends_with('\n') && input.len() >= 2
    }

    fn token_type(&self) -> Self::TokenType {
        CTokenType::MetaStatement
    }
}

impl TokenRule for CommentRule {
    type TokenType = CTokenType;
    /// a single-line comment starts with '//' and ends with a newline
    /// a multi-line or inline comment starts with '/*' and ends with '*/'
    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        (input.starts_with("/*") && input.ends_with("*/") && input.len() >= 4) // multi-line or inline comment
         ||(input.starts_with("//") && input.ends_with('\n')) // single-line comment
    }

    fn token_type(&self) -> Self::TokenType {
        CTokenType::MetaStatement
    }
}

impl TokenRule for ReservedWordRule {
    type TokenType = CTokenType;
    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        RESERVED_WORDS.iter().any(|word| input == *word)
    }

    fn token_type(&self) -> Self::TokenType {
        CTokenType::ReservedWord
    }
}

impl TokenRule for NumberRule {
    type TokenType = CTokenType;
    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        input.chars().all(|c| c.is_ascii_digit()) && !input.is_empty()
    }

    fn token_type(&self) -> Self::TokenType {
        CTokenType::Number
    }
}

impl TokenRule for SymbolRule {
    type TokenType = CTokenType;
    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let substr = &input[range.clone()];

        // Handle the symbol exceptions, which are primarily used to avoid incorrectly matching the first character of a two character symbol,
        // and to avoid matching comments as symbols.
        // If substr is a partial match for the start of an exception (but not a complete match), and next character(s) in the string would complete the exception,
        // then we should not match the partial match.
        if SYMBOL_EXCEPTIONS.iter().any(|exception| {
            // the input matches the exception
            input[range.start..].starts_with(*exception) &&
             // but it's not a complete match
             exception.len() > substr.len()
        }) {
            return false;
        }

        SYMBOLS.iter().any(|symbol| substr == *symbol)
    }

    fn token_type(&self) -> Self::TokenType {
        CTokenType::Symbol
    }
}

impl TokenRule for IdentifierRule {
    type TokenType = CTokenType;
    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        let mut chars = input.chars();
        chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
            && chars.all(|c| c.is_alphanumeric() || c == '_')
    }

    fn token_type(&self) -> Self::TokenType {
        CTokenType::Identifier
    }
}

impl TokenRule for StringRule {
    type TokenType = CTokenType;
    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        (input.starts_with('"') && input.ends_with('"') && input.len() >= 2) &&
             // handle case where the string has escaped quotes,
             // since our scanner works by checking increasingly long slices of input we don't need to look through the entire string
             // and only need to check if there are an even number of backslashes before the last quote
             input[..input.len() - 1]
                 .chars()
                 .rev()
                 .take_while(|c| *c == '\\')
                 .count()
                 % 2
                 == 0
    }

    fn token_type(&self) -> Self::TokenType {
        CTokenType::String
    }
}

impl TokenRule for SpaceRule {
    type TokenType = CTokenType;
    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        input.chars().all(char::is_whitespace) && !input.is_empty()
    }

    fn token_type(&self) -> Self::TokenType {
        CTokenType::Space
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::generic::{Scanner, TokenSpan};

    use super::*;

    use pretty_assertions::assert_eq;
    use rstest::rstest;

    #[rstest]
    #[case("int", true)]
    #[case("void", true)]
    #[case("if", true)]
    #[case("while", true)]
    #[case("return", true)]
    #[case("read", true)]
    #[case("write", true)]
    #[case("print", true)]
    #[case("continue", true)]
    #[case("break", true)]
    #[case("binary", true)]
    #[case("decimal", true)]
    #[case("intt", false)]
    #[case("voidd", false)]
    #[case(" if", false)]
    #[case(" 1", false)]
    fn test_reserved_word_rule(#[case] word: &str, #[case] expected: bool) {
        assert_eq!(ReservedWordRule.matches(word, 0..word.len()), expected);
    }

    #[test]
    fn test_symbol_exceptions_are_not_matched() {
        for exception in SYMBOL_EXCEPTIONS.iter() {
            // all the prefixes of the exception fail to match with the SymbolRule
            for i in 1..exception.len() {
                assert!(
                    !SymbolRule.matches(exception, 0..i),
                    "prefix of exception should not match, failed for slice \"{}\" of exception \"{}\"",
                    &exception[..i],
                    exception
                );
            }
        }
    }

    #[rstest]
    #[case::macros("#include <stdio.h>\n", true)]
    #[case::macros("#define read(x) scanf(\"%d\\n\", &x)\n", true)]
    #[case::macros("#define write(x) printf(\"%d\\n\", x)\n", true)]
    #[case("int a;\n", false)]
    #[case("int a;\n", false)]
    fn test_meta_statement_rule(#[case] statement: &str, #[case] expected: bool) {
        assert_eq!(MacroRule.matches(statement, 0..statement.len()), expected,);
    }

    #[rstest]
    #[case::comment("// function foo\n", true)]
    #[case("/* function foo\n", false)]
    #[case("/* function foo\n*/", true)]
    #[case(
        r"/* function foo 
*/",
        true
    )]
    fn test_multiline_comment_rules(#[case] comment: &str, #[case] expected: bool) {
        assert_eq!(CommentRule.matches(comment, 0..comment.len()), expected,);
    }

    #[test]
    fn test_issue_with_foo_comment() {
        let input = "\t// function foo\n";

        let scanner = Scanner::new(input, &RULES);
        let expected_tokens = vec![
            TokenSpan {
                token_type: CTokenType::Space,
                start: 0,
                end: 1,
                text: "\t",
            },
            TokenSpan {
                token_type: CTokenType::MetaStatement,
                start: 1,
                end: 17,
                text: "// function foo\n",
            },
        ];
        assert_eq!(scanner.iter().collect::<Vec<_>>(), expected_tokens);
    }

    #[test]
    fn test_reserved_words() {
        for word in RESERVED_WORDS.iter() {
            let scanner = Scanner::new(word, &RULES);
            let token = scanner.iter().next();
            assert_eq!(token.map(|t| t.token_type), Some(CTokenType::ReservedWord));
        }
    }

    #[rstest]
    #[case(r#""Give me a number: ""#)]
    #[case(r#""Give me a number: \\""#)]
    #[case::escaped_quotes(r#""\"""#)]
    #[case::escaped_quotes(r#""hello \"world\"""#)]
    #[should_panic]
    #[case(r#""Give me a number: \\\""#)]
    #[case(r#""Give me a number: \\\"""#)]
    #[should_panic]
    #[case(r#""Give me a number:\ \\\""#)]
    #[case(r#""Give me a number:\ \\\"""#)]
    #[case(r#""Give me a number:\\\\""#)]
    #[case::multiline_string(
        r#""hello\
    world""#
    )]
    fn test_string_scanning(#[case] input: &str) {
        let scanner = Scanner::new(input, &RULES);
        let expected_tokens = vec![TokenSpan {
            token_type: CTokenType::String,
            start: 0,
            end: input.len(),
            text: input,
        }];
        assert_eq!(scanner.iter().collect::<Vec<_>>(), expected_tokens);
    }

    #[test]
    // from automaton.c
    fn test_scan_assignment() {
        let input = "\ta = getnextdigit();";

        let scanner = Scanner::new(input, &RULES);
        let expected_tokens = vec![
            TokenSpan {
                token_type: CTokenType::Space,
                start: 0,
                end: 1,
                text: "\t",
            },
            TokenSpan {
                token_type: CTokenType::Identifier,
                start: 1,
                end: 2,
                text: "a",
            },
            TokenSpan {
                token_type: CTokenType::Space,
                start: 2,
                end: 3,
                text: " ",
            },
            TokenSpan {
                token_type: CTokenType::Symbol,
                start: 3,
                end: 4,
                text: "=",
            },
            TokenSpan {
                token_type: CTokenType::Space,
                start: 4,
                end: 5,
                text: " ",
            },
            TokenSpan {
                token_type: CTokenType::Identifier,
                start: 5,
                end: 17,
                text: "getnextdigit",
            },
            TokenSpan {
                token_type: CTokenType::Symbol,
                start: 17,
                end: 18,
                text: "(",
            },
            TokenSpan {
                token_type: CTokenType::Symbol,
                start: 18,
                end: 19,
                text: ")",
            },
            TokenSpan {
                token_type: CTokenType::Symbol,
                start: 19,
                end: 20,
                text: ";",
            },
        ];
        assert_eq!(scanner.iter().collect::<Vec<_>>(), expected_tokens);
    }

    #[rstest]
    #[case("==")]
    #[case("!=")]
    #[case(">=")]
    #[case("<=")]
    #[case("&&")]
    #[case("||")]
    fn test_two_character_symbols(#[case] symbol: &str) {
        let scanner = Scanner::new(symbol, &RULES);
        let tokens = scanner.iter().collect::<Vec<_>>();
        assert_eq!(tokens.len(), 1);
        assert_eq!(
            tokens,
            vec![TokenSpan {
                token_type: CTokenType::Symbol,
                start: 0,
                end: symbol.len(),
                text: symbol,
            }]
        );
    }
}
