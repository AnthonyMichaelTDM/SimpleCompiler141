//! This module contains generic library code for constructing
//! scanners, parsers, and eventually other compiler components.
//!
//! `crate::langauge` contains the language-specific implementations
//! of these components.

mod parser;
mod scanner;
pub use parser::*;
pub use scanner::*;

#[cfg(test)]
pub mod test_utils {
    use rstest::fixture;

    use crate::derivation;

    // contains grammars and fixtures used for multiple tests
    use super::{
        grammar::{Grammar, NonTerminating},
        *,
    };

    #[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
    #[repr(usize)]
    pub enum ExprNT {
        Goal,
        Expr,
        Term,
        Factor,
        // new symbols get added to the end, and we already know what order they'll be added in,
        // so i put them in the Enum for better readability
        ExprPrime,
        TermPrime,
    }
    impl From<ExprNT> for usize {
        fn from(nt: ExprNT) -> Self {
            nt as usize
        }
    }

    impl NonTerminal for ExprNT {}

    #[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
    pub enum ExprT {
        Num,
        Name,
        Plus,
        Minus,
        Div,
        Mult,
        LeftParen,
        RightParen,
        Eof,
        Whitespace,
    }

    impl Terminal for ExprT {
        fn eof() -> Self {
            Self::Eof
        }
    }

    #[fixture]
    pub fn expr_grammar() -> Grammar<ExprNT, ExprT, NonTerminating> {
        Grammar::new(
            ExprNT::Goal,
            vec![
                Production::new(ExprNT::Goal, derivation![ExprNT::Expr]),
                Production::new(
                    ExprNT::Expr,
                    derivation![ExprNT::Expr, ExprT::Plus, ExprNT::Term],
                ),
                Production::new(
                    ExprNT::Expr,
                    derivation![ExprNT::Expr, ExprT::Minus, ExprNT::Term],
                ),
                Production::new(ExprNT::Expr, derivation![ExprNT::Term]),
                Production::new(
                    ExprNT::Term,
                    derivation![ExprNT::Term, ExprT::Mult, ExprNT::Factor],
                ),
                Production::new(
                    ExprNT::Term,
                    derivation![ExprNT::Term, ExprT::Div, ExprNT::Factor],
                ),
                Production::new(ExprNT::Term, derivation![ExprNT::Factor]),
                Production::new(
                    ExprNT::Factor,
                    derivation![ExprT::LeftParen, ExprNT::Expr, ExprT::RightParen],
                ),
                Production::new(ExprNT::Factor, derivation![ExprT::Num]),
                Production::new(ExprNT::Factor, derivation![ExprT::Name]),
            ],
        )
        .unwrap()
    }

    /// The Expression grammar is simple enough we can use the same set of tokens
    /// for the scanner as we do for the parser.
    pub static EXPR_RULES: [&dyn TokenRule<TokenType = ExprT>; 9] = [
        &CharRule('(', ExprT::LeftParen),
        &CharRule(')', ExprT::RightParen),
        &CharRule('+', ExprT::Plus),
        &CharRule('-', ExprT::Minus),
        &CharRule('*', ExprT::Mult),
        &CharRule('/', ExprT::Div),
        &NumRule,
        &NameRule,
        &WhitespaceRule,
    ];

    #[derive(Debug)]
    struct NameRule;
    #[derive(Debug)]
    struct NumRule;
    #[derive(Debug)]
    struct CharRule(char, ExprT);
    #[derive(Debug)]
    struct WhitespaceRule;

    impl TokenRule for CharRule {
        type TokenType = ExprT;

        fn matches(&self, input: &str, range: std::ops::Range<usize>) -> bool {
            let str = &input[range];
            str.len() == 1 && str.chars().next().unwrap() == self.0
        }

        fn token_type(&self) -> Self::TokenType {
            self.1
        }
    }

    impl TokenRule for NameRule {
        type TokenType = ExprT;

        fn matches(&self, input: &str, range: std::ops::Range<usize>) -> bool {
            let str = &input[range];
            str.chars().all(|c| c.is_alphabetic() || c == '_') && !str.is_empty()
        }

        fn token_type(&self) -> Self::TokenType {
            ExprT::Name
        }
    }

    impl TokenRule for NumRule {
        type TokenType = ExprT;

        fn matches(&self, input: &str, range: std::ops::Range<usize>) -> bool {
            let str = &input[range];
            str.chars().all(|c| c.is_numeric()) && !str.is_empty()
        }

        fn token_type(&self) -> Self::TokenType {
            ExprT::Num
        }
    }

    impl TokenRule for WhitespaceRule {
        type TokenType = ExprT;

        fn matches(&self, input: &str, range: std::ops::Range<usize>) -> bool {
            let str = &input[range];
            str.chars().all(|c| c.is_whitespace()) && !str.is_empty()
        }

        fn token_type(&self) -> Self::TokenType {
            ExprT::Whitespace
        }

        fn is_whitespace(&self) -> bool {
            true
        }
    }

    pub fn expr_scanner(input: &str) -> Scanner<ExprT> {
        Scanner::new(input, &EXPR_RULES)
    }
}
