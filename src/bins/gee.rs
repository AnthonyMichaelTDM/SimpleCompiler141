//! This is an implementation of lab 3, which is a simple parser for the
//! Gee programming language.
//!
//! The task is to implement a program that parses the input according to the grammar of Gee
//! and produces a printed representation of the abstract syntax tree (AST).
//!
//! For reference, the grammar of Gee is as follows:
//! ```bnf
//! # Gee grammar in Wirth-style BNF, fall 2009
//! # adapted from Clite grammar, revised, Oct. 2 2008
//! # last modified Sept 26 2013
//!
//! #  expression operators
//! relation    = "==" | "!=" | "<" | "<=" | ">" | ">="
//!
//! #  expressions
//! expression = andExpr { "or" andExpr }
//! andExpr    = relationalExpr { "and" relationalExpr }
//! relationalExpr = addExpr [ relation addExpr ]
//! addExpr    = term { ("+" | "-") term }
//! term       = factor { ("*" | "/") factor }
//! factor     = number | string | ident |  "(" expression ")"
//!             
//! # statements
//! stmtList =  {  statement  }
//! statement = ifStatement |  whileStatement  |  assign
//! assign = ident "=" expression  eoln
//! ifStatement = "if" expression block   [ "else" block ]
//! whileStatement = "while"  expression  block
//! block = ":" eoln indent stmtList undent
//!
//! #  goal or start symbol
//! script = stmtList
//! ```
//!
//! but this grammar is an ambiguous grammar, so let's first transform it to remove the ambiguity.
//!
//! ```bnf
//! #  expression operators
//! relation    = "==" | "!=" | "<" | "<=" | ">" | ">="
//!
//! #  expressions
//! expression = relationalExpr expressionTail
//! expressionTail = "or" relationalExpr | "and" relationalExpr | ε
//! relationalExpr = addExpr relationExprTail
//! relationExprTail = relation addExpr | ε
//! addExpr    = term addExprTail
//! addExprTail = "+" term | "-" term | ε
//! term       = factor termTail
//! termTail = "*" factor | "/" factor | ε
//! factor     = number | string | ident |  "(" expression ")"
//!
//! # statements
//! stmtList =  statement stmtListRest
//! stmtListRest = statement stmtListRest | ε
//! statement = ifStatement |  whileStatement  |  assign
//! assign = ident "=" expression  eoln
//! ifStatement = "if" expression block elseStatement
//! elseStatement = "else" block | ε
//! whileStatement = "while"  expression  block
//! block = ":" eoln indent stmtList undent
//!
//! #  goal or start symbol
//! script = stmtList
//!
//! ```
//!
//!
//! # Strategy
//!
//! Trying to parse the Gee language directly is tricky since, like python, scope is defined by indentation.
//!
//! The current general purpose scanner is stateless, so it can't track this behavior.
//! Instead, we can first run the input through a hand-coded "preprocessor" that transforms the input into a form
//! that the scanner can understand.

use std::{fmt::Write, ops::Range};

use compiler::{
    derivation,
    generic::{
        grammar::{Grammar, TerminatingReady},
        Error, EulerTraversalVisit, NonTerminal, ParseTree, ParseTreeLeaf, ParseTreeNode, Parser,
        Production, Scanner, Terminal, TokenConversionError, TokenRule, TokenSpan,
    },
};

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GeeError {
    #[error("Library error: {0}")]
    LibraryError(#[from] Error<GeeNT, GeeT, GeeToken>),
    #[error("Uneven indentation on line {line}, expected {expected} tokens of indentation, but found {found}")]
    UnevenIndentation {
        line: usize,
        expected: usize,
        found: usize,
    },
    #[error("Don't mix tabs and spaces for indentation, line {0}")]
    MixedIndentation(usize),
}

/// A preprocessor for the Gee language.
///
/// This function takes a string input and transforms it into a form that can be parsed by the scanner,
/// it does this by replacing the indentation-based scoping with explicit indent and undent tokens.
///
/// # Errors
///
/// Returns an error if the input has uneven indentation or mixed indentation (tabs and spaces).
pub fn gee_preprocessor(input: &str) -> Result<String, GeeError> {
    const EOLN: char = ';';
    const INDENT: char = '@';
    const UNDENT: char = '~';

    let mut output = String::new();
    let mut indent_level = 0;
    // a stack holding the indent size at each level
    let mut indent_stack = vec![0];

    let mut indentation_char = Option::None;

    for (i, line) in input.lines().enumerate() {
        let trimmed_line = line.trim_start();

        if trimmed_line.is_empty() {
            // empty line
            output.push('\n');
            continue;
        } else if trimmed_line.len() >= line.len() {
            // no indent, undent to the root level
            while indent_level > 0 {
                output.push(UNDENT);
                indent_stack.pop();
                indent_level -= 1;
            }
        } else {
            // line indentation has changed
            let indent_size = line.len() - trimmed_line.len();
            debug_assert!(!indent_stack.is_empty());
            match indent_size.cmp(&indent_stack[indent_level]) {
                std::cmp::Ordering::Greater => {
                    // check for mixed indentation
                    if indentation_char.is_none() {
                        indentation_char = line.chars().next();
                    } else if line.chars().next() != indentation_char {
                        return Err(GeeError::MixedIndentation(i + 1));
                    }

                    // Indent
                    while indent_size > indent_stack[indent_level] {
                        output.push(INDENT);
                        indent_stack.push(indent_size);
                        indent_level += 1;
                    }
                }
                std::cmp::Ordering::Less => {
                    // Undent
                    while !indent_stack.is_empty() && indent_size < indent_stack[indent_level] {
                        indent_stack.pop();
                        output.push(UNDENT);
                        indent_level -= 1;
                    }

                    if indent_stack.is_empty() {
                        return Err(GeeError::UnevenIndentation {
                            line: i + 1,
                            expected: 0,
                            found: indent_size,
                        });
                    }
                    let expected = indent_stack[indent_level];
                    if indent_size != expected {
                        return Err(GeeError::UnevenIndentation {
                            line: i + 1,
                            expected,
                            found: indent_size,
                        });
                    }
                }
                std::cmp::Ordering::Equal => {
                    // no change
                }
            }
        }

        output.push_str(trimmed_line.trim_end());
        output.push(EOLN);
        output.push('\n');
    }

    // // Final undent to close any remaining indentation
    // while indent_level > 0 {
    //     output.push(Undent);
    //     indent_level -= 1;
    // }
    Ok(output.trim().to_string())
}

/// A token in the grammar of Gee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeeToken {
    Comment,
    ReservedWord,
    Identifier,
    Number,
    Symbol,
    String,
    Whitespace,
    Indent,
    Undent,
    Eoln,
}

struct CommentRule;
struct ReservedWordRule;
struct IdentifierRule;
struct NumberRule;
struct SymbolRule;
struct StringRule;
struct WhitespaceRule;
struct IndentRule;
struct UndentRule;
struct EolnRule;

/// The rules for tokenizing the input, ordered by precedence
pub static RULES: [&dyn TokenRule<TokenType = GeeToken>; 10] = [
    &IndentRule,
    &UndentRule,
    &EolnRule,
    &CommentRule,
    &ReservedWordRule,
    &IdentifierRule,
    &NumberRule,
    &SymbolRule,
    &StringRule,
    &WhitespaceRule,
];

/// Reserved words in the language
pub const RESERVED_WORDS: [&str; 5] = ["if", "else", "while", "or", "and"];

/// Symbols in the language
pub const SYMBOLS: [&str; 14] = [
    "(", ")", "+", "-", "*", "/", "=", "<", ">", "==", "!=", "<=", ">=", ":",
];

/// A list of exceptions for two character symbols (or other things that might be mistaken for symbols).
///
/// If the first character(s) of the two character symbol is matched, and the next character(s) would complete the symbol,
/// we should not match the first character(s).
///
/// I say character(s) because the implementation supports exceptions longer than 2 characters, even if none are currently defined.
pub const SYMBOL_EXCEPTIONS: [&str; 4] = ["==", "!=", ">=", "<="];

impl TokenRule for CommentRule {
    type TokenType = GeeToken;
    /// a macro statement starts with a '#' and ends with a newline
    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        input.starts_with('#') && input.ends_with('\n') && input.len() >= 2
    }

    fn token_type(&self) -> Self::TokenType {
        GeeToken::Comment
    }

    fn is_whitespace(&self) -> bool {
        true
    }
}

impl TokenRule for ReservedWordRule {
    type TokenType = GeeToken;

    /// A reserved word is a string that is in the list of reserved words
    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        RESERVED_WORDS.iter().any(|word| input == *word)
    }

    fn token_type(&self) -> Self::TokenType {
        GeeToken::ReservedWord
    }
}
impl TokenRule for IdentifierRule {
    type TokenType = GeeToken;

    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        input.chars().all(|c| c.is_ascii_alphanumeric()) && !input.is_empty()
    }

    fn token_type(&self) -> Self::TokenType {
        GeeToken::Identifier
    }
}
impl TokenRule for NumberRule {
    type TokenType = GeeToken;

    /// A number is a string that is a valid number
    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        input.chars().all(|c| c.is_ascii_digit()) && !input.is_empty()
    }

    fn token_type(&self) -> Self::TokenType {
        GeeToken::Number
    }
}
impl TokenRule for SymbolRule {
    type TokenType = GeeToken;

    /// A symbol is a string that is in the list of symbols
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
        GeeToken::Symbol
    }
}
impl TokenRule for StringRule {
    type TokenType = GeeToken;

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
        GeeToken::String
    }
}
impl TokenRule for WhitespaceRule {
    type TokenType = GeeToken;

    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        input.chars().all(char::is_whitespace) && !input.is_empty()
    }

    fn token_type(&self) -> Self::TokenType {
        GeeToken::Whitespace
    }

    fn is_whitespace(&self) -> bool {
        true
    }
}
impl TokenRule for IndentRule {
    type TokenType = GeeToken;

    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        input == "@"
    }

    fn token_type(&self) -> Self::TokenType {
        GeeToken::Indent
    }
}
impl TokenRule for UndentRule {
    type TokenType = GeeToken;

    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        input == "~"
    }

    fn token_type(&self) -> Self::TokenType {
        GeeToken::Undent
    }
}
impl TokenRule for EolnRule {
    type TokenType = GeeToken;

    fn matches(&self, input: &str, range: Range<usize>) -> bool {
        let input = &input[range];
        input == ";"
    }

    fn token_type(&self) -> Self::TokenType {
        GeeToken::Eoln
    }
}

/// A `NonTerminal` in the grammar of Gee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GeeNT {
    // Goal / start symbol
    Script,
    // Statements
    StmtList,
    StmtListRest,
    Statement,
    Assign,
    IfStatement,
    ElseStatement,
    WhileStatement,
    Block,
    // Expressions
    Expression,
    ExpressionTail,
    RelationalExpr,
    RelationalExprTail,
    AddExpr,
    AddExprTail,
    Term,
    TermTail,
    Factor,
    // Expression operators
    Relation,
    // AndOr,
}
impl NonTerminal for GeeNT {}

/// A Terminal in the grammar of Gee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GeeT {
    DoubleEqual,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
    Plus,
    Minus,
    Mult,
    Div,
    Colon,
    Assign,
    Or,
    And,
    If,
    Else,
    While,
    String,
    Number,
    Identifier,
    Indent,
    Undent,
    Eoln,
    EoF,
    LeftParen,
    RightParen,
}
impl Terminal for GeeT {
    fn eof() -> Self {
        Self::EoF
    }
}

impl TryFrom<TokenSpan<'_, GeeToken>> for GeeT {
    type Error = TokenConversionError<GeeToken>;

    fn try_from(value: TokenSpan<'_, GeeToken>) -> Result<Self, Self::Error> {
        match (value.kind, value.text) {
            (GeeToken::Comment | GeeToken::Whitespace, _) => {
                Err(TokenConversionError::SkipToken(value.into_owned()))
            }
            (GeeToken::ReservedWord, text) => match text {
                "if" => Ok(Self::If),
                "else" => Ok(Self::Else),
                "while" => Ok(Self::While),
                "or" => Ok(Self::Or),
                "and" => Ok(Self::And),
                _ => Err(TokenConversionError::MalformedToken(value.into_owned())),
            },
            (GeeToken::Identifier, _) => Ok(Self::Identifier),
            (GeeToken::Number, _) => Ok(Self::Number),
            (GeeToken::String, _) => Ok(Self::String),
            (GeeToken::Symbol, text) => match text {
                "==" => Ok(Self::DoubleEqual),
                "!=" => Ok(Self::NotEqual),
                "<" => Ok(Self::LessThan),
                "<=" => Ok(Self::LessThanEqual),
                ">" => Ok(Self::GreaterThan),
                ">=" => Ok(Self::GreaterThanEqual),
                "+" => Ok(Self::Plus),
                "-" => Ok(Self::Minus),
                "*" => Ok(Self::Mult),
                "/" => Ok(Self::Div),
                "=" => Ok(Self::Assign),
                ":" => Ok(Self::Colon),
                "(" => Ok(Self::LeftParen),
                ")" => Ok(Self::RightParen),
                _ => Err(TokenConversionError::MalformedToken(value.into_owned())),
            },
            (GeeToken::Indent, "@") => Ok(Self::Indent),
            (GeeToken::Undent, "~") => Ok(Self::Undent),
            (GeeToken::Eoln, ";") => Ok(Self::Eoln),
            _ => Err(TokenConversionError::MalformedToken(value.into_owned())),
        }
    }
}

fn gee_grammar() -> Grammar<GeeNT, GeeT, TerminatingReady<GeeNT, GeeT>> {
    Grammar::new(
        GeeNT::Script,
        vec![
            Production::new(GeeNT::Script, derivation![GeeNT::StmtList]),
            Production::new(
                GeeNT::StmtList,
                derivation![GeeNT::Statement, GeeNT::StmtListRest],
            ),
            Production::new(
                GeeNT::StmtListRest,
                derivation![GeeNT::Statement, GeeNT::StmtListRest],
            ),
            Production::new(GeeNT::StmtListRest, derivation![]),
            Production::new(GeeNT::Statement, derivation![GeeNT::Assign]),
            Production::new(GeeNT::Statement, derivation![GeeNT::IfStatement]),
            Production::new(GeeNT::Statement, derivation![GeeNT::WhileStatement]),
            Production::new(
                GeeNT::Assign,
                derivation![
                    GeeT::Identifier,
                    GeeT::Assign,
                    GeeNT::Expression,
                    GeeT::Eoln
                ],
            ),
            Production::new(
                GeeNT::IfStatement,
                derivation![
                    GeeT::If,
                    GeeNT::Expression,
                    GeeNT::Block,
                    GeeNT::ElseStatement
                ],
            ),
            Production::new(GeeNT::ElseStatement, derivation![GeeT::Else, GeeNT::Block]),
            Production::new(GeeNT::ElseStatement, derivation![]),
            Production::new(
                GeeNT::WhileStatement,
                derivation![GeeT::While, GeeNT::Expression, GeeNT::Block],
            ),
            Production::new(
                GeeNT::Block,
                derivation![
                    GeeT::Colon,
                    GeeT::Eoln,
                    GeeT::Indent,
                    GeeNT::StmtList,
                    GeeT::Undent
                ],
            ),
            Production::new(
                GeeNT::Expression,
                derivation![GeeNT::RelationalExpr, GeeNT::ExpressionTail],
            ),
            Production::new(
                GeeNT::ExpressionTail,
                derivation![GeeT::And, GeeNT::RelationalExpr],
            ),
            Production::new(
                GeeNT::ExpressionTail,
                derivation![GeeT::Or, GeeNT::RelationalExpr],
            ),
            Production::new(GeeNT::ExpressionTail, derivation![]),
            Production::new(
                GeeNT::RelationalExpr,
                derivation![GeeNT::AddExpr, GeeNT::RelationalExprTail],
            ),
            Production::new(
                GeeNT::RelationalExprTail,
                derivation![GeeNT::Relation, GeeNT::AddExpr],
            ),
            Production::new(GeeNT::RelationalExprTail, derivation![]),
            Production::new(GeeNT::AddExpr, derivation![GeeNT::Term, GeeNT::AddExprTail]),
            Production::new(GeeNT::AddExprTail, derivation![GeeT::Plus, GeeNT::Term]),
            Production::new(GeeNT::AddExprTail, derivation![GeeT::Minus, GeeNT::Term]),
            Production::new(GeeNT::AddExprTail, derivation![]),
            Production::new(GeeNT::Term, derivation![GeeNT::Factor, GeeNT::TermTail]),
            Production::new(GeeNT::TermTail, derivation![GeeT::Mult, GeeNT::Factor]),
            Production::new(GeeNT::TermTail, derivation![GeeT::Div, GeeNT::Factor]),
            Production::new(GeeNT::TermTail, derivation![]),
            Production::new(GeeNT::Factor, derivation![GeeT::Number]),
            Production::new(GeeNT::Factor, derivation![GeeT::String]),
            Production::new(GeeNT::Factor, derivation![GeeT::Identifier]),
            Production::new(
                GeeNT::Factor,
                derivation![GeeT::LeftParen, GeeNT::Expression, GeeT::RightParen],
            ),
            Production::new(GeeNT::Relation, derivation![GeeT::DoubleEqual]),
            Production::new(GeeNT::Relation, derivation![GeeT::NotEqual]),
            Production::new(GeeNT::Relation, derivation![GeeT::LessThan]),
            Production::new(GeeNT::Relation, derivation![GeeT::LessThanEqual]),
            Production::new(GeeNT::Relation, derivation![GeeT::GreaterThan]),
            Production::new(GeeNT::Relation, derivation![GeeT::GreaterThanEqual]),
        ],
    )
    .unwrap()
    .check_terminating()
    .unwrap()
    .generate_sets()
}

// AST nodes for the Gee programming language

pub enum Expression {}

pub enum AST {
    Expr(Expression),
}

fn run(input: &str) -> Result<String, GeeError> {
    let preprocessed_input = gee_preprocessor(input)?;
    let scanner = Scanner::new(preprocessed_input.as_str(), &RULES);
    let grammar = gee_grammar();
    let parse_table = grammar.generate_parse_table();
    let parser = Parser::new(parse_table);
    let parse_tree = parser.parse(scanner).unwrap();

    let mut output = String::new();

    for (i, line) in preprocessed_input
        .lines()
        .enumerate()
        .filter(|(_, l)| !(l.starts_with('#') || l.is_empty()))
    {
        let trimmed_line = line.trim_start();
        if trimmed_line.is_empty() {
            continue;
        }
        writeln!(output, "{} 	{line}\n", i + 1).unwrap();
    }

    parse_tree.visit_euler_tour(&mut |node| {
        let text = match node {
            EulerTraversalVisit::Pre(node) => match node {
                ParseTree::Node(node) => match node.non_terminal {
                    GeeNT::IfStatement => "if ",
                    GeeNT::ElseStatement => {
                        // // if the if statement has no else block, we don't need to print the else statement
                        // // recall: `elseStatement = "else" block | ε`, so if there is 1 child it means the else statement is empty
                        // if node.children.len() == 1 {
                        //     ""
                        // } else {
                        "else\n"
                        // }
                    }
                    GeeNT::WhileStatement => "while ",
                    GeeNT::Assign => "= ",
                    GeeNT::Expression => {
                        match &node.children[1].children().and_then(|c| c.first()) {
                            Some(ParseTree::Leaf(ParseTreeLeaf {
                                terminal: GeeT::And | GeeT::Or,
                                token_span,
                            })) => &format!("{} ", token_span.text),
                            _ => "",
                        }
                    }
                    GeeNT::RelationalExpr => {
                        match &node.children[1].children().and_then(|c| c.first()) {
                            Some(ParseTree::Node(ParseTreeNode {
                                non_terminal: GeeNT::Relation,
                                children,
                            })) => &format!("{} ", children[0].token_span().unwrap().text),
                            _ => "",
                        }
                    }
                    GeeNT::AddExpr => match &node.children[1].children().and_then(|c| c.first()) {
                        Some(ParseTree::Leaf(ParseTreeLeaf {
                            terminal: GeeT::Plus | GeeT::Minus,
                            token_span,
                        })) => &format!("{} ", token_span.text),
                        _ => "",
                    },
                    GeeNT::Term => match &node.children[1].children().and_then(|c| c.first()) {
                        Some(ParseTree::Leaf(ParseTreeLeaf {
                            terminal: GeeT::Mult | GeeT::Div,
                            token_span,
                        })) => &format!("{} ", token_span.text),
                        _ => "",
                    },
                    _ => "",
                },
                ParseTree::Leaf(leaf) => match leaf.terminal {
                    GeeT::String | GeeT::Number => leaf.token_span.text,
                    GeeT::Identifier => &format!("{} ", leaf.token_span.text),
                    GeeT::Eoln => "\n",
                    GeeT::LeftParen => "( ",
                    GeeT::RightParen => ") ",
                    _ => "",
                },
                ParseTree::Epsilon => "",
            },
            EulerTraversalVisit::Post { child, .. } => match child.non_terminal() {
                Some(GeeNT::IfStatement) => "endif\n",
                Some(GeeNT::WhileStatement) => "endwhile\n",
                _ => "",
            },
        };

        output.push_str(text);
    });

    Ok(output
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n"))
}

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

    // read the file into a string
    let input = std::fs::read_to_string(file_path).map_err(|e| e.to_string())?;
    // run the parser
    let result = run(&input);
    match result {
        Ok(output) => {
            println!("Parsing successful!");
            println!("{output}");
        }
        Err(e) => {
            eprintln!("Parsing failed: {e}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    #[rstest]
    #[case::max(
        r"m =  a
if b > a:
    m = b + 1
r = m",
        Ok("m =  a;
if b > a:;
@m = b + 1;
~r = m;".to_string())
    )]
    #[case::uneven_indentation(
        r"ans = b
if a > b:
    if a > c:
        ans = a
   else:
          ans = b",
        Err(GeeError::UnevenIndentation {
            line: 5,
            expected: 0,
            found: 3,
        })
    )]
    #[case::uneven_indentation(
        r"ans = b
if a > b:
   if a > c:
        ans = a
     else:
          ans = b",
        Err(GeeError::UnevenIndentation {
            line: 5,
            expected: 3,
            found: 5,
        })
    )]
    #[case::mixed_indentation(
        "ans = b
if a > b:
\tif a > c
\t\tans = a
    else:
        ans = b",
        Err(GeeError::MixedIndentation(5))
    )]
    #[case::test(
        "ans = b
if a > b:
\tif a > c:
\t\tans = a
\telse:
\t\tans = b
else:
\tif b < c:
\t\tans = c
max3 = ans",
        Ok(r"ans = b;
if a > b:;
@if a > c:;
@ans = a;
~else:;
@ans = b;
~~else:;
@if b < c:;
@ans = c;
~~max3 = ans;".to_string())
    )]
    #[case::fact(
        "n = 8
i = 1
f = i
while i < n:
\ti = i + 1
\tf = f * i
ans = f",
        Ok(
            r"n = 8;
i = 1;
f = i;
while i < n:;
@i = i + 1;
f = f * i;
~ans = f;".to_string()
        )
    )]
    fn test_indent(#[case] input: &str, #[case] expected: Result<String, GeeError>) {
        let result = gee_preprocessor(input);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parser_max_gee() {
        let input = r"#  max program 

m =  a
if b > a:
    m = b + 1
r = m";
        let expected = "3 	m =  a;
4 	if b > a:;
5 	@m = b + 1;
6 	~r = m;
= m a
if > b a
= m + b 1
else
endif
= r m";
        let result = run(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parser_test_gee() {
        let input = r"ans = b
if a > b:
   if a > c:
       ans = a
   else:
       ans = b
else:
   if b < c:
       ans = c
max3 = ans";
        let expected = r"1 	ans = b;
2 	if a > b:;
3 	@if a > c:;
4 	@ans = a;
5 	~else:;
6 	@ans = b;
7 	~~else:;
8 	@if b < c:;
9 	@ans = c;
10 	~~max3 = ans;
= ans b
if > a b
if > a c
= ans a
else
= ans b
endif
else
if < b c
= ans c
else
endif
endif
= max3 ans";
        let result = run(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parser_fact_gee() {
        let input = r"n = 8
i = 1
f = i
while i < n:
    i = i + 1
    f = f * i
ans = f";
        let expected = r"1 	n = 8;
2 	i = 1;
3 	f = i;
4 	while i < n:;
5 	@i = i + 1;
6 	f = f * i;
7 	~ans = f;
= n 8
= i 1
= f i
while < i n
= i + i 1
= f * f i
endwhile
= ans f";
        let result = run(input).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parser_compound() {
        let input = r"c = a + 5 * b";
        let expected = "1 	c = a + 5 * b;\n= c + a * 5 b";
        let result = run(input).unwrap();
        assert_eq!(result, expected);

        let input = r"c = (a + 5 * b) + 2";
        let expected = "1 	c = (a + 5 * b) + 2;\n= c + ( + a * 5 b ) 2";
        let result = run(input).unwrap();
        assert_eq!(result, expected);

        let input = r"c = 2 + (a + 5 * b)";
        let expected = "1 	c = 2 + (a + 5 * b);\n= c + 2 ( + a * 5 b )";
        let result = run(input).unwrap();
        assert_eq!(result, expected);

        let input = r"c = (a + b) /  5 > n";
        let expected = "1 	c = (a + b) /  5 > n;\n= c > / ( + a b ) 5 n";
        let result = run(input).unwrap();
        assert_eq!(result, expected);

        let input = r"c = n >= (a + b) / 5";
        let expected = "1 	c = n >= (a + b) / 5;\n= c >= n / ( + a b ) 5";
        let result = run(input).unwrap();
        assert_eq!(result, expected);

        let input = r"c = a + b and n > 5";
        let expected = "1 	c = a + b and n > 5;\n= c and + a b > n 5";
        let result = run(input).unwrap();
        assert_eq!(result, expected);
    }
}
