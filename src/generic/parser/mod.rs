//! Generic code for building a (top-down, table-driven LL(1)) predictive parser.
//!
//! ## Design
//!
//! ### Parser
//!
//! A `Parser` has:
//! - a `Scanner` that does all the work of creating tokens from the input.
//! - a `Grammar` that defines a set of rules (productions) for the language.
//!
//! Before starting, the parser must be able to:
//!
//! - ensure the grammar is LL(1) by building the FIRST, FOLLOW, and FIRST+ sets.
//!   - while doing so (or before), it may attempt to relax the grammar into an LL(1) grammar by
//!   - eliminating left-recursion and left-factoring.
//!
//! it will return an error if it is unable to do so.
//!
//! The parser will then build an LL(1) parsing table that it will use to parse the input of any
//! compatible scanner (i.e. a scanner that produces a token stream
//! of the same type used by the parsers grammar).
//!
//! ### Grammar
//!
//! A `Grammar` consists of a series of rules (productions) that match `NonTerminals` to their derivations
//! (sequences of Terminals and `NonTerminals` that can be derived from the `NonTerminal`).
//!
//! A `Grammar` can be in one of 5 states:
//! - `Grammar<NonTerminating>`: A grammar that may contain left-recursion.
//!     - This is the initial state of a grammar, and the only one that can be constructed directly.
//! - `Grammar<Terminating>`: A grammar that has been checked for left-recursion.
//!     - This is the result of calling `Grammar::check_terminating` on a `Grammar<NonTerminating>`.
//! - `Grammar<TerminatingReady>`: A Terminating grammar that has been prepared for table-driven parsing.
//!    - This is the result of calling `Grammar::generate_sets` on a `Grammar<Terminating>`.
//! - `Grammar<LL1>`: A `TerminatingReady` grammar that has been checked to be LL(1).
//!    - This is the result of calling `Grammar::check_ll1` on a `Grammar<TerminatingReady>`.
//!
//! A parse table can only be built from a `Grammar<LL1>`.

use core::fmt;
use std::collections::BTreeMap;

use crate::generic::ScannerResult;

use super::{Error, OwnedTokenSpan, Scanner, TokenConversionError, TokenSpan};

pub mod grammar;
mod tree;
pub use tree::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Symbol<NT, T> {
    /// A terminal symbol, which contains a token type
    Terminal(T),
    /// A non-terminal symbol, which contains an index into the grammar's list of non-terminals
    NonTerminal(NT),
    /// Epsilon, the empty string
    Epsilon,
    /// End of file
    Eof,
}

/// marker trait for a non-terminal symbol
pub trait NonTerminal: Sized + Copy + PartialEq + Eq + PartialOrd + Ord + std::fmt::Debug {
    // non-terminals need to be able to be converted into ids
    fn to_symbol<T>(self) -> Symbol<Self, T> {
        Symbol::NonTerminal(self)
    }
}

/// marker trait for a terminal symbol
pub trait Terminal: Sized + Copy + PartialEq + Eq + PartialOrd + Ord + std::fmt::Debug {
    fn eof() -> Self;

    fn to_symbol<NT>(self) -> Symbol<NT, Self> {
        if self == Self::eof() {
            Symbol::Eof
        } else {
            Symbol::Terminal(self)
        }
    }
}

/// Trait for merging two terminal symbols into a new terminal symbol of a possibly different type,
/// this is needed to enable left-factoring
pub trait Merge<R>: Terminal {
    /// Join a prefix of terminal symbols into a new terminal symbol
    fn merge(prefix: impl AsRef<[R]>) -> Self;
}

/// A derivation that a non-terminal can expand to
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Derivation<NT, T> {
    /// The symbols that the non-terminal expands to
    symbols: Vec<Symbol<NT, T>>,
}
impl<NT, T> fmt::Display for Derivation<NT, T>
where
    NT: fmt::Debug,
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.symbols)
    }
}

/// A production in a grammar
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Production<NT, T> {
    /// The non-terminal that the production expands
    pub(crate) non_terminal: NT,
    /// The derivation that the non-terminal can expand to
    pub(crate) derivation: Derivation<NT, T>,
    /// Phantom data to store the type of the non-terminal
    language: std::marker::PhantomData<NT>,
}
impl<NT, T> fmt::Display for Production<NT, T>
where
    NT: fmt::Debug,
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} -> {}", self.non_terminal, self.derivation)
    }
}

type Table<'a, NT, T> = BTreeMap<NT, BTreeMap<T, Vec<&'a Production<NT, T>>>>;
type LL1Table<'a, NT, T> = BTreeMap<NT, BTreeMap<T, &'a Production<NT, T>>>;

#[derive(Debug, Clone)]
pub struct ParseTable<'a, NT, T> {
    /// The table of productions
    /// `Table[non_terminal][terminal]` = list of productions that can be expanded to
    pub table: Table<'a, NT, T>,
    /// Start symbol
    pub start_symbol: NT,
}

#[derive(Debug, Clone)]
pub struct LL1ParseTable<'a, NT, T> {
    /// The table of productions
    /// `Table[non_terminal][terminal]` = index of the production that can be expanded to
    pub table: LL1Table<'a, NT, T>,
    /// Start symbol
    pub start_symbol: NT,
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum ParseError<
    NonTerminal: self::NonTerminal,
    Terminal: self::Terminal,
    Token: std::fmt::Debug,
> {
    #[error("Expected end of input, found {0:?}")]
    ExpectedEof(OwnedTokenSpan<Token>),
    #[error("Unexpected end of input, expected symbol {0:?}")]
    UnexpectedEof(Symbol<NonTerminal, Terminal>),
    #[error("No production found for non-terminal {0:?} and lookahead {1:?} (which is a {2:?})")]
    NoProduction(NonTerminal, Option<OwnedTokenSpan<Token>>, Terminal),
    #[error("Unexpected token {0:?}, expected symbol {1:?}")]
    UnexpectedToken(OwnedTokenSpan<Token>, Symbol<NonTerminal, Terminal>),
    #[error("Unexpected end of stack, ")]
    UnexpectedEndOfStack,
}

pub type ParseResult<OK, NT, T, Token> = Result<OK, Error<NT, T, Token>>;
pub type ParseTreeResult<'a, NT, T, Token> = ParseResult<ParseTree<'a, NT, T, Token>, NT, T, Token>;

#[derive(Debug, Clone)]
pub struct Parser<'productions, NT, T> {
    pub table: ParseTable<'productions, NT, T>,
}

#[derive(Debug, Clone)]
pub struct LL1Parser<'productions, NT, T> {
    pub table: LL1ParseTable<'productions, NT, T>,
}

/////////////////////////////////////////////////
// Macros
/////////////////////////////////////////////////
#[macro_export]
macro_rules! derivation {
    [] => {
        $crate::generic::Derivation::new(vec![$crate::generic::Symbol::Epsilon])
    };
    [$($symbol:expr),*] => {
        $crate::generic::Derivation::new(vec![$($symbol.to_symbol()),*])
    };
}

/////////////////////////////////////////////////
// Implementations
/////////////////////////////////////////////////

impl NonTerminal for usize {}

impl<NT: Eq, T: Eq> Derivation<NT, T> {
    /// Create a new derivation from the given symbols
    ///
    /// # Panics
    ///
    /// Panics if the symbols contain epsilon and other symbols
    #[must_use]
    pub fn new(symbols: Vec<Symbol<NT, T>>) -> Self {
        if symbols.contains(&Symbol::Epsilon) {
            assert!(
                symbols.len() == 1,
                "Derivation contained epsilon among other symbols"
            );
        }
        Self { symbols }
    }
}

impl<NT, T> Symbol<NT, T> {
    #[must_use]
    pub const fn to_symbol(self) -> Self {
        self
    }
}

impl<NT, T> From<Vec<Symbol<NT, T>>> for Derivation<NT, T> {
    fn from(symbols: Vec<Symbol<NT, T>>) -> Self {
        Self { symbols }
    }
}

impl<NT, T> Production<NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    /// Create a new production from the given non-terminal and derivation
    #[must_use]
    pub fn new<I: Into<NT>, D: Into<Derivation<NT, T>>>(non_terminal: I, derivation: D) -> Self {
        Self {
            non_terminal: non_terminal.into(),
            derivation: derivation.into(),
            language: std::marker::PhantomData::<NT>,
        }
    }
}

impl<'a, NT, T> ParseTable<'a, NT, T>
where
    NT: NonTerminal,
    T: Terminal + Copy + Ord,
{
    /// Create a new, empty parse table
    #[must_use]
    pub const fn new(start_symbol: NT) -> Self {
        Self {
            table: BTreeMap::new(),
            start_symbol,
        }
    }

    /// Get the productions for the given non-terminal and lookahead
    /// Returns None if no productions are found
    ///
    /// # Panics
    ///
    /// Panics if the returned slice would be empty
    pub fn get_productions(&self, nt: NT, lookahead: T) -> Option<&[&Production<NT, T>]> {
        let ret = self
            .table
            .get(&nt)
            .and_then(|m| m.get(&lookahead))
            .map(Vec::as_slice);
        assert!(ret.is_none() || !ret.unwrap().is_empty());
        ret
    }

    /// Add a production to the parse table
    pub fn add_production<'b: 'a>(
        &mut self,
        nt: NT,
        lookahead: T,
        production: &'b Production<NT, T>,
    ) {
        self.table
            .entry(nt)
            .or_default()
            .entry(lookahead)
            .or_default()
            .push(production);
    }
}

impl<'a, NT, T> LL1ParseTable<'a, NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    /// Create a new, empty parse table
    #[must_use]
    pub const fn new(start_symbol: NT) -> Self {
        Self {
            table: BTreeMap::new(),
            start_symbol,
        }
    }

    /// Get the production for the given non-terminal and lookahead
    /// Returns None if no production is found
    pub fn get_production(&self, nt: NT, lookahead: T) -> Option<&Production<NT, T>> {
        self.table.get(&nt).and_then(|m| m.get(&lookahead)).copied()
    }

    /// Add a production to the parse table
    ///
    /// # Panics
    ///
    /// Panics if the table already contains a production for the given non-terminal and lookahead,
    /// which would indicate that the grammar is not LL(1)
    pub fn add_production<'b: 'a>(
        &mut self,
        nt: NT,
        lookahead: T,
        production: &'b Production<NT, T>,
    ) {
        assert!(
            self.table
                .entry(nt)
                .or_default()
                .insert(lookahead, production)
                .is_none(),
            "Table already contains a production for non-terminal {nt:?} and lookahead {lookahead:?}",
        );
    }
}

impl<'a, NT, T> Parser<'a, NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    /// Create a new parser from the given parse table
    #[must_use]
    pub const fn new(table: ParseTable<'a, NT, T>) -> Self {
        Self { table }
    }

    /// Parse the given input using the parser's parse table
    ///
    /// This implementation combines recursive descent with table-driven parsing,
    /// recursion is needed to enable backtracking but we can use the table to avoid
    /// the need for hand-written recursive descent functions.
    ///
    /// # Errors
    ///
    /// Returns a `ParseError` if the input cannot be parsed,
    ///
    /// May return a `ParseError::TokenConversion` if the scanner produces a token that cannot be converted to a terminal, this indicates a bug in your scanner rules.
    pub fn parse<'token, Token>(
        &'a self,
        scanner: Scanner<'token, Token>,
    ) -> ParseTreeResult<'token, NT, T, Token>
    where
        Token: Copy + std::fmt::Debug,
        T: TryFrom<TokenSpan<'token, Token>, Error = TokenConversionError<Token>>,
    {
        /// Inner recursive descent function
        ///
        /// returns `Ok(Some(ParseTree))` if successful, `Ok(None)` if the symbol is epsilon,
        /// and `Err(ParseError)` if an error occurs
        fn inner<'productions, 'token, NT, T, Token>(
            // reference to the parser's parse table
            table: &ParseTable<'productions, NT, T>,
            // tokens from the scanner
            tokens: &[TokenSpan<'token, Token>],
            // index into the tokens, used to keep track of the current token
            index: &mut usize,
            // the current symbol we're trying to match
            symbol: Symbol<NT, T>,
        ) -> ParseTreeResult<'token, NT, T, Token>
        where
            NT: NonTerminal,
            T: Terminal + TryFrom<TokenSpan<'token, Token>, Error = TokenConversionError<Token>>,
            Token: Copy + std::fmt::Debug,
        {
            // skip whitespace
            while *index < tokens.len() && tokens[*index].is_whitespace {
                *index += 1;
            }
            match symbol {
                Symbol::Terminal(t) => {
                    let token = tokens.get(*index).ok_or(ParseError::UnexpectedEndOfStack)?;
                    if t == (*token).try_into()? {
                        *index += 1;
                        Ok(ParseTree::leaf(t, *token))
                    } else {
                        Err(Error::ParseError(ParseError::UnexpectedToken(
                            token.into_owned(),
                            symbol,
                        )))
                    }
                }
                Symbol::NonTerminal(nt) => {
                    let lookahead = tokens.get(*index).copied();
                    let kind = lookahead.map_or_else(|| Ok(T::eof()), TryInto::try_into)?;
                    let productions = table.get_productions(nt, kind);

                    if let Some(productions) = productions {
                        for production in productions {
                            let mut i = *index;
                            let mut children = Vec::new();

                            let error: ParseResult<(), NT, T, Token> = {
                                for symbol in &production.derivation.symbols {
                                    match inner(table, tokens, &mut i, *symbol) {
                                        Ok(tree) => children.push(tree),
                                        Err(e) => {
                                            return Err(e);
                                        }
                                    }
                                }
                                Ok(())
                            };

                            if error.is_ok() {
                                *index = i;
                                return Ok(ParseTree::node(nt, children));
                            }
                        }
                    }

                    Err(Error::ParseError(ParseError::NoProduction(
                        nt,
                        lookahead.map(TokenSpan::into_owned),
                        kind,
                    )))
                }
                Symbol::Epsilon => Ok(ParseTree::Epsilon),
                Symbol::Eof => {
                    if *index == tokens.len() {
                        Ok(ParseTree::Epsilon)
                    } else {
                        Err(Error::ParseError(ParseError::ExpectedEof(
                            tokens[*index].into_owned(),
                        )))
                    }
                }
            }
        }

        let tokens = scanner.iter().collect::<ScannerResult<Vec<_>>>()?;

        let mut index = 0;

        let tree = inner(
            &self.table,
            &tokens,
            &mut index,
            Symbol::NonTerminal(self.table.start_symbol),
        )?;

        if index == tokens.len() {
            Ok(tree)
        } else {
            Err(Error::ParseError(ParseError::ExpectedEof(
                tokens[index].into_owned(),
            )))
        }
    }
}

impl<'productions, NT, T> LL1Parser<'productions, NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    /// Create a new parser from the given parse table
    #[must_use]
    pub const fn new(table: LL1ParseTable<'productions, NT, T>) -> Self {
        Self { table }
    }

    /// Parse the given input using the parser's parse table
    ///
    /// TODO: return the parse tree instead of just true/false
    ///
    /// # Errors
    ///
    /// Returns a `ParseError` if the input cannot be parsed,
    ///
    /// May return a `ParseError::TokenConversion` if the scanner produces a token that cannot be converted to a terminal, this indicates a bug in your scanner rules.
    pub fn parse<'token, Token>(
        &self,
        scanner: Scanner<'token, Token>,
    ) -> ParseTreeResult<'token, NT, T, Token>
    where
        Token: Copy + std::fmt::Debug,
        T: TryFrom<TokenSpan<'token, Token>, Error = TokenConversionError<Token>>,
    {
        let mut stack: Vec<Symbol<NT, T>> = vec![Symbol::NonTerminal(self.table.start_symbol)];
        let mut input = scanner.iter().peekable();
        let mut eof = false;

        // the last non-terminal that was built
        let mut last_built_nt = ParseTree::Epsilon;

        // the stack of non-terminals to build
        let mut nt_stack = Vec::new();
        // the stack of:
        // - the children of the NT being build
        // - how many children it wants
        let mut nt_stack_child_counts: Vec<(Vec<_>, usize)> = vec![];

        loop {
            if let Some((children, wants)) = nt_stack_child_counts.last_mut() {
                if *wants == children.len() {
                    // we've built all the children for the NT at the top of the stack,
                    // now it's time to build the node for it, and add it to the parent's children
                    let node =
                        ParseTree::node(nt_stack[nt_stack.len() - 1], std::mem::take(children));
                    nt_stack.pop();
                    nt_stack_child_counts.pop();
                    node.clone_into(&mut last_built_nt);

                    if let Some((children, _)) = nt_stack_child_counts.last_mut() {
                        children.push(node);
                    }

                    // instead of looping, we'll just continue to the next iteration (letting the outer loop handle it)
                    continue;
                }
            }

            let lookahead: Option<TokenSpan<'_, Token>> = match input.peek().copied() {
                Some(Ok(token)) if token.is_whitespace => {
                    input.next();
                    eof |= token.is_eof;
                    continue;
                }
                Some(Ok(token)) => Some(token),
                Some(Err(e)) => {
                    return Err(Error::from(e));
                }
                None => None,
            };
            let top = stack.pop();

            match (top, lookahead) {
                (Some(Symbol::Terminal(t)), Some(token)) if t == token.try_into()? => {
                    if let Some((children, _)) = nt_stack_child_counts.last_mut() {
                        children.push(ParseTree::leaf(t, token));
                    }

                    input.next();
                    eof = token.is_eof;
                }
                (Some(Symbol::NonTerminal(nt)), token) => {
                    let kind = token.map_or_else(|| Ok(T::eof()), TryInto::try_into)?;

                    if let Some(production) = self.table.get_production(nt, kind) {
                        let number_of_production_symbols = production.derivation.symbols.len();
                        nt_stack.push(nt);
                        nt_stack_child_counts.push((
                            Vec::with_capacity(number_of_production_symbols),
                            number_of_production_symbols,
                        ));

                        let production_symbols =
                            production.derivation.symbols.iter().rev().copied();
                        stack.extend(production_symbols);
                    } else {
                        return Err(Error::ParseError(ParseError::NoProduction(
                            nt,
                            token.map(TokenSpan::into_owned),
                            kind,
                        )));
                    }
                    eof |= token.is_some_and(|t| t.is_eof);
                }
                (Some(Symbol::Eof), Some(token)) => {
                    return Err(Error::ParseError(ParseError::ExpectedEof(
                        token.into_owned(),
                    )));
                }
                (Some(Symbol::Eof) | None, None) => {
                    if eof {
                        break;
                    }
                    dbg!(stack);
                    unreachable!("Should have broken out of the loop by now");
                }
                (Some(s @ Symbol::Terminal(_)), Some(token)) => {
                    // dbg!(s, token, stack);
                    return Err(Error::ParseError(ParseError::UnexpectedToken(
                        token.into_owned(),
                        s,
                    )));
                }
                (Some(s @ Symbol::Terminal(_)), None) => {
                    return Err(Error::ParseError(ParseError::UnexpectedEof(s)));
                }
                (None, Some(token)) => {
                    return Err(Error::ParseError(ParseError::UnexpectedToken(
                        token.into_owned(),
                        Symbol::Eof,
                    )));
                }
                (Some(Symbol::Epsilon), _) => {
                    if let Some((children, _)) = nt_stack_child_counts.last_mut() {
                        children.push(ParseTree::Epsilon);
                    }
                }
            }
        }

        Ok(last_built_nt)
    }
}

#[cfg(test)]
mod ll1_tests {
    //! Tests to ensure we can parse with an LL(1) grammar
    use std::vec;

    use super::{
        grammar::{Grammar, NonTerminating, TerminatingReady},
        ParseTree, Parser,
    };
    use crate::generic::{test_utils::*, LL1Parser, ParseTreeLeaf, Scanner};
    use pretty_assertions::assert_eq;
    use rstest::{fixture, rstest};

    #[fixture]
    fn grammar(
        expr_grammar_terminating: Grammar<ExprNT, ExprT, NonTerminating>,
    ) -> Grammar<ExprNT, ExprT, TerminatingReady<ExprNT, ExprT>> {
        expr_grammar_terminating
            .check_terminating()
            .unwrap()
            .generate_sets()
    }

    #[rstest]
    #[case("1 + 2", true)]
    #[case("1 + 2 * 3", true)]
    #[case("1 + 2 * 3 + 4", true)]
    #[case("(1 + 2) * 3", true)]
    #[case(") 1 + 2", false)]
    #[case("1 + 2 +", false)]
    #[case("1 + 2 *", false)]
    #[case("1 + 2 /", false)]
    fn parse_expr_grammar(
        grammar: Grammar<ExprNT, ExprT, TerminatingReady<ExprNT, ExprT>>,
        #[case] input: &str,
        #[case] expected: bool,
    ) {
        let grammar = grammar.check_ll1().unwrap();
        let table = grammar.generate_parse_table();

        let parser = LL1Parser::new(table);

        let scanner = Scanner::new(input, &EXPR_RULES);

        let result = parser.parse(scanner);

        assert_eq!(
            result.is_ok(),
            expected,
            "Failed to parse input: {input}, {result:?}"
        );
    }

    #[test]
    fn parse_expr_to_tree() {
        let input = "1 + 2 * 3";

        let scanner = Scanner::new(input, &EXPR_RULES);

        let mut exp_iter = scanner
            .iter()
            .filter_map(Result::ok)
            .filter(|t| !t.is_whitespace);

        let expected = ParseTree::node(
            ExprNT::Goal,
            vec![ParseTree::node(
                ExprNT::Expr,
                vec![
                    ParseTree::node(
                        ExprNT::Term,
                        vec![
                            ParseTree::node(
                                ExprNT::Factor,
                                vec![ParseTree::leaf(ExprT::Num, exp_iter.next().unwrap())],
                            ),
                            ParseTree::node(ExprNT::TermPrime, vec![ParseTree::Epsilon]),
                        ],
                    ),
                    ParseTree::node(
                        ExprNT::ExprPrime,
                        vec![
                            ParseTree::leaf(ExprT::Plus, exp_iter.next().unwrap()),
                            ParseTree::node(
                                ExprNT::Term,
                                vec![
                                    ParseTree::node(
                                        ExprNT::Factor,
                                        vec![ParseTree::leaf(ExprT::Num, exp_iter.next().unwrap())],
                                    ),
                                    ParseTree::node(
                                        ExprNT::TermPrime,
                                        vec![
                                            ParseTree::leaf(ExprT::Mult, exp_iter.next().unwrap()),
                                            ParseTree::node(
                                                ExprNT::Factor,
                                                vec![ParseTree::leaf(
                                                    ExprT::Num,
                                                    exp_iter.next().unwrap(),
                                                )],
                                            ),
                                            ParseTree::node(
                                                ExprNT::TermPrime,
                                                vec![ParseTree::Epsilon],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            ParseTree::node(ExprNT::ExprPrime, vec![ParseTree::Epsilon]),
                        ],
                    ),
                ],
            )],
        );
        let expected_kinds = vec![ExprT::Num, ExprT::Plus, ExprT::Num, ExprT::Mult, ExprT::Num];

        // test the parser
        let grammar = grammar(expr_grammar_terminating());
        let table = grammar.generate_parse_table();
        let parser = Parser::new(table);
        let scanner = Scanner::new(input, &EXPR_RULES);
        let result = parser.parse(scanner).unwrap();

        let mut kinds = Vec::new();
        result.visit_pre_order(&mut |node| {
            if let ParseTree::Leaf(ParseTreeLeaf { terminal, .. }) = node {
                kinds.push(*terminal);
            }
        });
        assert_eq!(kinds, expected_kinds);
        assert_eq!(result, expected);

        // test the LL1 parser
        let ll1_grammar = grammar.check_ll1().unwrap();
        let ll1_table = ll1_grammar.generate_parse_table();
        let ll1_parser = LL1Parser::new(ll1_table);
        let ll1_result = ll1_parser.parse(scanner).unwrap();

        let mut kinds = Vec::new();
        ll1_result.visit_pre_order(&mut |node| {
            if let ParseTree::Leaf(ParseTreeLeaf { terminal, .. }) = node {
                kinds.push(*terminal);
            }
        });
        assert_eq!(kinds, expected_kinds);
        assert_eq!(ll1_result, expected);
    }
}
