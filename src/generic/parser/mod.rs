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
//! A `Grammar` can be in one of three states:
//! - `Grammar<NonTerminating>`: A grammar that may contain left-recursion.
//!     - This is the initial state of a grammar, and the only one that can be constructed directly.
//! - `Grammar<Terminating>`: A grammar that has been converted to eliminate left-recursion.
//!     - This is the result of calling `Grammar::eliminate_left_recursion`.
//! - `Grammar<LL1>`: A grammar that has been converted to be LL(1) by left-factoring.
//!    - This is the result of calling `Grammar::left_factor`.
//!
//! A parse table can only be built from a `Grammar<LL1>`.
//!
//! When converting between states, it will often be necessary to create new rules and non-terminals.
//! To keep track of these new non-terminals, we will use a `Vec<(usize, usize)>` to store the generated non-terminals and their parent (user-provided) non-terminal.
//!
//! This will allow us to collapse the generated non-terminals into their parent non-terminals when building the AST from the parse tree.
//!
//! During parsing, this information will be used along with the `TokenSpans` to rewrite the parse tree to preserve the ordering of terminals.
//!
//!

use std::collections::BTreeMap;

use super::{Scanner, TokenSpan};
pub mod grammar;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Symbol<T> {
    /// A terminal symbol, which contains a token type
    Terminal(T),
    /// A non-terminal symbol, which contains an index into the grammar's list of non-terminals
    NonTerminal(usize),
    /// Epsilon, the empty string
    Epsilon,
    /// End of file
    Eof,
}

/// marker trait for a non-terminal symbol
pub trait NonTerminal: Into<usize> {
    // non-terminals need to be able to be converted into ids
    fn to_symbol<T>(self) -> Symbol<T> {
        Symbol::NonTerminal(self.into())
    }
}

/// marker trait for a terminal symbol
pub trait Terminal: Sized + PartialEq {
    fn eof() -> Self;

    fn to_symbol(self) -> Symbol<Self> {
        if self == Self::eof() {
            Symbol::Eof
        } else {
            Symbol::Terminal(self)
        }
    }
}

/// Trait for merging two terminal symbols into a new terminal symbol of a possibly different type,
/// this is needed to enable left-factoring
pub trait Merge<R>: PartialEq {
    /// Join a prefix of terminal symbols into a new terminal symbol
    fn merge(prefix: impl AsRef<[R]>) -> Self;
}

/// A derivation that a non-terminal can expand to
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Derivation<T> {
    /// The symbols that the non-terminal expands to
    symbols: Vec<Symbol<T>>,
}

/// A production in a grammar
pub struct Production<NT, T> {
    /// The non-terminal that the production expands
    pub(crate) non_terminal: usize,
    /// The derivation that the non-terminal can expand to
    pub(crate) derivation: Derivation<T>,
    /// Phantom data to store the type of the non-terminal
    language: std::marker::PhantomData<NT>,
}

impl<NT, T: Clone> Clone for Production<NT, T> {
    fn clone(&self) -> Self {
        Self {
            non_terminal: self.non_terminal,
            derivation: self.derivation.clone(),
            language: std::marker::PhantomData,
        }
    }
}

impl<NT, T: PartialEq> PartialEq for Production<NT, T> {
    fn eq(&self, other: &Self) -> bool {
        self.non_terminal == other.non_terminal
            && self.derivation == other.derivation
            && self.language == other.language
    }
}

impl<NT, T: Eq> Eq for Production<NT, T> {}

impl<NT, T: std::fmt::Debug> std::fmt::Debug for Production<NT, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Production")
            .field("non_terminal", &self.non_terminal)
            .field("derivation", &self.derivation)
            .field("language", &self.language)
            .finish()
    }
}

type Table<'a, NT, T> = BTreeMap<usize, BTreeMap<T, Vec<&'a Production<NT, T>>>>;
type LL1Table<'a, NT, T> = BTreeMap<usize, BTreeMap<T, &'a Production<NT, T>>>;

#[derive(Debug, Clone)]
pub struct ParseTable<'a, NT, T> {
    /// The table of productions
    /// `Table[non_terminal][terminal]` = list of productions that can be expanded to
    pub table: Table<'a, NT, T>,
    /// Start symbol
    pub start_symbol: usize,
}

#[derive(Debug, Clone)]
pub struct LL1ParseTable<'a, NT, T> {
    /// The table of productions
    /// `Table[non_terminal][terminal]` = index of the production that can be expanded to
    pub table: LL1Table<'a, NT, T>,
    /// Start symbol
    pub start_symbol: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseTree<'a, T, Token> {
    Node {
        non_terminal: usize,
        children: Vec<ParseTree<'a, T, Token>>,
    },
    Leaf {
        terminal: T,
        token_span: TokenSpan<'a, Token>,
    },
    Epsilon,
}

#[derive(thiserror::Error, Debug)]
pub enum ParseError<'a, Terminal, Token> {
    #[error("Expected end of input, found {0:?}")]
    ExpectedEof(TokenSpan<'a, Token>),
    #[error("Unexpected end of input, expected symbol {0:?}")]
    UnexpectedEof(Symbol<Terminal>),
    #[error("No production found for non-terminal {0} and lookahead {1:?}")]
    NoProduction(usize, Terminal),
    #[error("Unexpected token {0:?}, expected symbol {1:?}")]
    UnexpectedToken(TokenSpan<'a, Token>, Symbol<Terminal>),
    #[error("Unexpected end of stack")]
    UnexpectedEndOfStack,
}

#[derive(Debug, Clone)]
pub struct Parser<'a, 'b, NT, T> {
    pub table: &'a ParseTable<'b, NT, T>,
}

#[derive(Debug, Clone)]
pub struct LL1Parser<'a, 'b, NT, T> {
    pub table: &'a LL1ParseTable<'b, NT, T>,
}

/////////////////////////////////////////////////
// Macros
/////////////////////////////////////////////////
#[macro_export]
macro_rules! derivation {
    [$($symbol:expr),*] => {
        Derivation::new(vec![$($symbol.to_symbol()),*])
    };
}

/////////////////////////////////////////////////
// Implementations
/////////////////////////////////////////////////

impl NonTerminal for usize {}

impl<T: Eq> Derivation<T> {
    /// Create a new derivation from the given symbols
    ///
    /// # Panics
    ///
    /// Panics if the symbols contain epsilon and other symbols
    #[must_use]
    pub fn new(symbols: Vec<Symbol<T>>) -> Self {
        if symbols.contains(&Symbol::Epsilon) {
            assert!(
                symbols.len() == 1,
                "Derivation contained epsilon among other symbols"
            );
        }
        Self { symbols }
    }
}

impl<T> Symbol<T> {
    #[must_use]
    pub const fn to_symbol(self) -> Self {
        self
    }
}

impl<T> From<Vec<Symbol<T>>> for Derivation<T> {
    fn from(symbols: Vec<Symbol<T>>) -> Self {
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
    pub fn new<I: Into<usize>, D: Into<Derivation<T>>>(non_terminal: I, derivation: D) -> Self {
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
    pub fn new(start_symbol: NT) -> Self {
        Self {
            table: BTreeMap::new(),
            start_symbol: start_symbol.into(),
        }
    }

    /// Get the productions for the given non-terminal and lookahead
    /// Returns None if no productions are found
    pub fn get_productions(&self, nt: usize, lookahead: T) -> Option<&[&Production<NT, T>]> {
        let ret = self
            .table
            .get(&nt)
            .and_then(|m| m.get(&lookahead))
            .map(|v| v.as_slice());
        assert!(ret.is_none() || !ret.unwrap().is_empty());
        ret
    }

    /// Add a production to the parse table
    pub fn add_production<'b: 'a>(
        &mut self,
        nt: usize,
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
    T: Terminal + Copy + Ord + std::fmt::Debug,
{
    /// Create a new, empty parse table
    #[must_use]
    pub fn new(start_symbol: NT) -> Self {
        Self {
            table: BTreeMap::new(),
            start_symbol: start_symbol.into(),
        }
    }

    /// Get the production for the given non-terminal and lookahead
    /// Returns None if no production is found
    pub fn get_production(&self, nt: usize, lookahead: T) -> Option<&Production<NT, T>> {
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
        nt: usize,
        lookahead: T,
        production: &'b Production<NT, T>,
    ) {
        assert!(
            self.table
                .entry(nt)
                .or_default()
                .insert(lookahead, production)
                .is_none(),
            "Table already contains a production for non-terminal {nt} and lookahead {lookahead:?}",
        );
    }
}

impl<'a, T, Token> ParseTree<'a, T, Token>
where
    T: Terminal + Copy,
    Token: Copy,
{
    /// Create a new parse tree node
    #[must_use]
    fn node(non_terminal: impl Into<usize>, children: Vec<Self>) -> Self {
        Self::Node {
            non_terminal: non_terminal.into(),
            children,
        }
    }

    /// Create a new parse tree leaf
    #[must_use]
    fn leaf(terminal: T, token_span: TokenSpan<'a, Token>) -> Self {
        Self::Leaf {
            terminal,
            token_span,
        }
    }

    /// Perform a post-order traversal of the parse tree
    /// and apply the given function to each node
    pub fn visit_post_order<F>(&self, f: &mut F)
    where
        F: FnMut(&Self),
    {
        match self {
            Self::Node { children, .. } => {
                for child in children {
                    child.visit_post_order(f);
                }
            }
            Self::Leaf { .. } | Self::Epsilon => {}
        }
        f(self);
    }

    /// Perform a pre-order traversal of the parse tree
    /// and apply the given function to each node
    pub fn visit_pre_order<F>(&self, f: &mut F)
    where
        F: FnMut(&Self),
    {
        f(self);
        match self {
            Self::Node { children, .. } => {
                for child in children {
                    child.visit_pre_order(f);
                }
            }
            Self::Leaf { .. } | Self::Epsilon => {}
        }
    }

    // TODO: need a way to rewrite the parse tree to remove the generated non-terminals
    // while preserving the order of terminals and syntax
}

impl<'a, 'b: 'a, NT, T> Parser<'a, 'b, NT, T>
where
    NT: NonTerminal,
    T: Terminal + Copy + Ord,
{
    /// Create a new parser from the given parse table
    #[must_use]
    pub const fn new(table: &'a ParseTable<'b, NT, T>) -> Self {
        Self { table }
    }

    /// Parse the given input using the parser's parse table
    ///
    /// This implementation combines recursive descent with table-driven parsing,
    /// recursion is needed to enable backtracking but we can use the table to avoid
    /// the need for hand-written recursive descent functions.
    #[must_use]
    pub fn parse<'c: 'a, Token>(
        &'a self,
        scanner: &'c Scanner<'a, Token>,
    ) -> Result<ParseTree<'a, T, Token>, ParseError<'c, T, Token>>
    where
        Token: Into<T> + Copy,
    {
        /// Inner recursive descent function
        ///
        /// returns `Ok(Some(ParseTree))` if successful, `Ok(None)` if the symbol is epsilon,
        /// and `Err(ParseError)` if an error occurs
        fn inner<'a, 'b: 'a, 'c, NT, T, Token>(
            // reference to the parser's parse table
            table: &'a ParseTable<'b, NT, T>,
            // tokens from the scanner
            tokens: &[TokenSpan<'c, Token>],
            // index into the tokens, used to keep track of the current token
            index: &mut usize,
            // the current symbol we're trying to match
            symbol: Symbol<T>,
        ) -> Result<ParseTree<'c, T, Token>, ParseError<'c, T, Token>>
        where
            NT: NonTerminal,
            T: Terminal + Copy + Ord,
            Token: Into<T> + Copy,
        {
            match symbol {
                Symbol::Terminal(t) => {
                    let token = tokens.get(*index).ok_or(ParseError::UnexpectedEndOfStack)?;
                    if t == token.kind.into() {
                        *index += 1;
                        Ok(ParseTree::leaf(t, *token))
                    } else {
                        Err(ParseError::UnexpectedToken(*token, symbol))
                    }
                }
                Symbol::NonTerminal(nt) => {
                    let lookahead = tokens.get(*index).map(|t| t.kind.into());
                    let productions = table.get_productions(nt, lookahead.unwrap_or_else(T::eof));

                    if let Some(productions) = productions {
                        for production in productions {
                            let mut i = *index;
                            let mut children = Vec::new();
                            let error: Result<(), ParseError<'_, T, Token>> = {
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

                            match error {
                                Ok(()) => {
                                    *index = i;
                                    return Ok(ParseTree::node(nt, children));
                                }
                                Err(_) => {}
                            }
                        }

                        Err(ParseError::NoProduction(
                            nt,
                            lookahead.unwrap_or_else(T::eof),
                        ))
                    } else {
                        Err(ParseError::NoProduction(
                            nt,
                            lookahead.unwrap_or_else(T::eof),
                        ))
                    }
                }
                Symbol::Epsilon => Ok(ParseTree::Epsilon),
                Symbol::Eof => {
                    if *index == tokens.len() {
                        Ok(ParseTree::Epsilon)
                    } else {
                        Err(ParseError::ExpectedEof(tokens[*index]))
                    }
                }
            }
        }

        let tokens = scanner
            .iter()
            .filter(|t| !t.is_whitespace)
            .collect::<Vec<_>>();

        let mut index = 0;

        let tree = inner(
            self.table,
            &tokens,
            &mut index,
            Symbol::NonTerminal(self.table.start_symbol),
        )?;

        if index == tokens.len() {
            Ok(tree)
        } else {
            Err(ParseError::ExpectedEof(tokens[index]))
        }
    }
}

impl<'a, 'b: 'a, NT, T> LL1Parser<'a, 'b, NT, T>
where
    NT: NonTerminal,
    T: Terminal + Copy + Ord + std::fmt::Debug,
{
    /// Create a new parser from the given parse table
    #[must_use]
    pub const fn new(table: &'a LL1ParseTable<'b, NT, T>) -> Self {
        Self { table }
    }

    /// Parse the given input using the parser's parse table
    ///
    /// TODO: return the parse tree instead of just true/false
    ///
    /// # Errors
    ///
    /// Returns a `ParseError` if the input cannot be parsed
    pub fn parse<'c, Token>(
        &'a self,
        scanner: &'c Scanner<Token>,
    ) -> Result<(), ParseError<'c, T, Token>>
    where
        Token: Into<T> + Copy,
    {
        let mut stack: Vec<Symbol<T>> = vec![Symbol::NonTerminal(self.table.start_symbol)];
        let mut input = scanner.iter().filter(|t| !t.is_whitespace).peekable();
        let mut eof = false;

        loop {
            let lookahead = input.peek().copied();
            let top = stack.pop();

            match (top, lookahead) {
                (Some(Symbol::Terminal(t)), Some(token)) if t == token.kind.into() => {
                    input.next();
                    eof = token.is_eof;
                }
                (Some(Symbol::NonTerminal(nt)), token) => {
                    let kind = token.map(|t| t.kind.into()).unwrap_or_else(|| T::eof());

                    if let Some(production) = self.table.get_production(nt, kind) {
                        stack.extend(
                            production
                                .derivation
                                .symbols
                                .iter()
                                .rev()
                                .filter(|&&s| s != Symbol::Epsilon)
                                .copied(),
                        );
                    } else {
                        return Err(ParseError::NoProduction(nt, kind));
                    }
                    eof |= token.map(|t| t.is_eof).unwrap_or_default();
                }
                (Some(Symbol::Eof), Some(token)) => {
                    return Err(ParseError::ExpectedEof(token));
                }
                (Some(Symbol::Eof), _) | (None, None) => {
                    assert!(eof);
                    return Ok(());
                }
                (Some(s @ Symbol::Terminal(_)), Some(token)) => {
                    return Err(ParseError::UnexpectedToken(token, s));
                }
                (Some(s @ Symbol::Terminal(_)), None) => {
                    return Err(ParseError::UnexpectedEof(s));
                }
                (None, Some(token)) => {
                    return Err(ParseError::UnexpectedToken(token, Symbol::Eof));
                }
                (Some(Symbol::Epsilon), _) => {
                    stack.pop();
                }
            }
        }
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
    use crate::generic::{test_utils::*, LL1Parser, Scanner};
    use pretty_assertions::assert_eq;
    use rstest::{fixture, rstest};

    #[fixture]
    fn grammar(
        expr_grammar_terminating: Grammar<ExprNT, ExprT, NonTerminating>,
    ) -> Grammar<ExprNT, ExprT, TerminatingReady<ExprT>> {
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
        grammar: Grammar<ExprNT, ExprT, TerminatingReady<ExprT>>,
        #[case] input: &str,
        #[case] expected: bool,
    ) {
        let grammar = grammar.check_ll1().unwrap();
        let table = grammar.generate_parse_table();

        let parser = LL1Parser::new(&table);

        let scanner = Scanner::new(input, &EXPR_RULES);

        let result = parser.parse(&scanner);

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

        let mut exp_iter = scanner.iter().filter(|t| !t.is_whitespace);

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

        let grammar = grammar(expr_grammar_terminating());
        let table = grammar.generate_parse_table();

        let parser = Parser::new(&table);
        let scanner = Scanner::new(input, &EXPR_RULES);

        let result = parser.parse(&scanner).unwrap();

        let expected_kinds = vec![ExprT::Num, ExprT::Plus, ExprT::Num, ExprT::Mult, ExprT::Num];

        let mut kinds = Vec::new();
        result.visit_pre_order(&mut |node| {
            if let ParseTree::Leaf { terminal, .. } = node {
                kinds.push(*terminal);
            }
        });

        assert_eq!(kinds, expected_kinds);

        assert_eq!(result, expected);
    }
}
