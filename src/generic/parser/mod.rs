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
pub mod grammar;

// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
// /// An AST node that can be either a non-terminal or a terminal
// ///
// /// nodes whose only children would be Symbol::Epsilon are pruned from the AST
// pub enum AstNode<'a, NT, T, Tok>
// where
//     NT: NonTerminal,
//     T: Terminal,
//     Tok: Into<T>,
// {
//     /// A non-terminal node in the AST that contains a non-terminal symbol and a list of children
//     NonTerminal {
//         kind: NT,
//         children: Vec<AstNode<'a, NT, T, Tok>>,
//     },

//     /// A terminal node in the AST is a leaf node that contains a token
//     Terminal { kind: T, data: TokenSpan<'a, Tok> },
// }

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

impl<NT: NonTerminal, T: Terminal> Production<NT, T> {
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

#[cfg(test)]
pub mod test_utils {
    use rstest::fixture;

    // contains grammars and fixtures used for multiple tests
    use super::{
        grammar::{Grammar, NonTerminating},
        *,
    };

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
}
