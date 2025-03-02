use crate::derivation;

use super::{Derivation, LL1ParseTable, Merge, NonTerminal, ParseTable, Production, Symbol, Terminal};

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
};

/// A grammar that defines a set of rules for a language
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar<NT, T, State = NonTerminating> {
    /// The non-terminal in the grammar that represents the start symbol,
    /// and the root of the AST
    start_symbol: NT,
    /// The production of the grammar
    /// Each production is a pair of a non-terminal and a derivation of that non-terminal
    productions: Vec<Production<NT, T>>,
    /// The state of the grammar
    state: State,
}

/// Marker struct for a grammar that may contain left-recursion
pub struct NonTerminating;

/// Marker struct for a grammar that has been converted to eliminate left-recursion
pub struct Terminating {
    /// A list of the generated non-terminals, and their parent non-terminal.
    /// This is used to recover a usable AST from the parse tree by collapsing the
    /// generated non-terminals into their parents.
    ///
    /// `(generated_non_terminal, parent_non_terminal)`
    generated_non_terminals: Vec<(usize, usize)>,
}

/// Marker struct for a Terminating grammar, that is not LL(1) but is ready to use for parsing.
/// Created by generating the FIRST, FOLLOW, and FIRST+ sets from a `Grammar<_,Terminating>`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminatingReady<T>
where
    T: Clone + Copy + Eq,
{
    /// A list of the generated non-terminals, and their parent non-terminal.
    /// This is used to recover a usable AST from the parse tree by collapsing the
    /// generated non-terminals into their parents.
    ///
    /// `(generated_non_terminal, parent_non_terminal)`
    generated_non_terminals: Vec<(usize, usize)>,
    /// The first set's for each non-terminal.
    /// This is a map from non-terminals to the set of terminals that can appear as the first symbol
    /// of a sentence derived from that non-terminal.
    first_sets: FirstSets<T>,
    /// The follow set's for each non-terminal.
    /// The follow set of a non-terminal is the set of symbols in the grammar that appear immediately after the non-terminal
    follow_sets: FollowSets<T>,
    /// The first+ set's for each production.
    /// The first+ set of a production `A -> α` is:
    /// - FIRST(α) if FIRST(α) does not contain epsilon
    /// - FIRST(α) ∪ FOLLOW(A) if FIRST(α) contains epsilon
    first_plus_sets: FirstPlusSets<T>,
}

/// Marker struct for a grammar that has been converted to be LL(1) by left-factoring
pub struct LL1<T>
where
    T: Clone + Copy + Eq,
{
    /// A list of the generated non-terminals, and their parent non-terminal.
    /// This is used to recover a usable AST from the parse tree by collapsing the
    /// generated non-terminals into their parents.
    ///
    /// `(generated_non_terminal, parent_non_terminal)`
    generated_non_terminals: Vec<(usize, usize)>,
    /// The first+ set's for each production.
    /// The first+ set of a production `A -> α` is:
    /// - FIRST(α) if FIRST(α) does not contain epsilon
    /// - FIRST(α) ∪ FOLLOW(A) if FIRST(α) contains epsilon
    first_plus_sets: FirstPlusSets<T>,
}

#[derive(thiserror::Error, Debug)]
pub enum GrammarError {
    #[error(
        "Non-Terminal {0}, which appears in producion {1}, does not have any rules expanding it"
    )]
    NonTerminalWithoutRules(usize, usize),
    #[error(
        "A Derivation for Non-Terminal {0}, given in production {1}, contains no symbols, perhaps you meant to use epsilon"
    )]
    DerivationWithoutSymbols(usize, usize),
    #[error("Non-Terminal {0} expands to itself in rule {1}")]
    NonTerminalExpandsToItself(usize, usize),
    #[error("Start symbol {0} does not have any expansions")]
    GoalWithoutExpansions(usize),
    #[error("Grammar has no rules")]
    NoRules,
}

pub type GrammarResult<T> = Result<T, GrammarError>;

/// The FIRST set of a non-terminal `A` is the set of terminals that can appear as the first symbol of a sentence derived from `A`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstSet<T>
where
    T: Eq,
{
    set: BTreeSet<T>,
    contains_epsilon: bool,
}

impl<T: Eq> Default for FirstSet<T> {
    fn default() -> Self {
        Self {
            set: BTreeSet::new(),
            contains_epsilon: false,
        }
    }
}

/// The FIRST sets of all non-terminals in the grammar
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstSets<T>
where
    T: Eq,
{
    sets: BTreeMap<usize, FirstSet<T>>,
}

impl<T> Default for FirstSets<T>
where
    T: Eq,
{
    fn default() -> Self {
        Self {
            sets: BTreeMap::new(),
        }
    }
}

/// The FOLLOW set of a non-terminal `A` is the set of symbols in the grammar that appear immediately after the non-terminal
/// in some derivation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FollowSet<T>
where
    T: Eq,
{
    set: BTreeSet<T>,
}

impl<T: Eq> Default for FollowSet<T> {
    fn default() -> Self {
        Self {
            set: BTreeSet::new(),
        }
    }
}

/// The FOLLOW sets of all non-terminals in the grammar
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FollowSets<T>
where
    T: Eq,
{
    sets: BTreeMap<usize, FollowSet<T>>,
}

impl<T> Default for FollowSets<T>
where
    T: Eq,
{
    fn default() -> Self {
        Self {
            sets: BTreeMap::new(),
        }
    }
}

/// The FIRST+ set of a production `A -> α` is:
/// - FIRST(α) if FIRST(α) does not contain epsilon
/// - FIRST(α) ∪ FOLLOW(A) if FIRST(α) contains epsilon
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstPlusSet<T>
where
    T: Eq,
{
    set: BTreeSet<T>,
    contains_epsilon: bool,
}

impl<T: Eq> Default for FirstPlusSet<T> {
    fn default() -> Self {
        Self {
            set: BTreeSet::new(),
            contains_epsilon: false,
        }
    }
}

/// The FIRST+ sets of all productions in the grammar
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstPlusSets<T>
where
    T: Eq,
{
    sets: BTreeMap<usize, BTreeMap<Derivation<T>, FirstPlusSet<T>>>,
}

impl<T> Default for FirstPlusSets<T>
where
    T: Eq,
{
    fn default() -> Self {
        Self {
            sets: BTreeMap::new(),
        }
    }
}

/////////////////////////////////////////////////
// Implementations
/////////////////////////////////////////////////

impl<T> FirstSet<T>
where
    T: Eq + Clone + Copy + Ord + Terminal,
{
    /// Create a new empty FIRST set
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove epsilon from the FIRST set
    const fn remove_epsilon(&mut self) {
        self.contains_epsilon = false;
    }

    /// Check if the FIRST set contains epsilon
    #[must_use]
    pub const fn contains_epsilon(&self) -> bool {
        self.contains_epsilon
    }

    /// Add epsilon to the FIRST set and return a new FIRST set with epsilon
    #[cfg(test)]
    #[must_use]
    fn with_epsilon(&self) -> Cow<Self> {
        if self.contains_epsilon {
            Cow::Borrowed(self)
        } else {
            Cow::Owned(Self {
                set: self.set.clone(),
                contains_epsilon: true,
            })
        }
    }

    /// Append the FIRST set of another symbol to this FIRST set
    fn append(&mut self, mut other: Self) {
        self.set.append(&mut other.set);
        self.contains_epsilon |= other.contains_epsilon;
    }

    /// The FIRST set of a terminal is always just the terminal itself
    #[must_use]
    fn from_terminal(symbol: T) -> Self {
        let mut set = BTreeSet::new();
        set.insert(symbol);
        Self {
            set,
            contains_epsilon: false,
        }
    }

    /// The FIRST set of epsilon is just epsilon
    #[must_use]
    const fn epsilon() -> Self {
        Self {
            set: BTreeSet::new(),
            contains_epsilon: true,
        }
    }

    /// Get the length of the FIRST set
    #[must_use]
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// Check if the FIRST set is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }
}

#[cfg(test)]
impl<T: Eq + Ord> FromIterator<T> for FirstSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            set: BTreeSet::from_iter(iter),
            contains_epsilon: false,
        }
    }
}

impl<T> FirstSets<T>
where
    T: Eq + Clone + Copy + Ord + Terminal,
{
    /// Create a new empty set of FIRST sets
    #[must_use]
    fn new() -> Self {
        Self::default()
    }

    /// Get the FIRST set for the given non-terminal, if it exists
    #[must_use]
    pub fn get(&self, nt: &usize) -> Option<&FirstSet<T>> {
        self.sets.get(nt)
    }

    /// Get the FIRST set for the given Symbol
    #[must_use]
    pub fn get_symbol(&self, symbol: &Symbol<T>) -> Option<Cow<FirstSet<T>>> {
        match symbol {
            Symbol::NonTerminal(nt) => self.get(nt).map(Cow::Borrowed),
            Symbol::Terminal(t) => Some(Cow::Owned(FirstSet::from_terminal(*t))),
            Symbol::Epsilon => Some(Cow::Owned(FirstSet::epsilon())),
            Symbol::Eof => Some(Cow::Owned(FirstSet::from_terminal(T::eof()))),
        }
    }

    /// Get number of FIRST sets
    #[must_use]
    pub fn len(&self) -> usize {
        self.sets.len()
    }

    /// Check if there are no FIRST sets
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    /// Get an iterator over the FIRST sets
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &FirstSet<T>)> {
        self.sets.iter()
    }

    /// Get the first set of a string of symbols
    ///
    /// For a string of symbols `s=β1,β2,β3...βk`, we define FIRST(s) as the union
    /// of the FIRST sets for `β1,β2,β3...βn`, where βn is the first symbol whose FIRST set does not contain epsilon,
    /// and epsilon is in FIRST(s) iff it is in the FIRST set of all the `βi`, 1 <= i <= k
    ///
    /// # Returns
    ///
    /// None if any of the symbols in the string do not have a FIRST set in `self`
    #[must_use]
    pub fn get_first_set(&self, symbols: &[Symbol<T>]) -> Option<FirstSet<T>> {
        debug_assert!(!symbols.is_empty());
        let mut first_set = FirstSet::new();
        for set in symbols.iter().map(|s| self.get_symbol(s)) {
            match set {
                Some(set) if set.contains_epsilon() => {
                    first_set.append(set.into_owned());
                }
                Some(set) => {
                    first_set.append(set.into_owned());
                    first_set.remove_epsilon();
                    break;
                }
                None => return None,
            }
        }
        Some(first_set)
    }
}

#[cfg(test)]
impl<T> FromIterator<(usize, FirstSet<T>)> for FirstSets<T>
where
    T: Eq + Ord,
{
    fn from_iter<I: IntoIterator<Item = (usize, FirstSet<T>)>>(iter: I) -> Self {
        Self {
            sets: BTreeMap::from_iter(iter),
        }
    }
}

impl<T> FollowSet<T>
where
    T: Eq + Ord + Clone + Copy + Terminal,
{
    /// Create a new empty FOLLOW set
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the length of the FIRST set
    #[must_use]
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// Check if the FIRST set is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    pub fn union<'a, 'b: 'a>(
        &'b self,
        other: &'a Self,
    ) -> std::collections::btree_set::Union<'a, T> {
        self.set.union(&other.set)
    }
}

impl<T: Eq + Ord> FromIterator<T> for FollowSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            set: BTreeSet::from_iter(iter),
        }
    }
}

impl<T> FollowSets<T>
where
    T: Eq + Ord + Clone + Copy + Terminal,
{
    /// Get the FOLLOW set for the given non-terminal, if it exists
    #[must_use]
    pub fn get(&self, nt: &usize) -> Option<&FollowSet<T>> {
        self.sets.get(nt)
    }

    /// Get number of FOLLOW sets
    #[must_use]
    pub fn len(&self) -> usize {
        self.sets.len()
    }

    /// Check if there are no FOLLOW sets
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    /// Get an iterator over the FOLLOW sets
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &FollowSet<T>)> {
        self.sets.iter()
    }
}

#[cfg(test)]
impl<T> FromIterator<(usize, FollowSet<T>)> for FollowSets<T>
where
    T: Eq + Ord,
{
    fn from_iter<I: IntoIterator<Item = (usize, FollowSet<T>)>>(iter: I) -> Self {
        Self {
            sets: BTreeMap::from_iter(iter),
        }
    }
}

impl<T> FirstPlusSet<T>
where
    T: Eq + Ord + Clone + Copy + Terminal,
{
    /// Create a new empty FIRST+ set
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add epsilon to the FIRST+ set and return a new FIRST+ set with epsilon
    #[cfg(test)]
    #[must_use]
    fn with_epsilon(&self) -> Cow<Self> {
        if self.contains_epsilon {
            Cow::Borrowed(self)
        } else {
            Cow::Owned(Self {
                set: self.set.clone(),
                contains_epsilon: true,
            })
        }
    }

    /// Get the length of the FIRST+ set
    #[must_use]
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// Check if the FIRST+ set is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }
}

impl<T> FromIterator<T> for FirstPlusSet<T>
where
    T: Eq + Ord,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            set: BTreeSet::from_iter(iter),
            contains_epsilon: false,
        }
    }
}

impl<T> From<(FirstSet<T>, FollowSet<T>)> for FirstPlusSet<T>
where
    T: Eq + Ord,
{
    fn from((mut first, mut follow): (FirstSet<T>, FollowSet<T>)) -> Self {
        let mut set = BTreeSet::new();
        let contains_epsilon = first.contains_epsilon;
        set.append(&mut first.set);
        set.append(&mut follow.set);
        Self {
            set,
            contains_epsilon,
        }
    }
}

impl<T> From<FirstSet<T>> for FirstPlusSet<T>
where
    T: Eq,
{
    fn from(first: FirstSet<T>) -> Self {
        Self {
            set: first.set,
            contains_epsilon: first.contains_epsilon,
        }
    }
}

impl<T> FirstPlusSets<T>
where
    T: Eq + Ord + Clone + Copy + Terminal,
{
    /// Get the FIRST+ set for the given production, if there is one
    #[must_use]
    pub fn get(&self, nt: &usize, derivation: &Derivation<T>) -> Option<&FirstPlusSet<T>> {
        self.sets.get(nt).and_then(|m| m.get(derivation))
    }

    /// Check if the FIRST+ set of a production contains a given lookahead symbol
    /// If the FIRST+ set contains epsilon, this will also check if the FOLLOW set of the non-terminal contains the lookahead symbol
    #[must_use]
    pub fn contains(&self, nt: &usize, derivation: &Derivation<T>, lookahead: &T) -> bool {
        self.get(nt, derivation)
            .is_some_and(|set| set.set.contains(lookahead))
    }
}

#[cfg(test)]
impl<T> FromIterator<(usize, BTreeMap<Derivation<T>, FirstPlusSet<T>>)> for FirstPlusSets<T>
where
    T: Eq + Ord,
{
    fn from_iter<I: IntoIterator<Item = (usize, BTreeMap<Derivation<T>, FirstPlusSet<T>>)>>(
        iter: I,
    ) -> Self {
        Self {
            sets: BTreeMap::from_iter(iter),
        }
    }
}

impl<NT, T> Grammar<NT, T, NonTerminating>
where
    T: Copy + Eq + Terminal,
    NT: NonTerminal+ Copy,
{
    /// Create a new grammar from the given non-terminals, terminals, start symbol, and rules
    ///
    /// # Errors
    ///
    /// This will return an error if the grammar is invalid, such as if a non-terminal does not have any expansions,
    /// or if a non-terminal expands only to itself.
    pub fn new<P>(start_symbol: NT, productions: Vec<P>) -> GrammarResult<Self>
    where
        P: Into<Production<NT, T>>,
    {
        fn inner<NT: NonTerminal +Copy, T>(
            start_symbol: NT,
            productions: Vec<Production<NT, T>>,
        ) -> GrammarResult<Grammar<NT, T, NonTerminating>> {
            if productions.is_empty() {
                return Err(GrammarError::NoRules);
            }

            // Non-Terminals that have expansions
            let mut non_terminals = BTreeSet::new();

            for (i, production) in productions.iter().enumerate() {
                non_terminals.insert(production.non_terminal);
                if production.derivation.symbols.is_empty() {
                    return Err(GrammarError::DerivationWithoutSymbols(
                        production.non_terminal,
                        i,
                    ));
                }
            }

            if !non_terminals.contains(&start_symbol.into()) {
                return Err(GrammarError::GoalWithoutExpansions(start_symbol.into()));
            }

            // now we go through every rule again, and ensure that every non-terminal listed in a derivartion has
            // an expansion
            for (i, production) in productions.iter().enumerate() {
                // make sure the nt doesn't only expand to itself
                if production.derivation.symbols.len() == 1
                    && matches!(
                        production.derivation.symbols.first(),
                        Some(&Symbol::NonTerminal(nt)) if nt == production.non_terminal,
                    )
                {
                    return Err(GrammarError::NonTerminalExpandsToItself(
                        production.non_terminal,
                        i,
                    ));
                }
                // make sure every non-terminal in the derivation has an expansion
                if !production
                    .derivation
                    .symbols
                    .iter()
                    .all(|symbol| match symbol {
                        Symbol::NonTerminal(nt) => non_terminals.contains(nt),
                        _ => true,
                    })
                {
                    return Err(GrammarError::NonTerminalWithoutRules(
                        production.non_terminal,
                        i,
                    ));
                }
            }

            // If we've made it this far, the grammar is valid
            Ok(Grammar {
                start_symbol,
                productions,
                state: NonTerminating,
            })
        }

        inner(
            start_symbol,
            productions.into_iter().map(Into::into).collect(),
        )
    }

    /// Eliminate left-recursion from the grammar, returns a new grammar with the left-recursion removed
    ///
    /// We can eliminate left-recursion using the algorithm described in Engineering a Compiler, 2nd Edition, figure 3.6
    /// ```text
    /// impose an order on the nonterminals. A1, A2, . . . , An
    /// for i ← 1 to n do;
    ///     for j ← 1 to i - 1 do;
    ///         if ∃ a production Ai -> Aj γ
    ///             then replace Ai→Aj γ with one or more productions that expand Aj
    ///     end;
    ///     rewrite the productions to eliminate any direct left recursion on Ai
    /// end;
    /// ```
    ///
    /// An obvious question one may have is "how do we recover the original grammar when building an AST from the parse tree?"
    /// We can do this by collapsing the generated non-terminals into their parent non-terminals, which we will keep track of in the `Terminating` state.
    #[must_use]
    pub fn eliminate_left_recursion(mut self) -> Grammar<NT, T, Terminating> {
        debug_assert!(!self.productions.is_empty());

        // order productions by the non-terminal they expand
        self.productions.sort_by_key(|p| p.non_terminal);

        let mut generated_non_terminals = Vec::new();

        // This is the id of the next non-terminal we will generate, it will be the highest current non-terminal id + 1
        let mut next_non_terminal = self
            .productions
            .last()
            .map(|p| p.non_terminal)
            .unwrap_or_default()
            + 1;

        // we will be changing the number of productions, but we only care about the original productions
        // we also only ever append to the end of the list, so we only want to iterate over the original productions

        let rules = self.productions.clone();
        let mut rules = rules
            .chunk_by(|a, b| a.non_terminal == b.non_terminal)
            .map(|chunk| (&chunk[0].non_terminal, chunk.to_vec()))
            .collect::<Vec<_>>();

        self.productions.clear();

        let n = rules.len();
        for i in 0..n {
            for j in 0..i {
                // eliminate indirect recursion on A_i
                // Find all rules of the form A_i -> A_j γ

                // we're iterating over productions, not rules, so we need to group the productions by the non-terminal they expand

                let mut new_productions = Vec::new();
                for production in &rules[i].1 {
                    match production.derivation.symbols.split_first() {
                        Some((Symbol::NonTerminal(nt), gamma))
                            if *nt == self.productions[j].non_terminal =>
                        {
                            // Replace A_i -> A_j γ with A_i -> δ γ for each δ in A_j
                            for aj_production in &rules[j].1 {
                                let mut new_symbols = aj_production.derivation.symbols.clone();
                                new_symbols.extend_from_slice(gamma);
                                new_productions.push(Production::new(*rules[i].0, Derivation::new(new_symbols)));
                            }
                        }
                        _ => new_productions.push(production.clone()),
                    }
                }
                // Add the new derivations to the rule
                rules[i].1 = new_productions;
            }
            // Rewrite the productions to eliminate any direct left recursion on A_i
            // A_i -> A_i α
            //      | β
            // becomes
            // A_i  -> β A_i'
            // A_i' -> α A_i'
            //       | ε

            if rules[i].1.iter().any(|d| {
                matches!(
                    d.derivation.symbols.first(),
                    Some(Symbol::NonTerminal(nt)) if nt == rules[i].0
                )
            }) {
                // we have left-recursion on this non-terminal
                // we need to split the derivations into two groups, those that start with A_i and those that don't
                generated_non_terminals.push((next_non_terminal, *rules[i].0));

                for production in &rules[i].1 {
                    match production.derivation.symbols.first() {
                        Some(Symbol::NonTerminal(nt)) if nt == rules[i].0 => {
                            debug_assert!(production.derivation.symbols.len() > 1);
                            // remove the first symbol (A_i) from the derivation
                            let mut d = production.derivation.clone();
                            d.symbols.remove(0);
                            // add the new non-terminal to the end of the derivation
                            d.symbols.push(Symbol::NonTerminal(next_non_terminal));
                            // add the modified derivation to the new non-terminal
                            self.productions.push(Production::new(next_non_terminal, d));
                        }
                        _ => {
                            // add the new non-terminal to the end of the derivation
                            let mut d = production.derivation.clone();
                            d.symbols.push(Symbol::NonTerminal(next_non_terminal));
                            self.productions.push(Production::new(*rules[i].0, d));
                        }
                    }
                }

                // add Epsilon as a derivation for the new non-terminal
                self.productions.push(Production::new(
                    next_non_terminal,
                    derivation![Symbol::Epsilon],
                ));

                next_non_terminal += 1;
            } else {
                // no left recursion on this non-terminal
                for production in &rules[i].1 {
                    self.productions.push(Production::new(
                        production.non_terminal,
                        production.derivation.clone(),
                    ));
                }
            }
        }

        self.productions.sort_by_key(|p| p.non_terminal);

        // we have eliminated all left-recursion, so we can now convert the grammar to a Terminating state
        Grammar {
            start_symbol: self.start_symbol,
            productions: self.productions,
            state: Terminating {
                generated_non_terminals,
            },
        }
    }
}


impl<NT, T> Grammar<NT, T, Terminating>
where
    T: Clone + Copy + Eq + Ord + std::fmt::Debug + Terminal,
    NT: NonTerminal + Copy,
{
    /// Get the mapping of generated non-terminals to their parent non-terminals
    #[must_use]
    pub fn generated_non_terminals(&self) -> &[(usize, usize)] {
        &self.state.generated_non_terminals
    }

    /// Generate the FIRST, FOLLOW, and FIRST+ sets for the grammar, and convert it to a `TerminatingReady` grammar
    ///
    /// This can be used by a parser to generate a parse table for a grammar that is not LL(1), and parse input using that parse table with backtracking.
    #[must_use]
    pub fn generate_sets(self) -> Grammar<NT, T, TerminatingReady<T>> {
        let first_sets = generate_first_set(&self.productions);
        let follow_sets = generate_follow_set(&self.productions, self.start_symbol.into(), &first_sets);
        let first_plus_sets = generate_first_plus_set(&self.productions, &first_sets, &follow_sets);

        Grammar {
            start_symbol: self.start_symbol,
            productions: self.productions,
            state: TerminatingReady {
                generated_non_terminals: self.state.generated_non_terminals,
                first_sets,
                follow_sets,
                first_plus_sets,
            },
        }
    }

    /// Left-factor the grammar to make it LL(1), generating the FIRST, FOLLOW, and FIRST+ sets along the way
    ///
    /// # algorithm
    ///
    /// I can't find a generic algorithm for this, but the idea is as follows:
    ///
    /// ```psuedocode
    /// For each NT in the grammar // for each rule
    ///
    /// ```
    ///
    /// # Errors
    ///
    /// If the grammar cannot be converted to an LL(1) grammar, this will fail with an error
    ///
    /// TODO: implement a more general backtracking parser that can parse non-ll(1) grammars,
    pub fn left_factor<O>(self) -> GrammarResult<Grammar<O, LL1<T>>>
    where
        // Must be able to merge terminal symbols into a single terminal
        // symbol in order to perform left-factoring.
        O: Merge<T>,
    {
        todo!()
    }
}

/// Helper function to generate the FIRST set for a grammar.
///
/// The FIRST set of a non-terminal `A` is the set of terminals that can appear as the first symbol of a sentence derived from `A`.
///
/// We can generate the FIRST sets for symbols in a grammar using the algorithm described in Engineering a Compiler, 2nd Edition, figure 3.7:
/// ```text
/// for each α ∈ (T ∪ eof ∪ epsilon) do;
///     FIRST(α) ← α;
/// end;
/// for each A ∈ NT do;
///     FIRST(A) ← ∅;
/// end;
///
/// while (FIRST sets are still changing) do;
///     for each p ∈ P, where p has the form A -> β do;
///         if β is β1,β2,...,βk, where βi ∈ T ∪ NT, then begin;
///             rhs ← FIRST(β1) − {epsilon};
///             i ← 1;
///             while (epsilon ∈ FIRST(βi) and i ≤ k-1) do;
///                 rhs ← rhs ∪ (FIRST(βi+1) − {epsilon});
///                 i ← i + 1;
///             end;
///         end;
///         if i = k and epsilon ∈ FIRST(βk)
///             then rhs ← rhs ∪ {epsilon};
///         FIRST(A) ← FIRST(A) ∪ rhs;
///     end;
/// end;
/// ```
///
/// The FIRST set of a terminal is always just the terminal itself, so we don't need to create sets for those.
fn generate_first_set<NT, T>(productions: impl AsRef<[Production<NT, T>]>) -> FirstSets<T>
where
    T: Clone + Copy + Ord + Eq + std::fmt::Debug + Terminal,
    NT: NonTerminal,
{
    // I was having trouble getting this to work w/o following the algorithm exactly,
    // so what we do here is build up the first sets of all symbols, then only keep the first sets of our
    // non terminals
    //
    // Also, the way we track whether the first set is changing is not effificient and needs improvement
    let mut first_sets: FirstSets<T> = FirstSets::new();

    for nt in productions.as_ref().iter().map(|r| r.non_terminal) {
        first_sets.sets.insert(nt, FirstSet::new());
    }

    let mut changing = true;
    while changing {
        changing = false;
        for Production {
            non_terminal: nt,
            derivation,
            ..
        } in productions.as_ref()
        {
            assert!(
                !derivation.symbols.is_empty(),
                "FATAL: an empty derivation made its way into a grammar"
            );
            let mut symbols = derivation
                .symbols
                .iter()
                .take_while(|&&s| s != Symbol::Epsilon)
                .peekable();

            if symbols.peek().is_none() {
                // we know derivations can't be empty
                if !derivation.symbols.is_empty() {
                    first_sets.sets.entry(*nt).or_default().contains_epsilon = true;
                }
                continue;
            }
            let mut last_symbol = symbols.next().unwrap();
            let mut rhs = first_sets.get_symbol(last_symbol).unwrap().into_owned();
            rhs.contains_epsilon = false;

            while symbols.peek().is_some() {
                if !first_sets
                    .get_symbol(last_symbol)
                    .unwrap()
                    .contains_epsilon()
                {
                    break;
                }
                last_symbol = symbols.next().unwrap();

                rhs.set
                    .append(&mut first_sets.get_symbol(last_symbol).unwrap().into_owned().set);
            }

            let first_set = first_sets.sets.entry(*nt).or_default();
            if !first_set.set.is_superset(&rhs.set) {
                first_set.append(rhs);
                changing = true;
            }
        }
    }

    first_sets
}

/// Helper function for generating the FOLLOW sets of a grammar
///
/// The FOLLOW set of a non-terminal `A` is the set of symbols in the grammar that appear immediately after the non-terminal
///
/// We can generate the FOLLOW sets for symbols in a grammar using the algorithm described in Engineering a Compiler, 2nd Edition, figure 3.8:
/// ```text
/// for each A ∈ NT do;
///     FOLLOW(A) ← ∅;
/// end;
///
/// FOLLOW(S) ← {eof};
///
/// while (FOLLOW sets are still changing) do;
///    for each p ∈ P, where p has the form A -> β1,β2,...βk do;
///        TRAILER ← FOLLOW(A);
///        for i ← k down to 1 do;
///             if βi ∈ NT then
///                 FOLLOW(βi) ← FOLLOW(βi) ∪ TRAILER;
///                 if epsilon ∈ FIRST(βi)
///                     then TRAILER ← TRAILER ∪ (FIRST(βi) - {epsilon});
///                     else TRAILER ← FIRST(βi);
///             end;
///             else TRAILER ← FIRST(βi); // is {βi}
///        end;
///     end;
/// end;
/// ```
fn generate_follow_set<NT, T>(
    productions: impl AsRef<[Production<NT, T>]>,
    start_symbol: usize,
    first_sets: &FirstSets<T>,
) -> FollowSets<T>
where
    T: Clone + Copy + Eq + Ord + std::fmt::Debug + Terminal,
{
    let mut follow_sets = FollowSets::default();

    for nt in productions.as_ref().iter().map(|r| r.non_terminal) {
        follow_sets.sets.entry(nt).or_default();
    }

    follow_sets.sets.entry(start_symbol).and_modify(|fs| {
        fs.set.insert(T::eof());
    });

    let mut changing = true;
    while changing {
        changing = false;
        for Production {
            non_terminal: nt,
            derivation,
            ..
        } in productions.as_ref()
        {
            let mut trailer = follow_sets.get(nt).unwrap().clone();

            for symbol in derivation.symbols.iter().rev() {
                match symbol {
                    Symbol::NonTerminal(nt) => {
                        let follow_set = follow_sets.sets.entry(*nt).or_default();
                        if !follow_set.set.is_superset(&trailer.set) {
                            follow_set.set.extend(&trailer.set);
                            changing = true;
                        }

                        if first_sets.get(nt).unwrap().contains_epsilon {
                            trailer.set.extend(&first_sets.get(nt).unwrap().set);
                        } else {
                            trailer.set.clone_from(&first_sets.get(nt).unwrap().set);
                        }
                    }
                    Symbol::Terminal(t) => {
                        trailer.set.clear();
                        trailer.set.insert(*t);
                    }
                    Symbol::Epsilon => {}
                    Symbol::Eof => {
                        trailer.set.clear();
                        trailer.set.insert(T::eof());
                    }
                }
            }
        }
    }

    follow_sets
}

fn generate_first_plus_set<NT, T>(
    productions: impl AsRef<[Production<NT, T>]>,
    first_sets: &FirstSets<T>,
    follow_sets: &FollowSets<T>,
) -> FirstPlusSets<T>
where
    T: Clone + Copy + Eq + Ord + Terminal,
{
    let mut first_plus_sets = FirstPlusSets::default();

    for Production {
        non_terminal: nt,
        derivation,
        ..
    } in productions.as_ref()
    {
        assert!(
            !derivation.symbols.is_empty(),
            "FATAL: an empty derivation made its way into a grammar"
        );
        let first = first_sets.get_first_set(&derivation.symbols).unwrap();

        let first_plus = if first.contains_epsilon() {
            (first, follow_sets.get(nt).unwrap().clone()).into()
        } else {
            first.into()
        };
        first_plus_sets
            .sets
            .entry(*nt)
            .or_default()
            .insert(derivation.clone(), first_plus);
    }

    first_plus_sets
}

impl<NT,T> Grammar<NT,T,TerminatingReady<T>> 
where 
    T: Clone + Copy + Eq + Ord + Terminal,
    NT: NonTerminal + Copy
{
    /// Get the mapping of generated non-terminals to their parent non-terminals
    #[must_use]
    pub fn generated_non_terminals(&self) -> &[(usize, usize)] {
        &self.state.generated_non_terminals
    }

    /// Check is the grammar is LL(1), and if it is, convert self to a `LL1` grammar
    /// 
    /// A grammar is LL(1) if for every pair of productions `A -> α` and `A -> β`, the following conditions hold:
    /// 1. `FIRST+(α) ∩ FIRST+(β) = ∅`
    /// 2. If `ε ∈ FIRST+(α)`, then `FIRST+(α) ∩ FOLLOW(A) = ∅`
    /// 3. If `ε ∈ FIRST+(β)`, then `FIRST+(β) ∩ FOLLOW(A) = ∅`
    /// 
    /// # Errors
    /// 
    /// Returns:
    /// 
    /// - Ok(Grammar) if the grammar is LL(1)
    /// - Err(Grammar) if the grammar is not LL(1)
    pub fn check_ll1(self) -> Result<Grammar<NT,T,LL1<T>>, Self> {
        if !self.state.first_plus_sets.sets.iter().all(|(_, derivations)| {
            derivations.iter().all(|(d1, first_plus1)| {
                derivations.iter().all(|(d2, first_plus2)| {
                    d1 == d2 || first_plus1.set.is_disjoint(&first_plus2.set) 
                })
            })
        }) {
            return Err(self);
        }

        Ok(Grammar {
            start_symbol: self.start_symbol,
            productions: self.productions,
            state: LL1 {
                generated_non_terminals: self.state.generated_non_terminals,
                first_plus_sets: self.state.first_plus_sets,
            }
        })
    }

    /// Generate a parse table for the grammar
    ///
    /// The parse table is a 2D array where the rows are the non-terminals in the grammar, and the columns are the terminals in the grammar.
    /// Each cell in the table contains a production from the grammar, or an error if there is no production for that cell.
    ///
    /// The textbook doesn't actually have an algorithm for generating non-ll1 parse tables, but the idea isn't to complex.
    /// 
    /// # Panics
    /// 
    /// Might panic if a table wasn't generated properly
    pub fn generate_parse_table(&self) -> ParseTable<NT,T> {
        let mut parse_table = ParseTable::new(self.start_symbol);


        for production in &self.productions {
            let first_plus = self.state.first_plus_sets.get(&production.non_terminal, &production.derivation).unwrap();
            for symbol in &first_plus.set {
                // add the production to the parse table
                parse_table.add_production(production.non_terminal, *symbol, production);
            }
        }

        parse_table
    }
}

impl<NT,T> Grammar<NT,T,LL1<T>>
where T: Clone + Copy + Eq + Ord + Terminal + std::fmt::Debug,
    NT: NonTerminal + Copy
    {

    /// Get the mapping of generated non-terminals to their parent non-terminals
    #[must_use]
    pub fn generated_non_terminals(&self) -> &[(usize, usize)] {
        &self.state.generated_non_terminals
    }

    /// Generate a parse table for an LL(1) grammar
    /// 
    /// # Panics
    /// 
    /// Might panic if a table wasn't generated properly
    pub fn generate_parse_table(&self) -> LL1ParseTable<NT,T> {
        let mut parse_table = LL1ParseTable::new(self.start_symbol);

        for production in &self.productions {
            let first_plus = self.state.first_plus_sets.get(&production.non_terminal, &production.derivation).unwrap();
            for symbol in &first_plus.set {
                // add the production to the parse table
                parse_table.add_production(production.non_terminal, *symbol, production);
            }
        }

        parse_table
    }
}


#[cfg(test)]
mod eliminate_left_recursion_tests {
    //! Tests for ensuring we can correctly transform arbitrary grammars to eliminate both direct and indirect left recursion
    use crate::generic::test_utils::*;
    use super::*;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(usize)]
    enum AbcNT {
        A,
        B,
        C,
    }
    impl From<AbcNT> for usize {
        fn from(nt: AbcNT) -> Self {
            nt as usize
        }
    }

    impl NonTerminal for AbcNT {}
    impl Terminal for char {
        fn eof() -> Self {
            '\0'
        }
    }

    #[rstest]
    #[case(
        vec![
            Production::new(0usize, derivation![0, 'a']),
            Production::new(0usize,  derivation!['b']),
        ],
        vec![
            Production::new(0usize, derivation!['b', 1]),
            Production::new(1usize, derivation!['a', 1]),
            Production::new(1usize, derivation![Symbol::Epsilon]),
        ]
    )]
    #[case(
        vec![
            Production::new(0usize, derivation![1,'a']),
            Production::new(1usize, derivation!['d','a','b']),
            Production::new(1usize, derivation![2,'b']),
            Production::new(2usize, derivation!['c',1]),
            Production::new(2usize, derivation![2,'b','a','c']),
            Production::new(2usize, derivation!['d','a','b','a','c']),
        ],
        vec![
            Production::new(0usize, derivation![1, 'a']),
            Production::new(1usize, derivation!['d', 'a', 'b']), 
            Production::new(1usize, derivation![2, 'b']),
            Production::new(
                2usize,
                derivation!['c', 1, 3]),
                Production::new(
                    2usize,
                derivation!['d', 'a', 'b', 'a', 'c', 3]
            ),
            Production::new(
                3usize,
                derivation!['b', 'a', 'c', 3]),
                Production::new(
                    3usize,
                derivation![Symbol::Epsilon]
            ),
        ]

    )]
    fn direct_left_recursion(
        #[case] rules: Vec<Production<AbcNT, char>>,
        #[case] expected: Vec<Production<AbcNT, char>>,
    ) {
        let grammar = Grammar::new(AbcNT::A, rules)
            .unwrap()
            .eliminate_left_recursion();

        assert_eq!(grammar.productions, expected);
    }

    #[rstest]
    #[case::abc_grammar(
        Grammar::new(
            AbcNT::A,
            vec![
                Production::new(AbcNT::A, derivation![AbcNT::B, 'a']),
                Production::new(
                    AbcNT::B,
                    derivation!['d', 'a', 'b']),
                    Production::new(
                        AbcNT::B,
                    derivation![AbcNT::C, 'b']
                ),
                Production::new(
                    AbcNT::C,
                    derivation!['c', AbcNT::B]
                ),
                Production::new(
                    AbcNT::C,
                    derivation![AbcNT::A, 'c']
                ),
            ],
        )
        .unwrap(),
        vec![
            // A
            Production::new(AbcNT::A, derivation![AbcNT::B, 'a']),
            // B
            Production::new(
                AbcNT::B,
                derivation!['d', 'a', 'b']),Production::new(
                    AbcNT::B,
                derivation![AbcNT::C, 'b']
            ),
            // C
            Production::new(
                AbcNT::C,
                derivation!['c', AbcNT::B, 3]),Production::new(
                    AbcNT::C,
                derivation!['d', 'a', 'b', 'a', 'c', 3]
            ),
            // C'
            Production::new(
                3usize,
                derivation!['b', 'a', 'c', 3]),Production::new(
                    3usize,
                derivation![Symbol::Epsilon]
            ),
        ],
    )]
    #[case::expr_grammar(
        expr_grammar(),
        vec![
            // Goal
            Production::new(ExprNT::Goal, derivation![ExprNT::Expr]),
            // Expr
            Production::new(ExprNT::Expr, derivation![ExprNT::Term, ExprNT::ExprPrime]),
            // Term
            Production::new(ExprNT::Term, derivation![ExprNT::Factor, ExprNT::TermPrime]),
            // Factor
            Production::new(
                ExprNT::Factor,
                derivation![ExprT::LeftParen, ExprNT::Expr, ExprT::RightParen]),Production::new(
                    ExprNT::Factor,
                derivation![ExprT::Num]),Production::new(
                    ExprNT::Factor,
                derivation![ExprT::Name]
            ),
            // Expr'
            Production::new(
                ExprNT::ExprPrime,
                derivation![ExprT::Plus, ExprNT::Term, ExprNT::ExprPrime]),Production::new(
                    ExprNT::ExprPrime,
                derivation![ExprT::Minus, ExprNT::Term, ExprNT::ExprPrime]),Production::new(
                    ExprNT::ExprPrime,
                derivation![Symbol::Epsilon]
            ),
            // Term'
            Production::new(
                ExprNT::TermPrime,
                derivation![ExprT::Mult, ExprNT::Factor, ExprNT::TermPrime]), Production::new(
                    ExprNT::TermPrime,
                derivation![ExprT::Div, ExprNT::Factor, ExprNT::TermPrime]), Production::new(
                    ExprNT::TermPrime,
                derivation![Symbol::Epsilon]
            )
        ]
    )]
    fn indirect_left_recursion<NT: NonTerminal + Copy, T: Terminal + Copy + Eq + std::fmt::Debug>(
        #[case] grammar: Grammar<NT, T, NonTerminating>,
        #[case] expected: Vec<Production<NT, T>>,
    ) {
        let grammar = grammar.eliminate_left_recursion();

        assert_eq!(grammar.productions, expected);
    }
}

#[cfg(test)]
mod first_follow_firstplus_set_tests {
    //! Tests to ensure we can correctly generate the first, follow, and first+ sets of arbitrary grammars
    use crate::generic::test_utils::*;
    use super::*;
    use pretty_assertions::assert_eq;
    use rstest::{fixture, rstest};

    #[fixture]
    fn grammar(
        expr_grammar: Grammar<ExprNT, ExprT, NonTerminating>,
    ) -> Grammar<ExprNT, ExprT, TerminatingReady<ExprT>> {
        expr_grammar.eliminate_left_recursion().generate_sets()
    }

    #[rstest]
    fn test_first_set(grammar: Grammar<ExprNT,ExprT, TerminatingReady<ExprT>>) {
        let expected = FirstSets::from_iter([
            (
                ExprNT::Goal as usize,
                FirstSet::from_iter([ExprT::LeftParen, ExprT::Name, ExprT::Num]),
            ),
            (
                ExprNT::Expr as usize,
                FirstSet::from_iter([ExprT::LeftParen, ExprT::Name, ExprT::Num]),
            ),
            (
                ExprNT::ExprPrime as usize,
                FirstSet::from_iter([ExprT::Plus, ExprT::Minus])
                    .with_epsilon()
                    .into_owned(),
            ),
            (
                ExprNT::Term as usize,
                FirstSet::from_iter([ExprT::LeftParen, ExprT::Name, ExprT::Num]),
            ),
            (
                ExprNT::TermPrime as usize,
                FirstSet::from_iter([ExprT::Mult, ExprT::Div])
                    .with_epsilon()
                    .into_owned(),
            ),
            (
                ExprNT::Factor as usize,
                FirstSet::from_iter([ExprT::LeftParen, ExprT::Name, ExprT::Num]),
            ),
        ]);

        assert_eq!(grammar.state.first_sets, expected);
    }

    #[rstest]
    fn test_follow_set(grammar: Grammar<ExprNT,ExprT, TerminatingReady<ExprT>>) {
        let expected = FollowSets::from_iter([
            (ExprNT::Goal as usize, FollowSet::from_iter([ExprT::Eof])),
            (
                ExprNT::Expr as usize,
                FollowSet::from_iter([ExprT::Eof, ExprT::RightParen]),
            ),
            (
                ExprNT::ExprPrime as usize,
                FollowSet::from_iter([ExprT::Eof, ExprT::RightParen]),
            ),
            (
                ExprNT::Term as usize,
                FollowSet::from_iter([ExprT::Eof, ExprT::Plus, ExprT::Minus, ExprT::RightParen]),
            ),
            (
                ExprNT::TermPrime as usize,
                FollowSet::from_iter([ExprT::Eof, ExprT::Plus, ExprT::Minus, ExprT::RightParen]),
            ),
            (
                ExprNT::Factor as usize,
                FollowSet::from_iter([
                    ExprT::Eof,
                    ExprT::Plus,
                    ExprT::Minus,
                    ExprT::Mult,
                    ExprT::Div,
                    ExprT::RightParen,
                ]),
            ),
        ]);

        assert_eq!(grammar.state.follow_sets, expected);
    }

    #[rstest]
    fn test_first_plus_set(grammar: Grammar<ExprNT,ExprT, TerminatingReady<ExprT>>) {
        let expected = FirstPlusSets::from_iter([
            (
                ExprNT::Goal as usize,
                BTreeMap::from_iter([(
                    derivation![ExprNT::Expr],
                    FirstPlusSet::from(FirstSet::from_iter([
                        ExprT::LeftParen,
                        ExprT::Name,
                        ExprT::Num,
                    ])),
                )]),
            ),
            (
                ExprNT::Expr as usize,
                BTreeMap::from_iter([(
                    derivation![ExprNT::Term, ExprNT::ExprPrime],
                    FirstPlusSet::from(FirstSet::from_iter([
                        ExprT::LeftParen,
                        ExprT::Name,
                        ExprT::Num,
                    ])),
                )]),
            ),
            (
                ExprNT::ExprPrime as usize,
                BTreeMap::from_iter([
                    (
                        derivation![ExprT::Plus, ExprNT::Term, ExprNT::ExprPrime],
                        FirstPlusSet::from_iter([ExprT::Plus]),
                    ),
                    (
                        derivation![ExprT::Minus, ExprNT::Term, ExprNT::ExprPrime],
                        FirstPlusSet::from_iter([ExprT::Minus]),
                    ),
                    (
                        derivation![Symbol::Epsilon],
                        FirstPlusSet::from_iter([ExprT::Eof, ExprT::RightParen])
                            .with_epsilon()
                            .into_owned(),
                    ),
                ]),
            ),
            (
                ExprNT::Term as usize,
                BTreeMap::from_iter([(
                    derivation![ExprNT::Factor, ExprNT::TermPrime],
                    FirstPlusSet::from_iter([ExprT::LeftParen, ExprT::Name, ExprT::Num]),
                )]),
            ),
            (
                ExprNT::TermPrime as usize,
                BTreeMap::from_iter([
                    (
                        derivation![ExprT::Mult, ExprNT::Factor, ExprNT::TermPrime],
                        FirstPlusSet::from_iter([ExprT::Mult]),
                    ),
                    (
                        derivation![ExprT::Div, ExprNT::Factor, ExprNT::TermPrime],
                        FirstPlusSet::from_iter([ExprT::Div]),
                    ),
                    (
                        derivation![Symbol::Epsilon],
                        FirstPlusSet::from_iter([
                            ExprT::Eof,
                            ExprT::Plus,
                            ExprT::Minus,
                            ExprT::RightParen,
                        ])
                        .with_epsilon()
                        .into_owned(),
                    ),
                ]),
            ),
            (
                ExprNT::Factor as usize,
                BTreeMap::from_iter([
                    (
                        derivation![ExprT::LeftParen, ExprNT::Expr, ExprT::RightParen],
                        FirstPlusSet::from_iter([ExprT::LeftParen]),
                    ),
                    (
                        derivation![ExprT::Num],
                        FirstPlusSet::from_iter([ExprT::Num]),
                    ),
                    (
                        derivation![ExprT::Name],
                        FirstPlusSet::from_iter([ExprT::Name]),
                    ),
                ]),
            ),
        ]);

        assert_eq!(grammar.state.first_plus_sets, expected);
    }
}

#[cfg(test)]
mod ll1_tests {
    //! Tests to ensure we can correctly determine if a grammar is LL(1)
    use crate::generic::test_utils::*;
    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn grammar(
        expr_grammar: Grammar<ExprNT, ExprT, NonTerminating>,
    ) -> Grammar<ExprNT, ExprT, TerminatingReady<ExprT>> {
        expr_grammar.eliminate_left_recursion().generate_sets()
    }

    #[rstest]
    fn ll1_grammar(grammar: Grammar<ExprNT, ExprT, TerminatingReady<ExprT>>) {
        let grammar = grammar.check_ll1();

        assert!(grammar.is_ok());
    }

    #[derive(Debug, Copy, Clone)]
    #[repr(usize)]
    enum FactorNT {
        Factor,
        ArgList,
        MoreArgs,
        Arg,
    }
    impl From<FactorNT> for usize {
        fn from(nt: FactorNT) -> Self {
            nt as usize
        }
    }

    impl NonTerminal for FactorNT {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum FactorT {
        Name,
        LeftBracket,
        RightBracket,
        LeftParen,
        RightParen,
        Comma,
        EOF,
    }

    impl Terminal for FactorT {
        fn eof() -> Self {
            FactorT::EOF
        }
    }

    #[rstest]
    fn non_ll1_grammar() {
        let grammar = Grammar::new(
            FactorNT::Factor,
            vec![
                Production::new(FactorNT::Factor, derivation![FactorT::Name]),
                Production::new(FactorNT::Factor, derivation![FactorT::Name, FactorT::LeftBracket, FactorNT::ArgList, FactorT::RightBracket]),
                Production::new(FactorNT::Factor, derivation![FactorT::Name, FactorT::LeftParen, FactorNT::ArgList, FactorT::RightParen]),
                Production::new(FactorNT::ArgList, derivation![FactorNT::Arg, FactorNT::MoreArgs]),
                Production::new(FactorNT::MoreArgs, derivation![FactorT::Comma, FactorNT::Arg, FactorNT::MoreArgs]),
                Production::new(FactorNT::MoreArgs, derivation![Symbol::Epsilon]),
                Production::new(FactorNT::Arg, derivation![FactorNT::Factor]),
            ],
        )
        .unwrap()
        .eliminate_left_recursion()
        .generate_sets()
        .check_ll1();

        assert!(grammar.is_err());
    }

    #[rstest]
    fn parse_table_for_expr_grammar(grammar: Grammar<ExprNT, ExprT, TerminatingReady<ExprT>>) {
        let grammar = grammar.check_ll1().unwrap();
        let table = grammar.generate_parse_table();

        let mut expected = BTreeMap::new();

        let production_0 = Production::new(ExprNT::Goal, derivation![ExprNT::Expr]);
        let production_1 = Production::new(ExprNT::Expr, derivation![ExprNT::Term, ExprNT::ExprPrime]);
        let production_2 = Production::new(ExprNT::ExprPrime, derivation![ExprT::Plus, ExprNT::Term, ExprNT::ExprPrime]);
        let production_3 = Production::new(ExprNT::ExprPrime, derivation![ExprT::Minus, ExprNT::Term, ExprNT::ExprPrime]);
        let production_4 = Production::new(ExprNT::ExprPrime, derivation![Symbol::Epsilon]);
        let production_5 = Production::new(ExprNT::Term, derivation![ExprNT::Factor, ExprNT::TermPrime]);
        let production_6 = Production::new(ExprNT::TermPrime, derivation![ExprT::Mult, ExprNT::Factor, ExprNT::TermPrime]);
        let production_7 = Production::new(ExprNT::TermPrime, derivation![ExprT::Div, ExprNT::Factor, ExprNT::TermPrime]);
        let production_8 = Production::new(ExprNT::TermPrime, derivation![Symbol::Epsilon]);
        let production_9 = Production::new(ExprNT::Factor, derivation![ExprT::LeftParen, ExprNT::Expr, ExprT::RightParen]);
        let production_10 = Production::new(ExprNT::Factor, derivation![ExprT::Num]);
        let production_11 = Production::new(ExprNT::Factor, derivation![ExprT::Name]);

        expected.insert(
            ExprNT::Goal as usize,
            BTreeMap::from_iter([
                (ExprT::LeftParen, &production_0),
                (ExprT::Name, &production_0),
                (ExprT::Num, &production_0),
            ]),
        );

        expected.insert(
            ExprNT::Expr as usize,
            BTreeMap::from_iter([
                (ExprT::LeftParen, &production_1),
                (ExprT::Name, &production_1),
                (ExprT::Num, &production_1),
            ]),
        );

        expected.insert(
            ExprNT::ExprPrime as usize,
            BTreeMap::from_iter([
                (ExprT::Plus, &production_2),
                (ExprT::Minus, &production_3),
                (ExprT::Eof, &production_4),
                (ExprT::RightParen, &production_4),
            ]),
        );

        expected.insert(
            ExprNT::Term as usize,
            BTreeMap::from_iter([
                (ExprT::LeftParen, &production_5),
                (ExprT::Name, &production_5),
                (ExprT::Num, &production_5),
            ]),
        );

        expected.insert(
            ExprNT::TermPrime as usize,
            BTreeMap::from_iter([
                (ExprT::Mult, &production_6),
                (ExprT::Div, &production_7),
                (ExprT::Eof, &production_8),
                (ExprT::Plus, &production_8),
                (ExprT::Minus, &production_8),
                (ExprT::RightParen, &production_8),
            ]),
        );

        expected.insert(
            ExprNT::Factor as usize,
            BTreeMap::from_iter([
                (ExprT::LeftParen, &production_9),
                (ExprT::Num, &production_10),
                (ExprT::Name, &production_11),
            ]),
        );

        assert_eq!(table.table, expected);
    }

}