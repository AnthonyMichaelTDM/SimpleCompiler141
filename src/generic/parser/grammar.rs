use super::{
    Derivation, LL1ParseTable, Merge, NonTerminal, ParseTable, Production, Symbol, Terminal,
};

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, VecDeque},
};

/// A grammar that defines a set of rules for a language
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar<NT, T, State = NonTerminating> where NT: NonTerminal, T: Terminal {
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
#[derive(Debug)]
pub struct NonTerminating;

/// Marker struct for a grammar that has been converted to eliminate left-recursion
#[derive(Debug)]
pub struct Terminating;

/// Marker struct for a Terminating grammar, that is not LL(1) but is ready to use for parsing.
/// Created by generating the FIRST, FOLLOW, and FIRST+ sets from a `Grammar<_,Terminating>`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminatingReady<NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    /// The first set's for each non-terminal.
    /// This is a map from non-terminals to the set of terminals that can appear as the first symbol
    /// of a sentence derived from that non-terminal.
    first_sets: FirstSets<NT, T>,
    /// The follow set's for each non-terminal.
    /// The follow set of a non-terminal is the set of symbols in the grammar that appear immediately after the non-terminal
    follow_sets: FollowSets<NT, T>,
    /// The first+ set's for each production.
    /// The first+ set of a production `A -> α` is:
    /// - FIRST(α) if FIRST(α) does not contain epsilon
    /// - FIRST(α) ∪ FOLLOW(A) if FIRST(α) contains epsilon
    first_plus_sets: FirstPlusSets<NT, T>,
}

/// Marker struct for a grammar that has been converted to be LL(1) by left-factoring
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LL1<NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    /// The first+ set's for each production.
    /// The first+ set of a production `A -> α` is:
    /// - FIRST(α) if FIRST(α) does not contain epsilon
    /// - FIRST(α) ∪ FOLLOW(A) if FIRST(α) contains epsilon
    first_plus_sets: FirstPlusSets<NT, T>,
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum GrammarError<NT: NonTerminal, T: Terminal> {
    #[error(
        "Non-Terminal {0:?} does not have any rules expanding it, it appeared in the following production: {1}"
    )]
    NonTerminalWithoutRules(NT, Production<NT, T>),
    #[error(
        "The following production contains so symbols, perhaps you meant to use epsilon? {0}"
    )]
    DerivationWithoutSymbols(Production<NT, T>),
    #[error("The following production expands a non-terminal to only itself: {0}")]
    NonTerminalOnlyExpandsToItself(Production<NT, T>),
    #[error("The grammar contains direct or indirect left-recusion on one or more non-terminals, here is the in-degree list of the relevant non-terminals: {0:?}")]
    LeftRecusion(Vec<(NT, usize)>),
    #[error("Start symbol {0:?} does not have any expansions")]
    GoalWithoutExpansions(NT),
    #[error("Grammar has no rules")]
    NoRules,
    #[error("The grammar is not LL(1), the first+ set of {0} ({1:?}) is not disjoint from that of {2} ({3:?})")]
    NotLL1(Production<NT, T>, FirstPlusSet<T>, Production<NT, T>,FirstPlusSet<T>),
}

pub type GrammarResult<OK, NT,T> = Result<OK, GrammarError<NT,T>>;

/// The FIRST set of a non-terminal `A` is the set of terminals that can appear as the first symbol of a sentence derived from `A`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstSet<T>
where
    T: Terminal,
{
    set: BTreeSet<T>,
    contains_epsilon: bool,
}

impl<T: Terminal> Default for FirstSet<T> {
    fn default() -> Self {
        Self {
            set: BTreeSet::new(),
            contains_epsilon: false,
        }
    }
}

/// The FIRST sets of all non-terminals in the grammar
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstSets<NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    sets: BTreeMap<NT, FirstSet<T>>,
}

impl<NT, T> Default for FirstSets<NT, T>
where
    NT: NonTerminal,
    T: Terminal,
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
pub struct FollowSets<NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    sets: BTreeMap<NT, FollowSet<T>>,
}

impl<NT, T> Default for FollowSets<NT, T>
where
    NT: NonTerminal,
    T: Terminal,
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
    T: Terminal,
{
    set: BTreeSet<T>,
    contains_epsilon: bool,
}

impl<T: Terminal> Default for FirstPlusSet<T> {
    fn default() -> Self {
        Self {
            set: BTreeSet::new(),
            contains_epsilon: false,
        }
    }
}

/// The FIRST+ sets of all productions in the grammar
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstPlusSets<NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    sets: BTreeMap<NT, BTreeMap<Derivation<NT, T>, FirstPlusSet<T>>>,
}

impl<NT, T> Default for FirstPlusSets<NT, T>
where
    NT:NonTerminal,
    T: Terminal,
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
    T: Terminal,
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
impl<T: Terminal> FromIterator<T> for FirstSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            set: BTreeSet::from_iter(iter),
            contains_epsilon: false,
        }
    }
}

impl<NT, T> FirstSets<NT, T>
where
    NT: NonTerminal,
    T:  Terminal,
{
    /// Create a new empty set of FIRST sets
    #[must_use]
    fn new() -> Self {
        Self::default()
    }

    /// Get the FIRST set for the given non-terminal, if it exists
    #[must_use]
    pub fn get(&self, nt: &NT) -> Option<&FirstSet<T>> {
        self.sets.get(nt)
    }

    /// Get the FIRST set for the given Symbol
    #[must_use]
    pub fn get_symbol(&self, symbol: &Symbol<NT, T>) -> Option<Cow<FirstSet<T>>> {
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
    pub fn iter(&self) -> impl Iterator<Item = (&NT, &FirstSet<T>)> {
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
    pub fn get_first_set(&self, symbols: &[Symbol<NT,T>]) -> Option<FirstSet<T>> {
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
impl<NT, T> FromIterator<(NT, FirstSet<T>)> for FirstSets<NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    fn from_iter<I: IntoIterator<Item = (NT, FirstSet<T>)>>(iter: I) -> Self {
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

impl<NT, T> FollowSets<NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    /// Get the FOLLOW set for the given non-terminal, if it exists
    #[must_use]
    pub fn get(&self, nt: &NT) -> Option<&FollowSet<T>> {
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
    pub fn iter(&self) -> impl Iterator<Item = (&NT, &FollowSet<T>)> {
        self.sets.iter()
    }
}

#[cfg(test)]
impl<NT, T> FromIterator<(NT, FollowSet<T>)> for FollowSets<NT, T>
where
    NT: NonTerminal,
    T: Terminal,
{
    fn from_iter<I: IntoIterator<Item = (NT, FollowSet<T>)>>(iter: I) -> Self {
        Self {
            sets: BTreeMap::from_iter(iter),
        }
    }
}

impl<T> FirstPlusSet<T>
where
    T: Terminal,
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
    T: Terminal,
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
    T: Terminal,
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
    T: Terminal,
{
    fn from(first: FirstSet<T>) -> Self {
        Self {
            set: first.set,
            contains_epsilon: first.contains_epsilon,
        }
    }
}

impl<NT, T> FirstPlusSets<NT, T>
where
NT:NonTerminal,
    T:  Terminal,
{
    /// Get the FIRST+ set for the given production, if there is one
    #[must_use]
    pub fn get(&self, nt: &NT, derivation: &Derivation<NT, T>) -> Option<&FirstPlusSet<T>> {
        self.sets.get(nt).and_then(|m| m.get(derivation))
    }

    /// Check if the FIRST+ set of a production contains a given lookahead symbol
    /// If the FIRST+ set contains epsilon, this will also check if the FOLLOW set of the non-terminal contains the lookahead symbol
    #[must_use]
    pub fn contains(&self, nt: &NT, derivation: &Derivation<NT, T>, lookahead: &T) -> bool {
        self.get(nt, derivation)
            .is_some_and(|set| set.set.contains(lookahead))
    }
}

#[cfg(test)]
impl<NT, T> FromIterator<(NT, BTreeMap<Derivation<NT, T>, FirstPlusSet<T>>)> for FirstPlusSets<NT, T>
where
NT:NonTerminal,
    T: Terminal,
{
    fn from_iter<I: IntoIterator<Item = (NT, BTreeMap<Derivation<NT,T>, FirstPlusSet<T>>)>>(
        iter: I,
    ) -> Self {
        Self {
            sets: BTreeMap::from_iter(iter),
        }
    }
}

impl<NT, T> Grammar<NT, T, NonTerminating>
where
    NT: NonTerminal,
    T: Terminal,
{
    /// Create a new grammar from the given non-terminals, terminals, start symbol, and rules
    ///
    /// # Errors
    ///
    /// This will return an error if the grammar is invalid, such as if a non-terminal does not have any expansions,
    /// or if a non-terminal expands only to itself.
    pub fn new<P>(start_symbol: NT, productions: Vec<P>) -> GrammarResult<Self, NT,T>
    where
        P: Into<Production<NT, T>>,
    {
        
        let productions: Vec<Production<NT, T>> = productions.into_iter().map(Into::into).collect();

            if productions.is_empty() {
                return Err(GrammarError::NoRules);
            }

            // Non-Terminals that have expansions
            let mut non_terminals = BTreeSet::new();

            for production in &productions {
                non_terminals.insert(production.non_terminal);
                if production.derivation.symbols.is_empty() {
                    return Err(GrammarError::DerivationWithoutSymbols(
                        production.to_owned()
                    ));
                }
            }

            if !non_terminals.contains(&start_symbol) {
                return Err(GrammarError::GoalWithoutExpansions(start_symbol));
            }

            // now we go through every rule again, and ensure that every non-terminal listed in a derivartion has
            // an expansion
            for production in &productions {
                // make sure the nt doesn't only expand to itself
                if production.derivation.symbols.len() == 1
                    && matches!(
                        production.derivation.symbols.first(),
                        Some(&Symbol::NonTerminal(nt)) if nt == production.non_terminal,
                    )
                {
                    return Err(GrammarError::NonTerminalOnlyExpandsToItself(
                        production.to_owned()
                    ));
                }
                // make sure every non-terminal in the derivation has an expansion
                if let Some(nt) = production
                    .derivation
                    .symbols
                    .iter()
                    .find_map(|symbol| match symbol {
                        Symbol::NonTerminal(nt) if !non_terminals.contains(nt) => Some(*nt),
                        _ => None,
                    })
                {
                    return Err(GrammarError::NonTerminalWithoutRules(
                        nt,
                        production.to_owned()
                    ));
                }
            }
            

            // If we've made it this far, the grammar is valid
            Ok(Self {
                start_symbol,
                productions,
                state: NonTerminating,
            })
        
    }

    /// Ensure that a grammar does not contain any left-recursion
    ///
    /// Implementing this is similar to the algorithm described in Engineering a Compiler, 2nd Edition, figure 3.6
    /// for eliminating left-recursion, however, since eleminating left-recursion is a destructive operation that can make parse trees of the
    /// existing grammar hard to map back to the original grammar, we just return an error instead of rewriting anything.
    ///
    /// For reference, the algorithm is as follows:
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
    /// On second thought, checking for recursion is like seeing if there are cycles in a graph of the NT's in the grammar,
    /// where an edge from A to B means that B is the first symbol in a derivation of A.
    ///
    /// If there are cycles, then there is left-recursion.
    ///
    /// To make it so that we can report what the cycles are, we will store the production index of the production corresponding to each edge in the graph.
    /// 
    /// # Errors
    /// 
    /// Returns an error if a cycle is found in the first-link graph of non-terminals
    pub fn check_terminating(self) -> GrammarResult<Grammar<NT, T, Terminating>, NT, T> {
        debug_assert!(!self.productions.is_empty());

        // construct the 'first-link' graph
        let mut first_link = BTreeMap::new();

        for Production {
            non_terminal: nt,
            derivation,
            ..
        } in &self.productions
        {
            if let Some(Symbol::NonTerminal(first)) = derivation.symbols.first() {
                first_link
                .entry(*nt)
                .or_insert_with(BTreeSet::new)
                .insert(*first);
            } else {
                first_link.entry(*nt).or_insert(BTreeSet::new());
            }
        }

        // check for cycles in the graph
        let cycle = check_for_cycles(&first_link);

        if let Some(cycle) = cycle {
            return Err(GrammarError::LeftRecusion(cycle));
        }

        // if we've made it this far, the grammar is terminating
        Ok(Grammar {
            start_symbol: self.start_symbol,
            productions: self.productions,
            state: Terminating,
        })
    }
}

/// Checks for cycles in a graph of non-terminals.
///
/// Uses Khans algorithm to check for cycles,
/// this allows us to both check if there are cycles, and report what the cycles are.
///
/// Returns the in-degrees of the non-terminals involved in the cycle(s), if there are any.
fn check_for_cycles<NT:NonTerminal>(
    first_link_graph: &BTreeMap<NT, BTreeSet<NT>>,
) -> Option<Vec<(NT, usize)>> {
    let vertex_count = first_link_graph.len();
    
    let mut in_degree = BTreeMap::new();
    
    for node in first_link_graph.keys() {
        in_degree.entry(*node).or_insert(0);
        for adj in &first_link_graph[node] {
            in_degree.entry(*adj).and_modify(|degree| *degree += 1).or_insert(1);
        }
    }
    
    let mut queue = VecDeque::new();

    // enqueue vertices with in-degree 0
    for node in first_link_graph.keys() {
        if matches!(in_degree.get(node), Some(0) | None) {
            queue.push_back(*node);
        }
    }

    // count of visited vertices
    let mut count = 0;

    // topological order
    let mut top_order = Vec::new();

    while let Some(node) = queue.pop_front() {
        top_order.push(node);
        count += 1;

        if !first_link_graph.contains_key(&node) {
            continue;
        }

        for &adj in &first_link_graph[&node] {
            in_degree
                .entry(adj)
                .and_modify(|degree| *degree -= 1);

            if matches!(in_degree.get(&adj), Some(0) | None) {
                queue.push_back(adj);
            }
        }
    }

    let cycle_in_degree: Vec<(NT, usize)> = in_degree
        .iter()
        .filter(|(_, &degree)| degree != 0)
        .map(|(&node, _)| (node, in_degree[&node]))
        .collect();
    // is there a cycle
    if cycle_in_degree.is_empty() {
        None
    } else {
        assert!(cycle_in_degree
            .iter()
            .all(|(node, _)| !top_order.contains(node)));
        assert!(count != vertex_count);
        Some(cycle_in_degree)
    }
}

impl<NT, T> Grammar<NT, T, Terminating>
where
    NT: NonTerminal,
    T: Terminal,
{
    /// Generate the FIRST, FOLLOW, and FIRST+ sets for the grammar, and convert it to a `TerminatingReady` grammar
    ///
    /// This can be used by a parser to generate a parse table for a grammar that is not LL(1), and parse input using that parse table with backtracking.
    #[must_use]
    pub fn generate_sets(self) -> Grammar<NT, T, TerminatingReady<NT,T>> {
        let first_sets = generate_first_set(&self.productions);
        let follow_sets =
            generate_follow_set(&self.productions, self.start_symbol, &first_sets);
        let first_plus_sets = generate_first_plus_set(&self.productions, &first_sets, &follow_sets);

        Grammar {
            start_symbol: self.start_symbol,
            productions: self.productions,
            state: TerminatingReady {
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
    pub fn left_factor<O>(self) -> GrammarResult<Grammar<NT, O, LL1<NT, T>>,NT,T>
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
fn generate_first_set<NT, T>(productions: impl AsRef<[Production<NT, T>]>) -> FirstSets<NT, T>
where
NT: NonTerminal,
T: Terminal,
{
    // I was having trouble getting this to work w/o following the algorithm exactly,
    // so what we do here is build up the first sets of all symbols, then only keep the first sets of our
    // non terminals
    //
    // Also, the way we track whether the first set is changing is not effificient and needs improvement
    let mut first_sets: FirstSets<NT, T> = FirstSets::new();

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
    start_symbol: NT,
    first_sets: &FirstSets<NT, T>,
) -> FollowSets<NT, T>
where
NT: NonTerminal,
    T: Terminal,
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
    first_sets: &FirstSets<NT, T>,
    follow_sets: &FollowSets<NT, T>,
) -> FirstPlusSets<NT, T>
where NT:NonTerminal,
    T: Terminal,
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

impl<NT, T> Grammar<NT, T, TerminatingReady<NT, T>>
where
    NT: NonTerminal,
    T:  Terminal,
{
    /// Check is the grammar is LL(1), and if it is, convert self to a `LL1` grammar
    ///
    /// A grammar is LL(1) if for every pair of productions `A -> α` and `A -> β`, the following conditions hold:
    /// 1. `FIRST(α) ∩ FIRST(β) = ∅`
    /// 2. If `ε ∈ α`, then `FIRST(β) ∩ FOLLOW(A) = ∅`
    /// 
    /// These can be summarized as:
    /// - The FIRST+ sets of all productions for a given non-terminal are disjoint
    ///
    /// # Errors
    ///
    /// Returns:
    ///
    /// - Ok(Grammar) if the grammar is LL(1)
    /// - Err(Grammar) if the grammar is not LL(1)
    /// 
    /// # Panics
    /// 
    /// Panics if there is a production that does not have a FIRST+ set, which should never happen if 
    /// the grammar was generated with `generate_sets()`
    pub fn check_ll1(self) -> GrammarResult<Grammar<NT, T, LL1<NT, T>>,NT,T> {
        // group the productions by their non-terminal
        let mut productions_by_nt: BTreeMap<NT, Vec<Production<NT, T>>> = BTreeMap::new();
        
        // while doing so, check ll1 condition
        for production in &self.productions {
            let nt = production.non_terminal;

            // check that the first+ set of this production is disjoint
            // from the first+ sets of all other productions of the same non-terminal
            let first_plus = self
                .state
                .first_plus_sets
                .get(&nt, &production.derivation)
                .unwrap_or_else(|| panic!("FATAL: no first+ set for the production {production:?}"));
 
            for other_production in productions_by_nt.entry(nt).or_insert_with(Vec::default).iter() {
                let other_first_plus = self
                    .state
                    .first_plus_sets
                    .get(&nt, &other_production.derivation)
                    .unwrap_or_else(|| panic!("FATAL: no first+ set for the other production {other_production:?}"));

                if !first_plus.set.is_disjoint(&other_first_plus.set) {
                    return Err(GrammarError::NotLL1(production.to_owned(),first_plus.to_owned(), other_production.to_owned(), other_first_plus.to_owned()));
                }
            }

            // add the production to the list of productions for this non-terminal so we can check it against the others later
            productions_by_nt
                .entry(nt)
                .or_insert_with(Vec::new)
                .push(production.clone());
        }

        Ok(Grammar {
            start_symbol: self.start_symbol,
            productions: self.productions,
            state: LL1 {
                first_plus_sets: self.state.first_plus_sets,
            },
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
    pub fn generate_parse_table(&self) -> ParseTable<NT, T> {
        let mut parse_table = ParseTable::new(self.start_symbol);

        for production in &self.productions {
            let first_plus = self
                .state
                .first_plus_sets
                .get(&production.non_terminal, &production.derivation)
                .unwrap();
            for symbol in &first_plus.set {
                // add the production to the parse table
                parse_table.add_production(production.non_terminal, *symbol, production);
            }
        }

        parse_table
    }
}

impl<NT, T> Grammar<NT, T, LL1<NT, T>>
where
    T: Clone + Copy + Eq + Ord + Terminal + std::fmt::Debug,
    NT: NonTerminal + Copy,
{
    /// Generate a parse table for an LL(1) grammar
    ///
    /// # Panics
    ///
    /// Might panic if a table wasn't generated properly
    pub fn generate_parse_table(&self) -> LL1ParseTable<NT, T> {
        let mut parse_table = LL1ParseTable::new(self.start_symbol);

        for production in &self.productions {
            let first_plus = self
                .state
                .first_plus_sets
                .get(&production.non_terminal, &production.derivation)
                .unwrap();
            for symbol in &first_plus.set {
                // add the production to the parse table
                parse_table.add_production(production.non_terminal, *symbol, production);
            }
        }

        parse_table
    }
}

#[cfg(test)]
mod left_recursion_tests {
    //! Tests for ensuring we can correctly detect both direct and indirect left recursion
    use super::*;
    use crate::derivation;
    use crate::generic::test_utils::*;
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
    #[case::loop_with_self(
        BTreeMap::from_iter(vec![
            (0, BTreeSet::from_iter(vec![0])),
        ]),
        Some(vec![(0, 1)])
    )]
    #[case::simple_loop(
        BTreeMap::from_iter(vec![
            (0, BTreeSet::from_iter(vec![1])),
            (1, BTreeSet::from_iter(vec![0])),
        ]),
        Some(vec![(0, 1), (1, 1)])
    )]
    #[case::simple_line(
        BTreeMap::from_iter(vec![
            (0, BTreeSet::from_iter(vec![1])),
            (1, BTreeSet::from_iter(vec![2])),
            (2, BTreeSet::from_iter(vec![3])),
        ]),
        None
    )]
    #[case::simple_tree(
        BTreeMap::from_iter(vec![
            (0, BTreeSet::from_iter(vec![1,2])),
            (1, BTreeSet::from_iter(vec![3])),
            (2, BTreeSet::from_iter(vec![4])),
        ]),
        None
    )]
    #[case::simple_loop(
        BTreeMap::from_iter(vec![
            (0, BTreeSet::from_iter(vec![1])),
            (1, BTreeSet::from_iter(vec![2])),
            (2, BTreeSet::from_iter(vec![0])),
        ]),
        Some(vec![(0, 1), (1, 1), (2, 1)])
    )]
    #[case::long_loop(
        BTreeMap::from_iter(vec![
            (0, BTreeSet::from_iter(vec![1])),
            (1, BTreeSet::from_iter(vec![2])),
            (2, BTreeSet::from_iter(vec![3])),
            (3, BTreeSet::from_iter(vec![4])),
            (4, BTreeSet::from_iter(vec![5])),
            (5, BTreeSet::from_iter(vec![6])),
            (6, BTreeSet::from_iter(vec![7])),
            (7, BTreeSet::from_iter(vec![8])),
            (8, BTreeSet::from_iter(vec![9])),
            (9, BTreeSet::from_iter(vec![0])),
        ]),
       Some( 
            vec![0,1,2,3,4,5,6,7,8,9].iter().map(|&n| (n,1)).collect()
       )
    )]
    #[case::complex_graph_with_cycle(
        BTreeMap::from_iter(vec![
            (0, BTreeSet::from_iter(vec![1, 2])),
            (1, BTreeSet::from_iter(vec![3])),
            (2, BTreeSet::from_iter(vec![4, 5])),
            (3, BTreeSet::from_iter(vec![5, 6, 7])),
            (4, BTreeSet::from_iter(vec![5])),
            (5, BTreeSet::from_iter(vec![6,7])),
            (6, BTreeSet::from_iter(vec![7])),
            (7, BTreeSet::from_iter(vec![2])),
        ]),
       Some( vec![(2, 1), (4, 1), (5, 2), (6, 1), (7, 2)])
    )]
    #[case::many_weird_cycles(
        BTreeMap::from_iter(vec![
            (0, BTreeSet::from_iter(vec![1, 2])),
            (1, BTreeSet::from_iter(vec![3])),
            (2, BTreeSet::from_iter(vec![4, 5, 0])),
            (3, BTreeSet::from_iter(vec![5, 6, 7])),
            (4, BTreeSet::from_iter(vec![5])),
            (5, BTreeSet::from_iter(vec![6, 7])),
            (6, BTreeSet::from_iter(vec![7, 5])),
            (7, BTreeSet::from_iter(vec![8])),
            (8, BTreeSet::from_iter(vec![2, 4])),
        ]),
        Some(vec![
            (0,1), (1,1),
            (2,2), (3,1),
            (4,2), (5,4),
            (6,2), (7,3),
            (8,1)
        ])
    )]
    #[case::complex_graph_without_cycle(
        BTreeMap::from_iter(vec![
            (0, BTreeSet::from_iter(vec![1, 2])),
            (1, BTreeSet::from_iter(vec![3])),
            (2, BTreeSet::from_iter(vec![4, 5])),
            (3, BTreeSet::from_iter(vec![5, 6, 7])),
            (4, BTreeSet::from_iter(vec![5])),
            (5, BTreeSet::from_iter(vec![6,7])),
            (6, BTreeSet::from_iter(vec![7])),
            (7, BTreeSet::from_iter(vec![8])),
        ]),
        None,
    )]
    fn test_check_cycles(
        #[case] graph: BTreeMap<usize, BTreeSet<usize>>,
        #[case] expected: Option<Vec<(usize, usize)>>,
    ) {
        let result = check_for_cycles(&graph);

        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        vec![
            Production::new(AbcNT::A, derivation![AbcNT::A, 'a']),
            Production::new(AbcNT::A, derivation!['b']),
        ],
        GrammarError::LeftRecusion(
            vec![(AbcNT::A,1)]
        )
    )]
    #[case(
        vec![
            Production::new(AbcNT::A, derivation![AbcNT::B,'a']),
            Production::new(AbcNT::B, derivation!['d','a','b']),
            Production::new(AbcNT::B, derivation![AbcNT::C,'b']),
            Production::new(AbcNT::C, derivation!['c',AbcNT::A]),
            Production::new(AbcNT::C, derivation![AbcNT::C,'b','a','c']),
            Production::new(AbcNT::C, derivation!['d','a','b','a','c']),
        ],
        GrammarError::LeftRecusion(vec![
            (AbcNT::C,1)
        ])
    )]
    fn direct_left_recursion(
        #[case] rules: Vec<Production<AbcNT, char>>,
        #[case] expected: GrammarError<AbcNT,char>,
    ) {
        let grammar = Grammar::new(AbcNT::A, rules).unwrap();

        let result = grammar.check_terminating().unwrap_err();

        assert_eq!(result, expected);
    }

    #[rstest]
    #[case::abc_grammar(
        Grammar::new(
            AbcNT::A,
            vec![
                Production::new(AbcNT::A, derivation![AbcNT::B, 'a']),
                Production::new(
                    AbcNT::B,
                    derivation!['d', 'a', 'b']
                ),
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
        Some(vec![
            (AbcNT::A,1),
            (AbcNT::B,1),
            (AbcNT::C,1),
        ])
    )]
    #[case::abc_double_left_recursion(
        Grammar::new(
            AbcNT::A,
            vec![
                // A -> B A c
                Production::new(AbcNT::A, derivation![AbcNT::B, AbcNT::A, 'c']),
                // B -> A a c
                Production::new(AbcNT::B, derivation![AbcNT::A, 'a', 'c']),
                // B -> ε
                Production::new(AbcNT::B, derivation![Symbol::Epsilon]),
            ]
        ).unwrap(),
        Some(vec![(AbcNT::A, 1), (AbcNT::B,1)])
    )]
    #[case::abc_no_recursion(
        Grammar::new(
            AbcNT::A,
            vec![
                Production::new(AbcNT::A, derivation![AbcNT::B, 'a']),
                Production::new(
                    AbcNT::B,
                    derivation!['d', 'a', 'b']
                ),
                Production::new(
                    AbcNT::B,
                    derivation![AbcNT::C, 'b']
                ),
                Production::new(
                    AbcNT::C,
                    derivation!['c', 'b']
                ),
                Production::new(
                    AbcNT::C,
                    derivation!['d', 'a', 'b', 'a', 'c']
                ),
            ]
        ).unwrap(),
        None
    )]
    #[case::expr_grammar(
        expr_grammar_non_terminating(),
        Some(vec![(ExprNT::Expr,1), (ExprNT::Term,2), (ExprNT::Factor,1)]),
    )]
    #[case::expr_grammar(expr_grammar_terminating(), None)]
    fn indirect_left_recursion<
        NT: NonTerminal + Copy + std::fmt::Debug,
        T: Terminal + Copy + Eq + std::fmt::Debug,
    >(
        #[case] grammar: Grammar<NT, T, NonTerminating>,
        #[case] expected: Option<Vec<(NT, usize)>>,
    ) {
        if let Some(expected) = expected {
            let result = grammar.check_terminating().unwrap_err();
            assert_eq!(result, GrammarError::LeftRecusion(expected));
        } else {
            grammar.check_terminating().unwrap();
        }
    }
}

#[cfg(test)]
mod first_follow_firstplus_set_tests {
    //! Tests to ensure we can correctly generate the first, follow, and first+ sets of arbitrary grammars
    use super::*;
    use crate::derivation;
    use crate::generic::test_utils::*;
    use pretty_assertions::assert_eq;
    use rstest::{fixture, rstest};

    #[fixture]
    fn grammar(
        expr_grammar_terminating: Grammar<ExprNT, ExprT, NonTerminating>,
    ) -> Grammar<ExprNT, ExprT, TerminatingReady<ExprNT,ExprT>> {
        expr_grammar_terminating
            .check_terminating()
            .unwrap()
            .generate_sets()
    }

    #[rstest]
    fn test_first_set(grammar: Grammar<ExprNT, ExprT, TerminatingReady<ExprNT, ExprT>>) {
        let expected = FirstSets::from_iter([
            (
                ExprNT::Goal,
                FirstSet::from_iter([ExprT::LeftParen, ExprT::Name, ExprT::Num]),
            ),
            (
                ExprNT::Expr,
                FirstSet::from_iter([ExprT::LeftParen, ExprT::Name, ExprT::Num]),
            ),
            (
                ExprNT::ExprPrime,
                FirstSet::from_iter([ExprT::Plus, ExprT::Minus])
                    .with_epsilon()
                    .into_owned(),
            ),
            (
                ExprNT::Term,
                FirstSet::from_iter([ExprT::LeftParen, ExprT::Name, ExprT::Num]),
            ),
            (
                ExprNT::TermPrime,
                FirstSet::from_iter([ExprT::Mult, ExprT::Div])
                    .with_epsilon()
                    .into_owned(),
            ),
            (
                ExprNT::Factor,
                FirstSet::from_iter([ExprT::LeftParen, ExprT::Name, ExprT::Num]),
            ),
        ]);

        assert_eq!(grammar.state.first_sets, expected);
    }

    #[rstest]
    fn test_follow_set(grammar: Grammar<ExprNT, ExprT, TerminatingReady<ExprNT, ExprT>>) {
        let expected = FollowSets::from_iter([
            (ExprNT::Goal, FollowSet::from_iter([ExprT::Eof])),
            (
                ExprNT::Expr,
                FollowSet::from_iter([ExprT::Eof, ExprT::RightParen]),
            ),
            (
                ExprNT::ExprPrime,
                FollowSet::from_iter([ExprT::Eof, ExprT::RightParen]),
            ),
            (
                ExprNT::Term,
                FollowSet::from_iter([ExprT::Eof, ExprT::Plus, ExprT::Minus, ExprT::RightParen]),
            ),
            (
                ExprNT::TermPrime,
                FollowSet::from_iter([ExprT::Eof, ExprT::Plus, ExprT::Minus, ExprT::RightParen]),
            ),
            (
                ExprNT::Factor,
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
    fn test_first_plus_set(grammar: Grammar<ExprNT, ExprT, TerminatingReady<ExprNT,ExprT>>) {
        let expected = FirstPlusSets::from_iter([
            (
                ExprNT::Goal ,
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
                ExprNT::Expr ,
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
                ExprNT::ExprPrime ,
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
                ExprNT::Term ,
                BTreeMap::from_iter([(
                    derivation![ExprNT::Factor, ExprNT::TermPrime],
                    FirstPlusSet::from_iter([ExprT::LeftParen, ExprT::Name, ExprT::Num]),
                )]),
            ),
            (
                ExprNT::TermPrime ,
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
                ExprNT::Factor ,
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
    use super::*;
    use crate::derivation;
    use crate::generic::test_utils::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn grammar(
        expr_grammar_terminating: Grammar<ExprNT, ExprT, NonTerminating>,
    ) -> Grammar<ExprNT, ExprT, TerminatingReady<ExprNT,ExprT>> {
        expr_grammar_terminating
            .check_terminating()
            .unwrap()
            .generate_sets()
    }

    #[rstest]
    fn ll1_grammar(grammar: Grammar<ExprNT, ExprT, TerminatingReady<ExprNT,ExprT>>) {
        let grammar = grammar.check_ll1();

        assert!(grammar.is_ok());
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
                Production::new(
                    FactorNT::Factor,
                    derivation![
                        FactorT::Name,
                        FactorT::LeftBracket,
                        FactorNT::ArgList,
                        FactorT::RightBracket
                    ],
                ),
                Production::new(
                    FactorNT::Factor,
                    derivation![
                        FactorT::Name,
                        FactorT::LeftParen,
                        FactorNT::ArgList,
                        FactorT::RightParen
                    ],
                ),
                Production::new(
                    FactorNT::ArgList,
                    derivation![FactorNT::Arg, FactorNT::MoreArgs],
                ),
                Production::new(
                    FactorNT::MoreArgs,
                    derivation![FactorT::Comma, FactorNT::Arg, FactorNT::MoreArgs],
                ),
                Production::new(FactorNT::MoreArgs, derivation![Symbol::Epsilon]),
                Production::new(FactorNT::Arg, derivation![FactorNT::Factor]),
            ],
        )
        .unwrap()
        .check_terminating()
        .unwrap()
        .generate_sets()
        .check_ll1();

        assert!(grammar.is_err());
    }

    #[rstest]
    fn parse_table_for_expr_grammar(grammar: Grammar<ExprNT, ExprT, TerminatingReady<ExprNT,ExprT>>) {
        let grammar = grammar.check_ll1().unwrap();
        let table = grammar.generate_parse_table();

        let mut expected = BTreeMap::new();

        let production_0 = Production::new(ExprNT::Goal, derivation![ExprNT::Expr]);
        let production_1 =
            Production::new(ExprNT::Expr, derivation![ExprNT::Term, ExprNT::ExprPrime]);
        let production_2 = Production::new(
            ExprNT::ExprPrime,
            derivation![ExprT::Plus, ExprNT::Term, ExprNT::ExprPrime],
        );
        let production_3 = Production::new(
            ExprNT::ExprPrime,
            derivation![ExprT::Minus, ExprNT::Term, ExprNT::ExprPrime],
        );
        let production_4 = Production::new(ExprNT::ExprPrime, derivation![Symbol::Epsilon]);
        let production_5 =
            Production::new(ExprNT::Term, derivation![ExprNT::Factor, ExprNT::TermPrime]);
        let production_6 = Production::new(
            ExprNT::TermPrime,
            derivation![ExprT::Mult, ExprNT::Factor, ExprNT::TermPrime],
        );
        let production_7 = Production::new(
            ExprNT::TermPrime,
            derivation![ExprT::Div, ExprNT::Factor, ExprNT::TermPrime],
        );
        let production_8 = Production::new(ExprNT::TermPrime, derivation![Symbol::Epsilon]);
        let production_9 = Production::new(
            ExprNT::Factor,
            derivation![ExprT::LeftParen, ExprNT::Expr, ExprT::RightParen],
        );
        let production_10 = Production::new(ExprNT::Factor, derivation![ExprT::Num]);
        let production_11 = Production::new(ExprNT::Factor, derivation![ExprT::Name]);

        expected.insert(
            ExprNT::Goal ,
            BTreeMap::from_iter([
                (ExprT::LeftParen, &production_0),
                (ExprT::Name, &production_0),
                (ExprT::Num, &production_0),
            ]),
        );

        expected.insert(
            ExprNT::Expr ,
            BTreeMap::from_iter([
                (ExprT::LeftParen, &production_1),
                (ExprT::Name, &production_1),
                (ExprT::Num, &production_1),
            ]),
        );

        expected.insert(
            ExprNT::ExprPrime ,
            BTreeMap::from_iter([
                (ExprT::Plus, &production_2),
                (ExprT::Minus, &production_3),
                (ExprT::Eof, &production_4),
                (ExprT::RightParen, &production_4),
            ]),
        );

        expected.insert(
            ExprNT::Term ,
            BTreeMap::from_iter([
                (ExprT::LeftParen, &production_5),
                (ExprT::Name, &production_5),
                (ExprT::Num, &production_5),
            ]),
        );

        expected.insert(
            ExprNT::TermPrime ,
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
            ExprNT::Factor ,
            BTreeMap::from_iter([
                (ExprT::LeftParen, &production_9),
                (ExprT::Num, &production_10),
                (ExprT::Name, &production_11),
            ]),
        );

        assert_eq!(table.table, expected);
    }
}
