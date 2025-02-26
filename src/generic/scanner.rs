//! This module contains the generic scanner implementation
use core::ops::Range;
use std::marker::PhantomData;

/// A token, which specifies it type and the range it occupies in the input
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenSpan<'a, T> {
    /// The type of the token
    pub token_type: T,

    /// The start range of the token in the input (essentially where it is in the input text)
    pub start: usize,
    /// The end (exclusive) range of the token in the input (essentially where it is in the input text)
    pub end: usize,

    /// The text of the token
    pub text: &'a str,
}

/// A scanner that tokenizes input text
#[derive(Debug, Clone, Copy)]

pub struct Scanner<'a, T> {
    /// The input text to scan
    input: &'a str,
    /// The language (`TokenType`) of the scanner
    language: PhantomData<T>,
    /// The rules for tokenizing the input
    rules: &'a [&'a dyn TokenRule<TokenType = T>],
}

/// An iterator over the tokens in the input text
#[derive(Debug)]
pub struct TokenIter<'a, T> {
    /// A reference to the scanner that created this iterator
    scanner: &'a Scanner<'a, T>,
    /// The left pointer of the sliding window
    left: usize,
    /// The right pointer of the sliding window
    right: usize,
}

/// A trait that defines a `matches` function for each token type
pub trait TokenRule: Sync + Send + std::fmt::Debug {
    type TokenType;

    /// Check if the given text matches this rule on the given range
    fn matches(&self, input: &str, range: Range<usize>) -> bool;

    /// Type of the token this rule matches to
    fn token_type(&self) -> Self::TokenType;
}

/////////////////////////////////////////////////
// Implementations
/////////////////////////////////////////////////

impl<'a, T> Scanner<'a, T> {
    /// Create a new scanner from the input text
    #[must_use]
    pub const fn new(input: &'a str, rules: &'a [&'a dyn TokenRule<TokenType = T>]) -> Self {
        Scanner {
            input,
            language: PhantomData,
            rules,
        }
    }

    /// Attempt to match a token at the given range, only succeed if the range is maximal (i.e., increasing it by one would not match)
    ///
    /// Returned token span has the same lifetime as the input text, as scanning is a zero-copy operation.
    #[must_use]
    pub fn match_token(&self, range: Range<usize>) -> Option<TokenSpan<'a, T>> {
        // Get the (`TokenType` of the) first rule that the range matches, if any
        let token_type = self.rules.iter().find_map(|rule| {
            rule.matches(self.input, range.clone())
                .then(|| rule.token_type())
        })?;

        Some(TokenSpan {
            token_type,
            text: &self.input[range.clone()],
            start: range.start,
            end: range.end,
        })
    }

    /// Get an iterator over the tokens in the input text
    #[must_use]
    pub const fn iter(&'a self) -> TokenIter<'a, T> {
        TokenIter {
            scanner: self,
            left: 0,
            right: 1,
        }
    }
}

impl<'a, T> IntoIterator for &'a Scanner<'a, T> {
    type Item = TokenSpan<'a, T>;
    type IntoIter = TokenIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> Iterator for TokenIter<'a, T> {
    type Item = TokenSpan<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next;
        while self.right <= self.scanner.input.len() {
            next = self.right + 1;
            // does the slice match a token?
            match self.scanner.match_token(self.left..self.right) {
                // if it does, and including the next character (if any)
                // would **not** result in a match, we've found a token
                Some(token)
                    if self.right >= self.scanner.input.len()
                        || self.scanner.match_token(self.left..next).is_none() =>
                {
                    self.left = self.right;
                    self.right = next;
                    return Some(token);
                }
                // otherwise, keep moving the right pointer as we either:
                // - haven't found a token yet
                // - could get a longer token by including the next character
                _ => self.right = next,
            }
        }

        // if we're at the end of the input, we're done
        None
    }

    /// Give a hint to the iterator about the upper and lower bounds on the number of remaining elements
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.scanner.input.len() - self.left))
    }
}
