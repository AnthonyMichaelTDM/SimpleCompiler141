//! This module contains the generic scanner implementation
use core::ops::Range;
use std::marker::PhantomData;

/// A token, which specifies it type and the range it occupies in the input
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenSpan<'a, T> {
    /// The type of the token
    pub kind: T,

    /// The start range of the token in the input (essentially where it is in the input text)
    pub start: usize,
    /// The end (exclusive) range of the token in the input (essentially where it is in the input text)
    pub end: usize,

    /// The text of the token
    pub text: &'a str,

    /// Whether the token covers the last character of the input
    /// (useful for matching to the end of the input in your parser)
    pub is_eof: bool,

    /// Whether the token is a whitespace token
    pub is_whitespace: bool,
}

/// An owned version of `TokenSpan`, used for errors
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OwnedTokenSpan<T> {
    /// The type of the token
    pub kind: T,

    /// The start range of the token in the input (essentially where it is in the input text)
    pub start: usize,
    /// The end (exclusive) range of the token in the input (essentially where it is in the input text)
    pub end: usize,

    /// The text of the token
    pub text: String,

    /// Whether the token covers the last character of the input
    /// (useful for matching to the end of the input in your parser)
    pub is_eof: bool,

    /// Whether the token is a whitespace token
    pub is_whitespace: bool,
}

/// An error type for the scanner
#[derive(thiserror::Error, Debug, PartialEq, Eq, Clone, Copy)]
pub enum ScannerError {
    #[error("Illegal character '{0}' found at position {1}")]
    IllegalCharacter(char, usize),
}

pub type ScannerResult<T> = Result<T, ScannerError>;

/// A scanner that tokenizes input text
#[derive(Clone, Copy)]
pub struct Scanner<'a, T> {
    /// The input text to scan
    input: &'a str,
    /// The language (`TokenType`) of the scanner
    language: PhantomData<T>,
    /// The rules for tokenizing the input
    rules: &'a [&'a dyn TokenRule<TokenType = T>],
}

impl<T> core::fmt::Debug for Scanner<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Scanner")
            .field("input", &self.input)
            .field("number of rules", &self.rules.len())
            .finish()
    }
}

/// An iterator over the tokens in the input text
#[derive(Debug)]
pub struct TokenIter<'a, T> {
    /// A reference to the scanner that created this iterator
    scanner: Scanner<'a, T>,
    /// The left pointer of the sliding window
    left: usize,
    /// The right pointer of the sliding window
    right: usize,
    /// whether the iterator should stop on the next iteration (e.g., we've encountered an error in scanning)
    stop: bool,
}

/// A trait that defines a `matches` function for each token type
pub trait TokenRule: Sync + Send {
    type TokenType;

    /// Check if the given text matches this rule on the given range
    fn matches(&self, input: &str, range: Range<usize>) -> bool;

    /// Type of the token this rule matches to
    fn token_type(&self) -> Self::TokenType;

    /// Whether this token rule is a whitespace rule (i.e., should be skipped)
    /// By default, this is false
    fn is_whitespace(&self) -> bool {
        false
    }
}

/////////////////////////////////////////////////
// Implementations
/////////////////////////////////////////////////

impl<T> TokenSpan<'_, T> {
    /// Create an owned version of this token span
    pub fn into_owned(self) -> OwnedTokenSpan<T> {
        OwnedTokenSpan {
            kind: self.kind,
            start: self.start,
            end: self.end,
            text: self.text.to_string(),
            is_eof: self.is_eof,
            is_whitespace: self.is_whitespace,
        }
    }
}

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
        let (token_type, is_whitespace) = self.rules.iter().find_map(|rule| {
            rule.matches(self.input, range.clone())
                .then(|| (rule.token_type(), rule.is_whitespace()))
        })?;

        Some(TokenSpan {
            kind: token_type,
            text: &self.input[range.clone()],
            start: range.start,
            end: range.end,
            is_eof: range.end == self.input.len(),
            is_whitespace,
        })
    }

    /// Get an iterator over the tokens in the input text
    #[must_use]
    pub const fn iter(self) -> TokenIter<'a, T> {
        TokenIter {
            scanner: self,
            left: 0,
            right: 0,
            stop: false,
        }
    }
}

impl<'a, T> IntoIterator for Scanner<'a, T> {
    type Item = Result<TokenSpan<'a, T>, ScannerError>;
    type IntoIter = TokenIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> Iterator for TokenIter<'a, T> {
    type Item = Result<TokenSpan<'a, T>, ScannerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stop {
            return None;
        }

        let mut next;
        while self.right <= self.scanner.input.len() {
            next = self.right + 1;

            // if right would split a character boundary (e.g., a UTF-8 character),
            // continue until it doesn't
            while !self.scanner.input.is_char_boundary(next) && next <= self.scanner.input.len() {
                next += 1;
            }

            // does the slice match a token?
            match self.scanner.match_token(self.left..self.right) {
                // if it does, and including the next character (if any)
                // would **not** result in a match, we've found a token
                Some(token)
                    if self.right >= self.scanner.input.len()
                        || self.scanner.match_token(self.left..next).is_none() =>
                {
                    #[cfg(debug_assertions)]
                    if token.is_eof {
                        assert_eq!(self.right, self.scanner.input.len());
                        assert_eq!(token.end, self.scanner.input.len());
                    }
                    self.left = self.right;
                    self.right = next;
                    return Some(Ok(token));
                }
                // otherwise, keep moving the right pointer as we either:
                // - haven't found a token yet
                // - could get a longer token by including the next character
                _ => self.right = next,
            }
        }

        if self.left < self.scanner.input.len() {
            // if we haven't reached the end of the input, but there's no more tokens to match
            // this means there's an invalid character in the input
            let invalid_char = self.scanner.input.chars().nth(self.left).unwrap();
            self.stop = true;
            Some(Err(ScannerError::IllegalCharacter(invalid_char, self.left)))
        } else {
            // if we've reached the end of the input, return None
            None
        }
    }

    /// Give a hint to the iterator about the upper and lower bounds on the number of remaining elements
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.scanner.input.len() - self.left))
    }
}
