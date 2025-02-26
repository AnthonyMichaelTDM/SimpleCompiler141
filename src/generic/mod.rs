//! This module contains generic library code for constructing
//! scanners, parsers, and eventually other compiler components.
//!
//! `crate::langauge` contains the language-specific implementations
//! of these components.

mod scanner;
pub use scanner::*;
