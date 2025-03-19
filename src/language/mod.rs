//! The language module contains the language rules, grammars, and other language-specific
//! implementations for the compiler.

mod parser;
mod scanner;

pub use parser::*;
pub use scanner::*;
