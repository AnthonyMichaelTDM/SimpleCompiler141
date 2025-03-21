//! Definitions necessary for the C parser.

use crate::derivation;
use crate::generic::{
    grammar::{Grammar, NonTerminating},
    NonTerminal, Production, Symbol, Terminal,
};
use crate::generic::{TokenConversionError, TokenSpan};

use super::CTokenType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CNonTerminal {
    ProgramStart,
    Program,
    TypeName,
    Program0,
    Id0,
    IdList0,
    Program1,
    Func0,
    FuncPath,
    Expression,
    Id,
    FuncOrData,
    ParameterList,
    Func1,
    Func4,
    FuncList,
    Factor,
    Term0,
    Expression0,
    FuncList0,
    ParameterList0,
    NonEmptyList0,
    Func2,
    Func5,
    Func,
    Factor0,
    MulOp,
    AddOp,
    Term,
    DataDecls,
    Func3,
    Statements,
    Func6,
    Factor1,
    IdList,
    DataDecls0,
    Statement,
    Statements0,
    ExprList,
    ConditionExpression,
    BlockStatements,
    Statement2,
    NonEmptyExprList,
    Statement0,
    Statement1,
    Condition,
    ConditionExpression0,
    BlockStatements0,
    NonEmptyExprList0,
    ComparisonOp,
    ConditionOp,
}

impl NonTerminal for CNonTerminal {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CTerminal {
    // Reserved Keywords
    Int,
    Void,
    Binary,
    Decimal,
    If,
    While,
    Return,
    Break,
    Continue,
    Read,
    Write,
    Print,

    // Symbols
    Semicolon,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Comma,
    Plus,
    Minus,
    Star,
    ForwardSlash,
    Equal,
    DoubleEqual,
    NotEqual,
    LeftAngle,
    RightAngle,
    LessEqual,
    GreaterEqual,
    DoubleAnd,
    DoubleOr,

    // String
    String,

    // other
    Identifer,
    Number,
    EoF,
}

impl Terminal for CTerminal {
    fn eof() -> Self {
        Self::EoF
    }
}

#[must_use]
#[allow(clippy::too_many_lines)]
pub fn c_grammar() -> Grammar<CNonTerminal, CTerminal, NonTerminating> {
    type P = Production<CNonTerminal, CTerminal>;
    use CNonTerminal::{
        AddOp, BlockStatements, BlockStatements0, ComparisonOp, Condition, ConditionExpression,
        ConditionExpression0, ConditionOp, DataDecls, DataDecls0, ExprList, Expression,
        Expression0, Factor, Factor0, Factor1, Func, Func0, Func1, Func2, Func3, Func4, Func5,
        Func6, FuncList, FuncList0, FuncOrData, FuncPath, Id, Id0, IdList, IdList0, MulOp,
        NonEmptyExprList, NonEmptyExprList0, NonEmptyList0, ParameterList, ParameterList0, Program,
        Program0, Program1, ProgramStart, Statement, Statement0, Statement1, Statement2,
        Statements, Statements0, Term, Term0, TypeName,
    };
    use CTerminal::{
        Binary, Break, Comma, Continue, Decimal, DoubleAnd, DoubleEqual, DoubleOr, EoF, Equal,
        ForwardSlash, GreaterEqual, Identifer, If, Int, LeftAngle, LeftBrace, LeftBracket,
        LeftParen, LessEqual, Minus, NotEqual, Number, Plus, Print, Read, Return, RightAngle,
        RightBrace, RightBracket, RightParen, Semicolon, Star, String, Void, While, Write,
    };

    let productions = vec![
        //
        P::new(ProgramStart, derivation![Program, EoF]),
        P::new(ProgramStart, derivation![EoF]),
        //
        P::new(Program, derivation![TypeName, Identifer, Program0]),
        //
        P::new(TypeName, derivation![Int]),
        P::new(TypeName, derivation![Void]),
        P::new(TypeName, derivation![Binary]),
        P::new(TypeName, derivation![Decimal]),
        //
        P::new(Program0, derivation![Id0, IdList0, Semicolon, Program1]),
        P::new(Program0, derivation![LeftParen, Func0, FuncPath]),
        //
        P::new(Id0, derivation![LeftBracket, Expression, RightBracket]),
        P::new(Id0, derivation![Symbol::Epsilon]),
        //
        P::new(IdList0, derivation![Comma, Id, IdList0]),
        P::new(IdList0, derivation![Symbol::Epsilon]),
        //
        P::new(Program1, derivation![TypeName, Identifer, FuncOrData]),
        P::new(Program1, derivation![Symbol::Epsilon]),
        //
        P::new(Func0, derivation![ParameterList, RightParen, Func1]),
        P::new(Func0, derivation![RightParen, Func4]),
        //
        P::new(FuncPath, derivation![FuncList]),
        P::new(FuncPath, derivation![Symbol::Epsilon]),
        //
        P::new(Expression, derivation![Factor, Term0, Expression0]),
        //
        P::new(Id, derivation![Identifer, Id0]),
        //
        P::new(FuncOrData, derivation![Id0, IdList0, Semicolon, Program1]),
        P::new(FuncOrData, derivation![LeftParen, Func0, FuncList0]),
        //
        P::new(ParameterList, derivation![Void, ParameterList0]),
        P::new(ParameterList, derivation![Int, Identifer, NonEmptyList0]),
        P::new(
            ParameterList,
            derivation![Decimal, Identifer, NonEmptyList0],
        ),
        P::new(ParameterList, derivation![Binary, Identifer, NonEmptyList0]),
        //
        P::new(Func1, derivation![Semicolon]),
        P::new(Func1, derivation![LeftBrace, Func2]),
        //
        P::new(Func4, derivation![Semicolon]),
        P::new(Func4, derivation![LeftBrace, Func5]),
        //
        P::new(FuncList, derivation![Func, FuncList0]),
        //
        P::new(Factor, derivation![Identifer, Factor0]),
        P::new(Factor, derivation![Number]),
        P::new(Factor, derivation![Minus, Number]),
        P::new(Factor, derivation![LeftParen, Expression, RightParen]),
        //
        P::new(Term0, derivation![MulOp, Factor, Term0]),
        P::new(Term0, derivation![Symbol::Epsilon]),
        //
        P::new(Expression0, derivation![AddOp, Term, Expression0]),
        P::new(Expression0, derivation![Symbol::Epsilon]),
        //
        P::new(FuncList0, derivation![FuncList]),
        P::new(FuncList0, derivation![Symbol::Epsilon]),
        //
        P::new(ParameterList0, derivation![Identifer, NonEmptyList0]),
        P::new(ParameterList0, derivation![Symbol::Epsilon]),
        //
        P::new(
            NonEmptyList0,
            derivation![Comma, TypeName, Identifer, NonEmptyList0],
        ),
        P::new(NonEmptyList0, derivation![Symbol::Epsilon]),
        //
        P::new(Func2, derivation![DataDecls, Func3]),
        P::new(Func2, derivation![Statements, RightBrace]),
        P::new(Func2, derivation![RightBrace]),
        //
        P::new(Func5, derivation![DataDecls, Func6]),
        P::new(Func5, derivation![Statements, RightBrace]),
        P::new(Func5, derivation![RightBrace]),
        //
        P::new(Func, derivation![TypeName, Identifer, LeftParen, Func0]),
        //
        P::new(Factor0, derivation![LeftBracket, Expression, RightBracket]),
        P::new(Factor0, derivation![LeftParen, Factor1]),
        P::new(Factor0, derivation![Symbol::Epsilon]),
        //
        P::new(MulOp, derivation![Star]),
        P::new(MulOp, derivation![ForwardSlash]),
        //
        P::new(AddOp, derivation![Plus]),
        P::new(AddOp, derivation![Minus]),
        //
        P::new(Term, derivation![Factor, Term0]),
        //
        P::new(
            DataDecls,
            derivation![TypeName, IdList, Semicolon, DataDecls0],
        ),
        //
        P::new(Func3, derivation![Statements, RightBrace]),
        P::new(Func3, derivation![RightBrace]),
        //
        P::new(Statements, derivation![Statement, Statements0]), // * flag for simplification
        //
        P::new(Func6, derivation![Statements, RightBrace]),
        P::new(Func6, derivation![RightBrace]),
        //
        P::new(Factor1, derivation![ExprList, RightParen]),
        P::new(Factor1, derivation![RightParen]),
        //
        P::new(IdList, derivation![Id, IdList0]),
        //
        P::new(DataDecls0, derivation![DataDecls]),
        P::new(DataDecls0, derivation![Symbol::Epsilon]),
        //
        P::new(Statement, derivation![Identifer, Statement0]),
        P::new(
            Statement,
            derivation![
                If,
                LeftParen,
                ConditionExpression,
                RightParen,
                BlockStatements
            ],
        ),
        P::new(
            Statement,
            derivation![
                While,
                LeftParen,
                ConditionExpression,
                RightParen,
                BlockStatements
            ],
        ),
        P::new(Statement, derivation![Return, Statement2]),
        P::new(Statement, derivation![Break, Semicolon]),
        P::new(Statement, derivation![Continue, Semicolon]),
        P::new(
            Statement,
            derivation![Read, LeftParen, Identifer, RightParen, Semicolon],
        ),
        P::new(
            Statement,
            derivation![Write, LeftParen, Expression, RightParen, Semicolon],
        ),
        P::new(
            Statement,
            derivation![Print, LeftParen, String, RightParen, Semicolon],
        ),
        //
        P::new(Statements0, derivation![Statements]),
        P::new(Statements0, derivation![Symbol::Epsilon]),
        //
        P::new(ExprList, derivation![NonEmptyExprList]),
        //
        P::new(Statement0, derivation![Equal, Expression, Semicolon]),
        P::new(
            Statement0,
            derivation![
                LeftBracket,
                Expression,
                RightBracket,
                Equal,
                Expression,
                Semicolon
            ],
        ),
        P::new(Statement0, derivation![LeftParen, Statement1]),
        //
        P::new(
            ConditionExpression,
            derivation![Condition, ConditionExpression0],
        ),
        //
        P::new(BlockStatements, derivation![LeftBrace, BlockStatements0]),
        //
        P::new(Statement2, derivation![Expression, Semicolon]),
        P::new(Statement2, derivation![Semicolon]),
        //
        P::new(NonEmptyExprList, derivation![Expression, NonEmptyExprList0]),
        //
        P::new(Statement1, derivation![ExprList, RightParen, Semicolon]),
        P::new(Statement1, derivation![RightParen, Semicolon]),
        //
        P::new(Condition, derivation![Expression, ComparisonOp, Expression]),
        //
        P::new(ConditionExpression0, derivation![ConditionOp, Condition]),
        P::new(ConditionExpression0, derivation![Symbol::Epsilon]),
        //
        P::new(BlockStatements0, derivation![Statements, RightBrace]),
        P::new(BlockStatements0, derivation![RightBrace]),
        //
        P::new(
            NonEmptyExprList0,
            derivation![Comma, Expression, NonEmptyExprList0],
        ),
        P::new(NonEmptyExprList0, derivation![Symbol::Epsilon]),
        //
        P::new(ComparisonOp, derivation![DoubleEqual]),
        P::new(ComparisonOp, derivation![NotEqual]),
        P::new(ComparisonOp, derivation![LeftAngle]),
        P::new(ComparisonOp, derivation![RightAngle]),
        P::new(ComparisonOp, derivation![LessEqual]),
        P::new(ComparisonOp, derivation![GreaterEqual]),
        //
        P::new(ConditionOp, derivation![DoubleAnd]),
        P::new(ConditionOp, derivation![DoubleOr]),
    ];

    Grammar::new(ProgramStart, productions).unwrap_or_else(|_| unreachable!())
}

impl TryFrom<TokenSpan<'_, CTokenType>> for CTerminal {
    type Error = TokenConversionError<CTokenType>;

    fn try_from(value: TokenSpan<CTokenType>) -> Result<Self, Self::Error> {
        match (value.kind, value.text) {
            (CTokenType::MetaStatement | CTokenType::Space, _) => {
                Err(TokenConversionError::SkipToken(value.into_owned()))
            }
            (CTokenType::ReservedWord, word) => match word {
                "int" => Ok(Self::Int),
                "void" => Ok(Self::Void),
                "binary" => Ok(Self::Binary),
                "decimal" => Ok(Self::Decimal),
                "if" => Ok(Self::If),
                "while" => Ok(Self::While),
                "return" => Ok(Self::Return),
                "break" => Ok(Self::Break),
                "continue" => Ok(Self::Continue),
                "read" => Ok(Self::Read),
                "write" => Ok(Self::Write),
                "print" => Ok(Self::Print),
                _ => Err(TokenConversionError::MalformedToken(value.into_owned())),
            },
            (CTokenType::Identifier, _) => Ok(Self::Identifer),
            (CTokenType::Number, _) => Ok(Self::Number),
            (CTokenType::Symbol, sym) => match sym {
                ";" => Ok(Self::Semicolon),
                "(" => Ok(Self::LeftParen),
                ")" => Ok(Self::RightParen),
                "[" => Ok(Self::LeftBracket),
                "]" => Ok(Self::RightBracket),
                "{" => Ok(Self::LeftBrace),
                "}" => Ok(Self::RightBrace),
                "," => Ok(Self::Comma),
                "+" => Ok(Self::Plus),
                "-" => Ok(Self::Minus),
                "*" => Ok(Self::Star),
                "/" => Ok(Self::ForwardSlash),
                "=" => Ok(Self::Equal),
                "==" => Ok(Self::DoubleEqual),
                "!=" => Ok(Self::NotEqual),
                "<" => Ok(Self::LeftAngle),
                ">" => Ok(Self::RightAngle),
                "<=" => Ok(Self::LessEqual),
                ">=" => Ok(Self::GreaterEqual),
                "&&" => Ok(Self::DoubleAnd),
                "||" => Ok(Self::DoubleOr),
                _ => Err(TokenConversionError::MalformedToken(value.into_owned())),
            },
            (CTokenType::String, _) => Ok(Self::String),
        }
    }
}

#[cfg(test)]
mod tests {
    //! there are more tests implemented as integration tests, this is just to ensure:
    //!
    //! 1. The grammar compiles
    //! 2. The grammar is correctly identified to be LL(1)
    //! 3. The First+ sets are correctly calculated (TODO)

    use super::*;

    #[test]
    fn test_c_grammar() {
        let grammar = c_grammar();
        let res = grammar.check_terminating();
        assert!(res.is_ok(), "{:?}", res);
        let res = res.unwrap().generate_sets();
        let res = res.check_ll1();
        assert!(res.is_ok(), "{:?}", res);
    }
}
