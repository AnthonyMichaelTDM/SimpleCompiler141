use crate::generic::TokenSpan;

use super::{NonTerminal, Terminal};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseTree<'a, NT, T, Token> {
    Node(ParseTreeNode<'a, NT, T, Token>),
    Leaf(ParseTreeLeaf<'a, T, Token>),
    Epsilon,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseTreeNode<'a, NT, T, Token> {
    pub non_terminal: NT,
    pub children: Vec<ParseTree<'a, NT, T, Token>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseTreeLeaf<'a, T, Token> {
    pub terminal: T,
    pub token_span: TokenSpan<'a, Token>,
}

pub enum EulerTraversalVisit<T> {
    Pre(T),
    Post { parent: T, child: T },
}

impl<'a, NT, T, Token> ParseTree<'a, NT, T, Token>
where
    NT: NonTerminal,
    T: Terminal,
    Token: Copy,
{
    /// Create a new parse tree node
    #[must_use]
    pub(super) const fn node(non_terminal: NT, children: Vec<Self>) -> Self {
        Self::Node(ParseTreeNode {
            non_terminal,
            children,
        })
    }

    /// Create a new parse tree leaf
    #[must_use]
    pub(super) const fn leaf(terminal: T, token_span: TokenSpan<'a, Token>) -> Self {
        Self::Leaf(ParseTreeLeaf {
            terminal,
            token_span,
        })
    }

    /// Return `Some(terminal)` if the parse tree is a leaf
    /// Otherwise return None
    #[must_use]
    pub const fn terminal(&self) -> Option<T> {
        match self {
            Self::Leaf(ParseTreeLeaf { terminal, .. }) => Some(*terminal),
            Self::Node(_) | Self::Epsilon => None,
        }
    }

    /// Return `Some(token_span)` if the parse tree is a leaf
    /// Otherwise return None
    #[must_use]
    pub const fn token_span(&self) -> Option<TokenSpan<'a, Token>> {
        match self {
            Self::Leaf(ParseTreeLeaf { token_span, .. }) => Some(*token_span),
            Self::Node(_) | Self::Epsilon => None,
        }
    }

    /// Return `Some(non_terminal)` if the parse tree is a node
    /// Otherwise return None
    #[must_use]
    pub const fn non_terminal(&self) -> Option<NT> {
        match self {
            Self::Node(ParseTreeNode { non_terminal, .. }) => Some(*non_terminal),
            Self::Leaf(_) | Self::Epsilon => None,
        }
    }
    /// Return `Some(children)` if the parse tree is a node
    /// Otherwise return None
    #[allow(clippy::missing_const_for_fn)] // TODO: remove when const as_slice is stable
    #[must_use]
    pub fn children(&self) -> Option<&[Self]> {
        match self {
            Self::Node(ParseTreeNode { children, .. }) => Some(children),
            Self::Leaf(_) | Self::Epsilon => None,
        }
    }

    /// Perform a post-order traversal of the parse tree
    /// and apply the given function to each node
    ///
    /// # Note
    ///
    /// A post-order traversal will visit the children of a node before visiting the node itself
    pub fn visit_post_order<F>(&self, f: &mut F)
    where
        F: FnMut(&Self),
    {
        match self {
            Self::Node(ParseTreeNode { children, .. }) => {
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
    ///
    /// # Note
    ///
    /// A pre-order traversal will visit the node itself before visiting its children
    pub fn visit_pre_order<F>(&self, f: &mut F)
    where
        F: FnMut(&Self),
    {
        f(self);
        match self {
            Self::Node(ParseTreeNode { children, .. }) => {
                for child in children {
                    child.visit_pre_order(f);
                }
            }
            Self::Leaf { .. } | Self::Epsilon => {}
        }
    }

    /// Perform a euler-tour traversal of the parse tree
    /// and apply the given function to each node
    ///
    /// # Note
    ///
    /// A euler-tour traversal will visit the node itself, then its children,
    /// then the node itself again before moving on to the next sibling
    pub fn visit_euler_tour<F>(&self, f: &mut F)
    where
        F: FnMut(EulerTraversalVisit<&Self>),
    {
        f(EulerTraversalVisit::Pre(self));
        match self {
            Self::Node(ParseTreeNode { children, .. }) => {
                for child in children {
                    child.visit_euler_tour(f);
                    f(EulerTraversalVisit::Post {
                        parent: self,
                        child,
                    });
                }
            }
            Self::Leaf { .. } | Self::Epsilon => {}
        }
    }

    // TODO: need a way to rewrite the parse tree to remove the generated non-terminals
    // while preserving the order of terminals and syntax
}
