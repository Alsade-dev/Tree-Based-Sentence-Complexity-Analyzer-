"""
CSC111 Project 2: Tree-Based Sentence Complexity Analyzer
Module: parse_tree.py
Description: Contains the ParseTree class and recursive metric functions
for computing syntactic complexity of English sentences.
Authors: Tugra Canbaz, Elif Cakici, Alsade Brianna Daley
"""

from __future__ import annotations
from typing import Optional
import doctest
from nltk import word_tokenize, RegexpParser
import python_ta
import nltk


class ParseTree:
    """A tree representing the syntactic parse structure of an English sentence.

    Each node stores a grammatical label. Internal nodes represent parts of the
    phrase ('S', 'NP', 'VP', 'SBAR'), leaf nodes store word tokens.

    Instance Attributes:
        - label: The grammatical label at this node (like 'NP', 'VP', 'SBAR'),
                 or a word token string if this node is a leaf.

    Representation Invariants:
        - self.label != ''
        - self.is_leaf() == (self._subtrees == [])
    """
    label: str

    # Private Instance Attributes:
    #  - _subtrees:
    #      the subtrees of this tree, which represent the syntactic children of
    #      this node in the parse tree. Empty if and only if this node is a leaf.
    _subtrees: list[ParseTree]

    def __init__(self, label: str) -> None:
        """Initialize a new ParseTree node with the given label and no subtrees.
        """
        self.label = label
        self._subtrees = []

    def get_subtrees(self) -> list[ParseTree]:
        """Return the subtrees (children) of this parse tree node.

        >>> root = ParseTree('NP')
        >>> root.get_subtrees()
        []
        >>> root.add_subtree(ParseTree('the'))
        >>> len(root.get_subtrees())
        1
        >>> root.get_subtrees()[0].label
        'the'
        """
        return self._subtrees

    def find_subtree_by_label(self, label: str) -> Optional[ParseTree]:
        """Return the first direct subtree with the given label, or None if no such subtree exists.

        >>> root = ParseTree('NP')
        >>> root.add_subtree(ParseTree('DT'))
        >>> root.add_subtree(ParseTree('NN'))
        >>> root.find_subtree_by_label('DT').label
        'DT'
        >>> root.find_subtree_by_label('VP') is None
        True
        """
        for subtree in self._subtrees:
            if subtree.label == label:
                return subtree
        return None

    def add_subtree(self, subtree: ParseTree) -> None:
        """Add a subtree to this parse tree node.

        >>> root = ParseTree('VP')
        >>> root.add_subtree(ParseTree('runs'))
        >>> len(root.get_subtrees())
        1
        >>> root.get_subtrees()[0].label
        'runs'
        """
        self._subtrees.append(subtree)

    def is_leaf(self) -> bool:
        """Return whether this node is a leaf (has no subtrees).
        A leaf node represents a word token in the sentence.

        >>> leaf = ParseTree('cat')
        >>> leaf.is_leaf()
        True
        >>> root = ParseTree('NP')
        >>> root.add_subtree(ParseTree('cat'))
        >>> root.is_leaf()
        False
        """
        return self._subtrees == []

    def __str__(self) -> str:
        """Return a string representation of this parse tree.
        """
        return self._str_indented(0)

    def _str_indented(self, depth: int) -> str:
        """Return an indented string representation of this tree.
        The indentation level is specified by the depth parameter.
        """
        s = '  ' * depth + self.label + '\n'
        for subtree in self._subtrees:
            s += subtree._str_indented(depth + 1)
        return s


def convert_to_parsetree(nltk_tree: nltk.Tree) -> ParseTree:
    """Recursively convert an NLTK Tree into a custom ParseTree instance.

    An NLTK Tree is either:
    - An nltk.Tree object with a string label and a list of children, where
        each child is either another nltk.Tree or a (word, pos_tag) tuple
        representing a leaf token.leaf_tree = nltk.Tree('NP', [('the', 'DT'), ('cat', 'NN')])
    - A (word, pos_tag) tuple at the leaf level (handled inline below).

    Each nltk.Tree node becomes a ParseTree with the same label. Each
    (word, pos_tag) tuple becomes a ParseTree leaf whose label is the word
    string (the pos_tag is discarded, as only the word token is needed).

    Preconditions:
        - nltk_tree is a valid nltk.Tree produced by RegexpParser.parse()

    >>> import nltk
    >>> t = nltk.Tree('NP', [('the', 'DT'), ('cat', 'NN')])
    >>> result = convert_to_parsetree(t)
    >>> result.label
    'NP'
    >>> [child.label for child in result.get_subtrees()]
    ['the', 'cat']
    >>> nested = nltk.Tree('S', [nltk.Tree('NP', [('she', 'PRP')]), ('runs', 'VBZ')])
    >>> r = convert_to_parsetree(nested)
    >>> r.label
    'S'
    >>> r.get_subtrees()[0].label
    'NP'
    >>> r.get_subtrees()[1].label
    'runs'
    """
    node = ParseTree(nltk_tree.label())
    for child in nltk_tree:
        if isinstance(child, tuple):
            node.add_subtree(ParseTree(child[0]))
        else:
            node.add_subtree(convert_to_parsetree(child))
    return node


def get_depth(tree: ParseTree) -> int:
    """Return the depth of tree, defined as the length of the longest
    root-to-leaf path.

    A single leaf node has depth 0. An internal node's depth is one more
    than the maximum depth among its children.

    >>> leaf = ParseTree('cat')
    >>> get_depth(leaf)
    0
    >>> root = ParseTree('NP')
    >>> root.add_subtree(ParseTree('the'))
    >>> root.add_subtree(ParseTree('cat'))
    >>> get_depth(root)
    1
    """
    if tree.is_leaf():
        return 0
    return 1 + max(get_depth(child) for child in tree.get_subtrees())


def count_nodes(tree: ParseTree) -> int:
    """Return the total number of nodes in tree, including both internal
    nodes and leaf nodes.

    >>> leaf = ParseTree('cat')
    >>> count_nodes(leaf)
    1
    >>> root = ParseTree('NP')
    >>> root.add_subtree(ParseTree('the'))
    >>> root.add_subtree(ParseTree('cat'))
    >>> count_nodes(root)
    3
    """
    if tree.is_leaf():
        return 1
    return 1 + sum(count_nodes(child) for child in tree.get_subtrees())


def count_leaves(tree: ParseTree) -> int:
    """Return the number of leaf nodes in tree.

    Leaf nodes represent word tokens in the sentence. A node is a leaf
    if and only if it has no subtrees.

    >>> leaf = ParseTree('cat')
    >>> count_leaves(leaf)
    1
    >>> root = ParseTree('NP')
    >>> root.add_subtree(ParseTree('the'))
    >>> root.add_subtree(ParseTree('cat'))
    >>> count_leaves(root)
    2
    """
    if tree.is_leaf():
        return 1
    return sum(count_leaves(child) for child in tree.get_subtrees())


def count_clauses(tree: ParseTree) -> int:
    """Return the number of clause nodes in tree.

    A clause node is any node whose label is 'S' (a full clause) or
    'SBAR' (a subordinate clause). The count includes tree itself if
    its label qualifies.

    >>> root = ParseTree('S')
    >>> count_clauses(root)
    1
    >>> sub = ParseTree('SBAR')
    >>> root.add_subtree(sub)
    >>> count_clauses(root)
    2
    """
    count = 1 if tree.label in ('S', 'SBAR') else 0
    return count + sum(count_clauses(child) for child in tree.get_subtrees())


def branching_factor(tree: ParseTree) -> float:
    """Return the average branching factor of tree, defined as the mean
    number of children across all internal (non-leaf) nodes.

    Returns 0.0 if tree has no internal nodes (tree is itself a leaf).

    >>> leaf = ParseTree('cat')
    >>> branching_factor(leaf)
    0.0
    >>> root = ParseTree('NP')
    >>> root.add_subtree(ParseTree('the'))
    >>> root.add_subtree(ParseTree('cat'))
    >>> branching_factor(root)
    2.0
    """
    internal_values = []

    def _get_values(t: ParseTree) -> None:
        if not t.is_leaf():
            internal_values.append(len(t.get_subtrees()))
            for child in t.get_subtrees():
                _get_values(child)

    _get_values(tree)
    if not internal_values:
        return 0.0
    return sum(internal_values) / len(internal_values)


_GRAMMAR = r"""
    NP:   {<DT>?<JJ>*<NN.*>+}
    PP:   {<IN><NP>}
    ADJP: {<RB>?<JJ>}
    VP:   {<VB.*><NP|PP|ADJP>*}
    SBAR: {<IN|WDT|WP><NP><VP>}
    S:    {<NP><VP>}
"""
_PARSER = RegexpParser(_GRAMMAR)


def sentence_to_parsetree(sentence: str) -> ParseTree:
    """Converts a raw sentence string into a custom ParseTree using NLTK.

    Preconditions:
        - sentence != ''
    """
    tokens = word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    nltk_tree = _PARSER.parse(tagged)
    return convert_to_parsetree(nltk_tree)


if __name__ == "__main__":
    doctest.testmod(verbose=True)

    python_ta.check_all(config={
        'extra-imports': ['nltk', 'nltk.tree', 'nltk.tokenize'],
        'allowed-io': [],
        'max-line-length': 120
    })
