# PACE files format

This document defines the following file formats:

- [PACE graph format](@ref).
- [PACE tree decomposition format](@ref).

## PACE graph format

We describe the file format *.gr*, which is similar to the format used by
DIMACS challenges.

Lines are separated by the character ‘\n’. Each line that starts with the
character `c` is considered to be a comment line. The first non-comment line
must be a line starting with p followed by the problem descriptor `tw` and the
number of vertices ``n`` and edges ``m`` (separated by a single space each
time). No other line may start with `p`. Every other line indicates an edge,
and must consist of two decimal integers from 1 to n separated by a space;
moreover, graphs are considered undirected (though they may contain isolated
vertices, multiple edges, and loops). For example, a path with four edges can
be defined as follows:

    c This file describes a path with five vertices and four edges.
    p tw 5 4
    1 2
    2 3
    c we are half-way done with the instance definition.
    3 4
    4 5

## PACE tree decomposition format

Recall the definition of a tree decomposition of a graph ``G``: It is a tree
``T`` such that every vertex ``x`` in ``V(T)`` has an associated bag ``B(x)``
that is a subset of ``V(G)``. Every edge e in ``E(G)`` must be a subset of at
least one bag ``B(x)``. Moreover, for every vertex ``v`` in ``V(G)``, the set
of tree vertices whose bags contain ``v`` induce a connected subtree of ``T``.
The width of ``T`` is the maximum size of its bags minus one. The goal is to
compute a tree decomposition of minimum width.

We define the file format *.td*. As above, `c` lines are comments and can occur
throughout the file. Instead of a `p`-line, we now expect a unique solution
line s as the first non-comment line, which contains the string `td`, followed
by the number ``N`` of bags of the tree decomposition, the width of the tree
decomposition plus one (i.e., the largest bag size), as well as the number of
vertices of the original input graph. The next non-comment lines we expect
start with b and specify the contents of each bag; for example, `b 4 3 4 6 7`
specifies that bag number 4 contains the vertices 3, 4, 6, and 7 of the
original graph. Bags may be empty. For every bag ``i``, there must be exactly
one line starting with `b` ``i``. All remaining non-comment lines indicate an
edge in the tree decomposition, so it must consist of two decimal integers from
1 and ``N`` where the first integer is smaller than the second, and the graph
described this way must be a tree. For example, the following is a suboptimal
tree decomposition of the path with four edges.

    c This file describes a tree decomposition with 4 bags, width 2, for a graph with 5 vertices
    s td 4 3 5
    b 1 1 2 3
    b 2 2 3 4
    b 3 3 4 5
    b 4
    1 2
    2 3
    2 4

## Validity checker

The validity checker at <https://github.com/holgerdell/td-validate/> verifies
that a given tree decomposition (specified in the *.td* file format) is indeed
a valid tree decomposition of a given graph (specified in the *.gr* format).
The validity checker will be used to evaluate whether your submitted
implementation computed a correct tree decomposition.
