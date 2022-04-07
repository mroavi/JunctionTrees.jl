# Background

The junction tree algorithm is a general algorithmic framework to perform
discrete inference in general graphs, such as Bayesian Networks or Markov
random fields. The general problem is to calculate the conditional probability
of a variable or a set of variables, given observed values of another set of
variables. This is known as the inference problem.

## The inference problem

Given a set of **random variables** ``\mathcal{V}`` and their **joint
distribution** ``P(\mathcal{V})``, compute one or more conditional
distributions over a set of **query variables** ``\bm{Q}`` given observations
``\bm{e}`` for the set of **observed variables** ``\bm{E}``.

```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    %\draw[help lines] (0,0) grid (10,-7);

    % mrv: the "node distances" refer to the distance between the edge of a shape
    % to the edge of the other shape. That is why I use "ie_aux" and "mar_aux"
    % below: to have equal distances between nodes with respect to the center of
    % the shapes.

    % row 1
    \node[myroundbox] (rv) {Random Variables\\$\mathcal{V}$};
    \node[right=of rv](aux1) {};
    \node[right=of aux1,myroundbox] (jd) {Joint Distribution\\$P(\mathcal{V})$};
    \node[right=of jd](aux2) {};
    \node[right=of aux2,myroundbox] (e) {Evidence\\$\bm{E=e}$};
    \node[right=of e](aux3) {};
    \node[right=of aux3,myroundbox] (qv) {Query Variables\\$\bm{Q}$};
    % row 2
    \node[below=of aux2,myrectbox] (ie) {Inference Engine};
    \node[below=of aux2] (ie_aux) {};
    % row 3
    \node[below=of ie_aux,myroundbox] (mar) {$P(\bm{Q} \mid {\bf E=e})$};
    \node[below=of ie_aux] (mar_aux) {};
    % row 0
    \node[above=of aux2,yshift=-12mm,text=black] (in) {\textbf{Input}};
    % row 4
    \node[below=of mar_aux,yshift=7mm,text=black] (out) {\textbf{Output}};

    %% edges
    \draw[myarrow] (rv) -- (ie);
    \draw[myarrow] (jd) -- (ie);
    \draw[myarrow] (e)  -- (ie);
    \draw[myarrow] (qv) -- (ie);
    \draw[myarrow] (ie) -- (mar);
  """,
  options="transform shape, scale=1.4",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "the-inference-problem") * "}",
)
save(SVG("the-inference-problem"), tp)
```
![](the-inference-problem.svg)


## The junction tree algorithm

We briefly describe the junction tree algorithm (JTA). For a more elaborate
presentation, see [^huang1996inference]. The figure below presents an overview
of the JTA.

```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    \node[mybox] (pgm) {Probabilistic Graphical Model};
    \node[mybox,below=of pgm,yshift=-0.4cm] (jts) {Junction Tree};
    \node[mybox,below=of jts] (ijt) {Inconsistent Junction Tree};
    \node[mybox,below=of ijt,yshift=0.4cm] (cjt) {Consistent Junction Tree};
    \node[mybox,below=of ijt,yshift=0.4cm] (cjt) {Consistent Junction Tree};

    \node[text=black,below=of cjt] (mar) {$P(V \mid \bm{E=e})$};

    \draw[myarrow] (pgm) -- (jts);
    \draw[myarrow] (jts) -- (ijt);
    \draw[myarrow] (ijt) -- (cjt);
    \draw[myarrow] (cjt) -- (mar);

    \path (pgm) -- node[mylabel] (gt) {1. Moralization\\2. Triangulation\\3. Connection of clusters} (jts);
    \path (jts) -- node[mylabel] (ini) {1. Initialization\\2. Observation entry} (ijt);
    \path (ijt) -- node[mylabel] (pro) {Propagation} (cjt);
    \path (cjt) -- node[mylabel] (mar-nor) {1. Marginalization\\2. Normalization} (mar);
  """,
  options="transform shape, scale=1.4",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "pptc-flow-diagram") * "}",
)
save(SVG(joinpath(@__DIR__, "pptc-flow-diagram")), tp)
```
![](pptc-flow-diagram.svg)

A probabilistic graphical model (PGM) is the input to the JTA. PGMs illustrate
the mathematical modeling of reasoning in the presence of uncertainty. Bayesian
networks and Markov random fields are popular types of PGMs. Consider the
Bayesian network shown in the figure below. It consists of a graph ``G =
(\mathcal{V},\mathcal{E})`` and a probability distribution ``P(\mathcal{V})``
where ``G`` is a directed acyclic graph, ``\mathcal{V}`` is the set of
variables and ``\mathcal{E}`` is the set of edges connecting the variables. We
assume all variables to be discrete. Each variable ``V`` is quantified with a
*conditional probability distribution* ``P(V \mid pa(V))`` where ``pa(V)`` are
the parents of ``V``. These conditional probability distributions together with
the graph ``G`` induce a *joint probability distribution* over
``P(\mathcal{V})``, given by
```math
P(\mathcal{V}) = \prod_{V\in\mathcal{V}} P(V \mid pa(V)).
```

| **Random variable**  | **Meaning**                   |
|        :---:         | :---                          |
|        ``A``         | Recent trip to Asia           |
|        ``T``         | Patient has tuberculosis      |
|        ``S``         | Patient is a smoker           |
|        ``L``         | Patient has lung cancer       |
|        ``B``         | Patient has bronchitis        |
|        ``E``         | Patient hast `T` and/or `L`   |
|        ``X``         | Chest X-Ray is positive       |
|        ``D``         | Patient has dyspnoea          |

```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    % The various elements are conveniently placed using a matrix:
    \matrix[row sep=0.5cm,column sep=0.5cm] {
      % First line
      \node (a) [mybag] {$A$};  &
                                &
                                &
      \node (s) [mybag] {$S$};  &
                               \\
      % Second line
      \node (t) [mybag] {$T$};  &
                                &
      \node (l) [mybag] {$L$};  &
                                &
      \node (b) [mybag] {$B$}; \\
      % Third line
                                &
      \node (e) [mybag] {$E$};  &
                                &
                                &
                               \\
      % Forth line
      \node (x) [mybag] {$X$};  &
                                &
                                &
      \node (d) [mybag] {$D$};  &
                               \\
  };

  \draw [myarrow] (a) edge (t);
  \draw [myarrow] (s) edge (l);
  \draw [myarrow] (s) edge (b);
  \draw [myarrow] (t) edge (e);
  \draw [myarrow] (l) edge (e);
  \draw [myarrow] (e) edge (x);
  \draw [myarrow] (e) edge (d);
  \draw [myarrow] (b) edge (d);
  """,
  options="transform shape, scale=1.4",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "asia", "bayesian-network") * "}",
)
save(SVG(joinpath(@__DIR__, "asia-bayesian-network")), tp)
```
![](asia-bayesian-network.svg)

### Graphical transformation

JTA performs probabilistic inference on a secondary structure known as a
*junction tree*. A junction tree is constructed from a PGM by means of two
graphical transformations: *moralization* and *triangulation*.

#### Moralization

Moralization converts a directed acyclic graph into a undirected graph by
dropping the directions of the edges and connecting the parents of each node.
The figure below shows the corresponding moral graph of the Bayesian network in
the figure above.

```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    \matrix[row sep=0.5cm,column sep=0.5cm] {
      % First line
      \node (a) [mybag, fill=A] {$A$};  &
                                        &
                                        &
      \node (s) [mybag, fill=S] {$S$};  &
                                       \\
      % Second line
      \node (t) [mybag, fill=T] {$T$};  &
                                        &
      \node (l) [mybag, fill=L] {$L$};  &
                                        &
      \node (b) [mybag, fill=B] {$B$}; \\
      % Third line
                                        &
      \node (e) [mybag, fill=E] {$E$};  &
                                        &
                                        &
                                       \\
      % Forth line
      \node (x) [mybag, fill=X] {$X$};  &
                                        &
                                        &
      \node (d) [mybag, fill=D] {$D$};  &
                                       \\
    };

    \draw (a) edge (t);
    \draw (s) edge (l);
    \draw (s) edge (b);
    \draw (t) edge (l); % <- added edge
    \draw (t) edge (e);
    \draw (l) edge (e);
    \draw (e) edge (x);
    \draw (e) edge (d);
    \draw (b) edge (e); % <- added edge
    \draw (b) edge (d);

    \path (t) -- node[anchor=center,xshift=-0.0cm,yshift=0.3cm,rotate=-90] (be) {$\rightarrow$} (l);
    \path (b) -- node[anchor=center,xshift=0.2cm,yshift=-0.3cm,rotate=110] (be) {$\rightarrow$} (e);
  """,
  options="transform shape, scale=1.4",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "asia", "moral-graph") * "}",
)
save(SVG(joinpath(@__DIR__, "asia-moral-graph")), tp)
```
![](asia-moral-graph.svg)

#### Triangulation

Triangulation of an undirected graph is carried out by connecting two
non-adjacent nodes in every cycle of length greater than three. The figure
below shows a triangulated graph of the moral graph in the figure above. Note
that, in general, there is more than one way of triangulating a given
undirected graph. An optimal triangulation is one that minimizes the sum of the
state space sizes of the *maximal cliques*[^1] (denoted with colored boundaries
in the figure below. This problem is NP-complete [^arnborg1987complexity].

```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    % The various elements are conveniently placed using a matrix:
    \matrix[row sep=0.5cm,column sep=0.5cm] {
      % First line
      \node (a) [mybag] {$A$};  &
                              &
                              &
      \node (s) [mybag] {$S$};  &
                            \\
      % Second line
      \node (t) [mybag] {$T$};  &
                              &
      \node (l) [mybag] {$L$};  &
                              &
      \node (b) [mybag] {$B$}; \\
      % Third line
                              &
      \node (e) [mybag] {$E$};  &
                              &
                              &
                            \\
      % Forth line
      \node (x) [mybag] {$X$};  &
                              &
                              &
      \node (d) [mybag] {$D$};  &
                            \\
    };

    \draw (a) edge (t);
    \draw (s) edge (l);
    \draw (s) edge (b);
    \draw (t) edge (l);
    \draw (t) edge (e);
    \draw (l) edge (e);
    \draw (l) edge (b); % <- added edge
    \draw (e) edge (x);
    \draw (e) edge (d);
    \draw (b) edge (e);
    \draw (b) edge (d);

    \path (l) -- node[anchor=center,xshift=-0.15cm,yshift=-0.25cm,rotate=90] (bc) {$\rightarrow$} (b);

    \draw[AT, myclique] \convexpath{a,t}{15pt};
    \draw[TLE, myclique] \convexpath{t,l,e}{15pt};
    \draw[EX, myclique] \convexpath{e,x}{15pt};
    \draw[LBS, myclique] \convexpath{l,s,b}{15pt};
    \draw[LBE, myclique] \convexpath{l,b,e}{15pt};
    \draw[DEB, myclique] \convexpath{d,e,b}{15pt};
  """,
  options="transform shape, scale=1.4",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "asia", "triangulated-graph") * "}",
)
save(SVG(joinpath(@__DIR__, "asia-triangulated-graph")), tp)
```
![](asia-triangulated-graph.svg)

#### Connection of clusters

The maximal cliques of the triangulated graph correspond to the nodes of the
junction tree. We call these *clusters*. Clusters are then connected in a tree
structure such that the *running intersection property* is satisfied: given two
clusters ``\bm{X}`` and ``\bm{Y}`` in the tree, all clusters on the path
between ``\bm{X}`` and ``\bm{Y}`` contain ``\bm{X} \cap \bm{Y}``. Each edge is
labeled with the intersection of the adjacent clusters. Such labels are called
separator sets or *sepsets*. [^jensen1994optimal] present an optimal method to
construct a junction tree from a triangulated graph. The figure below shows the
result of applying this method to the triangulated graph in the figure above.

```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    % The various elements are conveniently placed using a matrix:
    \matrix[row sep=0.26cm,column sep=0.20cm] {
      % First line
      \node (lbs) [mybag, draw=LBS] {$\{\Circled[fill color=L]{L},\Circled[fill color=B]{B},\Circled[fill color=S]{S}\}$};  &
      \node (lb) [myvsepset] {$\{L,B\}$};                            &
      \node (lbe) [mybag, draw=LBE] {$\{L,B,E\}$};  &
      \node (eb) [myvsepset] {$\{E,B\}$};                          &
      \node (deb) [mybag, draw=DEB] {$\{\Circled[fill color=D]{D},E,B\}$}; \\
      % Second line
                                                          &
                                                          &
      \node (le) [myhsepset] {$\{L,E\}$};                    &
                                                          &
                                                          \\
      % Third line
      \node (at) [mybag, draw=AT] {$\{\Circled[fill color=A]{A},\Circled[fill color=T]{T}\}$};  &
      \node (t) [myvsepset] {$\{T\}$};                    &
      \node (tle) [mybag, draw=TLE] {$\{T,L,\Circled[fill color=E]{E}\}$};  &
      \node (e) [myvsepset] {$\{E\}$};                    &
      \node (ex) [mybag,draw=EX] {$\{E,\Circled[fill color=X]{X}\}$}; \\
    };

    % The diagram elements are now connected through lines:
    \path[-]
      (lbs) edge (lb)
      (lb) edge (lbe)
      (lbe) edge (eb)
      (eb) edge (deb)
      (lbe) edge (le)
      (le) edge (tle)
      (at) edge (t)
      (t) edge (tle)
      (tle) edge (e)
      (e) edge (ex)
      ;
  """,
  options="transform shape, scale=1.4",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "asia", "junction-tree") * "}",
)
save(SVG(joinpath(@__DIR__, "asia-junction-tree")), tp)
```
![](asia-junction-tree.svg)

### Initialization

Each cluster ``{\bf X}`` in the junction tree is associated with a *potential*
``\psi_{{\bf X}}``. A potential is a function over a set of variables
``\bm{V}`` that maps each instantiation ``\bm{V} = \bm{v}`` into a nonnegative
number. First, all cluster potentials in the junction tree are initialized to
unity. Then, each conditional probability distribution ``P(V \mid pa(V))`` in
the equation presented in section [The junction tree algorithm](@ref) is
multiplied into a cluster potential ``{\bf X}`` that contains its variable and
its parents:
```math
\psi_{{\bf X}} \leftarrow \psi_{{\bf X}} \cdot P(V \mid pa(V)).
```
Note that a probability distribution is a special case of a potential. The
encircled variables in the figure above indicate which conditional
distributions in the Bayesian network figure were multiplied into which cluster
potentials of our running example.

### Observation entry

Observations take the form of ``{\bf E=e}``, where ``{\bf e}`` is the
instantiation of the set of variables ``{\bf E}``. These are incorporated into
the junction tree by finding a cluster potential ``\psi_{{\bf X}}`` for each
evidence variable in ``{\bf E}`` that contains it and setting all its entries
that are not consistent with the evidence to zero. This operation is known as a
*reduction* in the PGM literature [^koller2009probabilistic].

### Propagation

Propagation refers to a series of synchronized local manipulations between
clusters that guarantee consistency throughout the entire junction tree. These
manipulations are called *messages*. The propagation of messages begins by
designating an arbitrary cluster as the *root*, which gives direction to the
edges of the junction tree. Messages then flow between clusters in two
recursive phases: an *inward* and an *outward* phase. In the inward phase, each
cluster passes a message to its parent. In the outward phase, each cluster
passes a message to each of its children. A cluster passes a message to a
neighbor only after it has received messages from all its *other* neighbors. A
message from cluster ``{\bf X}`` to cluster ``{\bf Y}`` is a potential
``\phi_{{\bf X} \rightarrow {\bf Y}}`` defined by
```math
\phi_{{\bf X} \rightarrow {\bf Y}} =
\sum_{{\bf X} \setminus {\bf Y}} \psi_{{\bf X}} \prod_{{\bf N} \in
\mathcal{N}_{{\bf X} \setminus {\bf Y}}} \phi_{{\bf N} \rightarrow {\bf X}},
```
where ``\psi_{{\bf X}}`` is the cluster potential of ``{\bf X}`` and
``\mathcal{N}_{{\bf X}}`` is the set of neighbors of ``{\bf X}``. The figure
below shows an admissible schedule for the propagation of messages of our
running example.

```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    % The various elements are conveniently placed using a matrix:
    \matrix[row sep=1.0cm,column sep=1.0cm] {
      % First line
      \node (lbs) [mybag, label={below:{Root}}] {$\{L,B,S\}$};  &
      \node (lbe) [mybag] {$\{L,B,E\}$};                        &
      \node (deb) [mybag] {$\{D,E,B\}$};                       \\
      % Second line
      \node (at) [mybag] {$\{A,T\}$};                           &
      \node (tle) [mybag] {$\{T,L,E\}$};                        &
      \node (ex) [mybag] {$\{E,X\}$};                          \\
    };
      
    % The diagram elements are now connected through lines:
    \path[-]
      (lbs) edge (lbe)
      (lbe) edge (deb)
      (lbe) edge (tle)
      (at) edge (tle)
      (tle) edge (ex)
      ;

    \msgcircle{up}{right}{at}{tle}{0.5}{$1$}{IP};
    \msgcircle{up}{left}{ex}{tle}{0.5}{$2$}{IP};
    \msgcircle{left}{up}{tle}{lbe}{0.5}{$3$}{IP};
    \msgcircle{up}{left}{deb}{lbe}{0.5}{$4$}{IP};
    \msgcircle{up}{left}{lbe}{lbs}{0.5}{$5$}{IP};
    \msgcircle{down}{right}{lbs}{lbe}{0.5}{$6$}{OP};
    \msgcircle{right}{down}{lbe}{tle}{0.5}{$7$}{OP};
    \msgcircle{down}{right}{lbe}{deb}{0.5}{$8$}{OP};
    \msgcircle{down}{left}{tle}{at}{0.5}{$9$}{OP};
    \msgcircle{down}{right}{tle}{ex}{0.5}{$10$}{OP};
  """,
  options="transform shape, scale=1.4",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "asia", "message-passing") * "}",
)
save(SVG(joinpath(@__DIR__, "asia-message-passing")), tp)
```
![](asia-message-passing.svg)

### Marginalization

After the propagation phase, each edge holds two messages; one in each
direction. The joint marginal probabilities for each sepset are given by the
product of the two messages passing through the corresponding edge, i.e.
```math
P(S_{{\bf X}{\bf Y}}, {\bf E=e}) = \phi_{{\bf X} \rightarrow {\bf Y}} \cdot
  \phi_{{\bf Y} \rightarrow {\bf X}},
```
where ``{\bf X}`` and ``{\bf Y}`` are adjacent clusters. On the other hand, the
joint marginal probabilities for each cluster are given by the product of the
cluster's incoming messages and its potential, i.e.
```math
P({\bf X}, {\bf E=e}) = \psi_{{\bf X}} \prod_{{\bf N} \in \mathcal{N}_{{\bf
X}}} \phi_{{\bf N} \rightarrow {\bf X}}.
```
The marginal probability $P(V,{\bf E=e})$ for each variable of interest $V$ is
then computed from the joint marginal of a sepset or cluster containing $V$.
This is achieved by marginalizing all other variables:
```math
P(V,{\bf E=e}) = \sum_{{\bf X^\prime} \setminus V} P({\bf X^\prime}, {\bf
E=e}),
```
where ``{\bf X^\prime}`` is a sepset or cluster potential that contains ``V``.

### Normalization

The last step is to compute $P(V \mid {\bf E=e})$ for each variable of interest
$V$. We do so by normalizing $P(V, {\bf E=e})$:
```math
P(V \mid {\bf E=e}) = \frac{P(V, {\bf E=e})}{\sum_{V} P(V, {\bf E=e})}.
```

## Compiler-based framework

JunctionTrees.jl exploits Julia's metaprogramming capabilities to separate the
algorithm into two phases: a compilation and a runtime phase. The compilation
phase consists of the creation and subsequent optimization of the algorithm.
The run-time phase consists of processing online data with the compiled
algorithm to provide answers about variables of interest. This distinction
between a compilation and a runtime phase opens a wide range of optimization
possibilities in an offline stage that aims to generate the minimal piece of
software that is required online. Moreover, it allows moving the computational
burden from the runtime to the compilation phase. The figure below illustrates
the compiler-based framework design used in JunctionTrees.jl.

```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    %\draw[help lines] (0,0) grid (7,-9);

    %\draw [thick] (-1.5, 0.6) rectangle (7.3, -7.45);
    \draw [dashed,draw=black,line width=2pt] (-7.8, -4.8) -- (2.4, -4.8);

    \node [above,yshift=4.0pt,text=black] at (-7.2, -4.75) {Off-line};
    \node [below,yshift=-4.0pt,text=black] at (-7.2, -4.85) {On-line};

    % row 1
    \node[myroundbox] (pgm) {PGM};
    \node[myroundbox,left=of pgm] (qv) {Query Vars};
    % row 2
    \node[myrectbox,below=of pgm] (pptc) {JTA};
    % row 3
    \node[myroundbox,below=of pptc] (sup) {Subject Program};
    \node[myroundbox,left=of sup] (evv) {Evidence Vars};
    % row 4
    \node[myrectbox,below=of sup] (pe) {Optimization};
    % row 5
    \node[myroundbox,below=of pe,yshift=-8mm] (spp) {Optimized Program};
    \node[myroundbox,left=of spp] (ev) {Evidence};
    % row 6
    \node[myroundbox,below=of spp] (mar) {$P(\text{Query Vars} \mid {\bf E=e})$};
    % edges
    \draw[myarrow] (pgm) -- (pptc);
    \draw[myarrow] (qv) -- (pptc);
    \draw[myarrow] (pptc) -- (sup);
    \draw[myarrow] (sup) -- (pe);
    \draw[myarrow] (evv) -- (pe);
    \draw[myarrow] (pe) -- (spp);
    \draw[myarrow] (ev) -- (spp);
    \draw[myarrow] (spp) -- (mar);
  """,
  options="transform shape, scale=1.4",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "compiler-based-framework") * "}",
)
save(SVG(joinpath(@__DIR__, "compiler-based-framework")), tp)
```
![](compiler-based-framework.svg)

[^1]: A clique in an undirected graph is a subgraph in which every pair of nodes is connected by an edge. A maximal clique is a clique that is not contained in a larger clique.

[^huang1996inference]:
    Cecil Huang and Adnan Darwiche. Inference in belief networks: A procedural guide. International Journal of Approximate Reasoning, 15 (3):225–263, 1996. ISSN 0888-613X. doi: <https://doi.org/10.1016/S0888-613X(96)00069-2>. URL <https://www.sciencedirect.com/science/article/pii/S0888613X96000692>.

[^arnborg1987complexity]:
    Stefan Arnborg, Derek G Corneil, and Andrzej Proskurowski. Complexity of finding embeddings in ak-tree. SIAM Journal on Algebraic Discrete Methods, 8(2):277–284, 1987.

[^jensen1994optimal]:
    Finn V. Jensen and Frank Jensen. Optimal junction trees. In Proceedings of the Tenth International Conference on Un- certainty in Artificial Intelligence, UAI’94, page 360–366, San Francisco, CA, USA, 1994. Morgan Kaufmann Publishers Inc. ISBN 1558603328.

[^koller2009probabilistic]:
    Daphne Koller and Nir Friedman. Probabilistic graphical models: principles and techniques, pg 111. MIT press, 2009.
