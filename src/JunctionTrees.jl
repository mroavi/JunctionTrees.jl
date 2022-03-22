module JunctionTrees

import Base:
  eltype,
  IteratorEltype,
  parent

export computeMarginalsExpr,
  Factor,
  product,
  marg,
  redu,
  norm,
  getGraph

export LastStage, ForwardPass, BackwardPass, JointMarginals, UnnormalizedMarginals, Marginals

using Graphs, MetaGraphs, AbstractTrees, MacroTools, Combinatorics, DataStructures

include("factors.jl")
include("tree.jl")
include("utils.jl")
include("junction_tree_algorithm.jl")
include("graphical_transformation.jl")
include("initialization.jl")
include("observation_entry.jl")
include("partial_evaluation.jl")
include("propagation.jl")
include("marginalization.jl")
include("normalization.jl")

getGraph() = @isdefined(g) ? g : nothing

end # module

