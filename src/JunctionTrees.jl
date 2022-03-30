"""

Main module for `JunctionTrees.jl` -- a Julia implementation of the junction tree algorithm.

One main function is exported from this module for public use:

- [`computeMarginalsExpr`](@ref). Computes the posterior marginals using the
junction tree algorithm given evidence.

# Exports

$(EXPORTS)

"""
module JunctionTrees

using Graphs, MetaGraphs, AbstractTrees, MacroTools, Combinatorics,
  DataStructures, DocStringExtensions

export computeMarginalsExpr, Factor, product, marg, redu, norm, getGraph,
  LastStage, ForwardPass, BackwardPass, JointMarginals, UnnormalizedMarginals,
  Marginals

import Base:
  eltype,
  IteratorEltype,
  parent

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

"""
$(TYPEDSIGNATURES)

Return the current graph.
"""
getGraph() = @isdefined(g) ? g : nothing

end # module
