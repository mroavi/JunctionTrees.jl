"""
Main module for `JunctionTrees.jl` -- a Julia implementation of the junction tree algorithm.

One main function is exported from this module for public use:

- [`compile_algo`](@ref). Compiles and returns an expression that computes the posterior marginals of the model given evidence using the junction tree algorithm.

# Exports

$(EXPORTS)
"""
module JunctionTrees

using Graphs, MetaGraphs, AbstractTrees, MacroTools, Combinatorics,
  DataStructures, DocStringExtensions

export compile_algo, Factor, prod, marg, redu, norm, LastStage, ForwardPass,
  BackwardPass, JointMarginals, UnnormalizedMarginals, Marginals

import Base:
  prod,
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

end # module
