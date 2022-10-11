"""
Main module for `JunctionTrees.jl` -- a Julia implementation of the junction tree algorithm.

One main function is exported from this module for public use:

- [`compile_algo`](@ref). Compiles and returns an expression that computes the posterior marginals of the model given evidence using the junction tree algorithm.

# Exports

$(EXPORTS)
"""
module JunctionTrees

using Graphs, MetaGraphs, AbstractTrees, DataStructures, DocStringExtensions, OMEinsum
using Combinatorics: combinations
using MacroTools: @capture, rmlines
using MLStyle: @match

export compile_algo, Factor, prod, sum, redu, norm, LastStage, ForwardPass,
  BackwardPass, JointMarginals, UnnormalizedMarginals, Marginals

import Base:
  prod,
  sum,
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
include("boost.jl")

end # module
