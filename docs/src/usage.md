# Usage

This section presents a series of examples that illustrate different uses of
JunctionTrees.jl. Load the package to run the examples.

```@example main
using JunctionTrees
```

#### Example 1

Calculates the posterior marginal of each variable in the input graph. The
input graph should be defined in the [UAI model file format](@ref).

```@example main
algo = @posterior_marginals("problems/asia/asia.uai")
obsvars, obsvals = Int64[], Int64[]
marginals = algo(obsvars, obsvals)
```

#### Example 2

Calculates the posterior marginal of each variable in the input graph given
some evidence. The input graph should be defined in the [UAI model file
format](@ref). The evidence variables and values should be given in the [UAI
evidence file format](@ref).

```@example main
algo = @posterior_marginals(
         "problems/asia/asia.uai",
         uai_evid_filepath = "problems/asia/asia.uai.evid")
obsvars, obsvals = JunctionTrees.read_uai_evid_file("problems/asia/asia.uai.evid")
marginals = algo(obsvars, obsvals)
```

#### Example 3

Same as the previous example with the difference that a pre-constructed
junction tree (which is passed as an argument) is used. This junction tree
should be defined in the [PACE graph format](@ref).

```@example main
algo = @posterior_marginals(
         "problems/asia/asia.uai",
         uai_evid_filepath = "problems/asia/asia.uai.evid",
         td_filepath = "problems/asia/asia.td")
obsvars, obsvals = JunctionTrees.read_uai_evid_file("problems/asia/asia.uai.evid")
marginals = algo(obsvars, obsvals)
```

#### Example 4

Returns the expression of the junction tree algorithm up to the backward pass
stage.

```@example main
backward_pass_expr = @posterior_marginals("problems/asia/asia.uai", last_stage = BackwardPass)
```

The stages supported are:

```@example main
instances(LastStage)
```

