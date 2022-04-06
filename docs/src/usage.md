# Usage

This section presents a series of examples that illustrate different uses of
JunctionTrees.jl.

#### Example 1

Calculates the posterior marginals of each variable in the input graph. The
input graph should be defined in the [UAI model file format](@ref).

```@example
using JunctionTrees

algo = compile_algo("problems/paskin/paskin.uai")
eval(algo)
obsvars, obsvals = Int64[], Int64[]
marginals = run_algo(obsvars, obsvals)
```

#### Example 2

Calculates the posterior marginals of each variable in the input graph given
some evidence. The input graph should be defined in the [UAI model file
format](@ref). The evidence variables and values should be given in the [UAI
evidence file format](@ref).

```@example
using JunctionTrees

algo = compile_algo(
         "problems/paskin/paskin.uai",
         uai_evid_filepath = "problems/paskin/paskin.uai.evid")
eval(algo)
obsvars, obsvals = JunctionTrees.read_uai_evid_file("problems/paskin/paskin.uai.evid")
marginals = run_algo(obsvars, obsvals)
```

#### Example 3

Same as the previous example with the difference that a pre-constructed
junction tree (which is passed as an argument) is used. This junction tree
should be defined in the [PACE graph format](@ref).

```@example
using JunctionTrees

algo = compile_algo(
         "problems/paskin/paskin.uai",
         uai_evid_filepath = "problems/paskin/paskin.uai.evid",
         td_filepath = "problems/paskin/paskin.td")
eval(algo)
obsvars, obsvals = JunctionTrees.read_uai_evid_file("problems/paskin/paskin.uai.evid")
marginals = run_algo(obsvars, obsvals)
```

#### Example 4

Returns the expression of the junction tree algorithm up to the backward pass
stage.

```@example
using JunctionTrees

backward_pass_expr = compile_algo(
                       "problems/paskin/paskin.uai",
                       last_stage = BackwardPass)
```

The stages supported are:

```@example
using JunctionTrees

instances(LastStage)
```

