# JunctionTrees.jl

*A metaprogramming-based implementation of the junction tree algorithm.*

```@raw html
<div align="left">
<img width="600px" src="./problems/Promedus_34/Promedus_34.svg" alt="Junction tree example"></img>
</div>
```

## Package features

- Posterior marginal computation of discrete variables given evidence.
- Metaprogramming-based design that separates the off-line and on-line
  computations for better efficiency of real-time inference.
- Visualization of junction trees, Bayesian networks and Markov random fields.

## Outline
```@contents
Pages = [
  "background.md",
  "usage.md",
  "visualization/junction_trees.md",
  "visualization/markov_random_fields.md",
  "file_formats/uai.md",
  "file_formats/pace.md",
  "library/public.md",
  "library/internals.md",
]
Depth = 1
```
