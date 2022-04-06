# Visualization of Markov random fields

The Markov random fields used in JunctionTrees.jl can be conveniently
visualized with [GraphMakie.jl](https://github.com/JuliaPlots/GraphMakie.jl).
This document presents a series of examples that illustrate different
possibilities of plotting this type of graphs.

#### Example 1

Plots a Markov random field with no node labels.

```@example
using JunctionTrees, CairoMakie, GraphMakie

nvars, _, _, factors = JunctionTrees.read_uai_file("../problems/paskin/paskin.uai")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)

fig = Figure(backgroundcolor = :white, resolution=(900, 300))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(ax, mrf, layout = GraphMakie.NetworkLayout.Stress(seed=1))

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
fig
```
######

#### Example 2

Plots a Markov random field with variable IDs as node labels.

```@example
using JunctionTrees, CairoMakie, GraphMakie, Graphs, Printf

nvars, _, _, factors = JunctionTrees.read_uai_file("../problems/paskin/paskin.uai")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)

fig = Figure(backgroundcolor = :white, resolution=(900, 300))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(
  ax,
  mrf,
  layout = GraphMakie.NetworkLayout.Stress(seed=1),
  nlabels = [@sprintf "%i" v for v in vertices(mrf)],
  nlabels_align = (:center,:center),
  nlabels_color = :white,
  nlabels_textsize = 14,
  node_size = 25,
)

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
fig
```
######

#### Example 3

Plots a Markov random field with variable IDs as node labels. Observed
variables are colored in red.

```@example
using JunctionTrees, CairoMakie, GraphMakie, Graphs, Printf, ColorSchemes

nvars, _, _, factors = JunctionTrees.read_uai_file("../problems/paskin/paskin.uai")
obsvars, obsvals = JunctionTrees.read_uai_evid_file("../problems/paskin/paskin.uai.evid")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)

# Default node color
node_color = [ColorSchemes.watermelon[1] for i in 1:nv(mrf)]

# Observed node color
node_color[obsvars] .= ColorSchemes.watermelon[8]

fig = Figure(backgroundcolor = :white, resolution=(900, 300))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(
  ax,
  mrf,
  layout = GraphMakie.NetworkLayout.Stress(seed=1),
  nlabels = [@sprintf "%i" v for v in vertices(mrf)],
  nlabels_align = (:center,:center),
  nlabels_color = :white,
  nlabels_textsize = 14,
  node_size = 25,
  node_color = node_color,
)

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
fig
```
######

#### Example 4

Plots a Markov random field of problem 34 of the [Promedus
benchmark](https://github.com/PACE-challenge/UAI-2014-competition-graphs).

```@example
using JunctionTrees, CairoMakie, GraphMakie, Graphs, Printf, ColorSchemes

nvars, _, _, factors = JunctionTrees.read_uai_file("../problems/Promedus_34/Promedus_34.uai")
obsvars, obsvals = JunctionTrees.read_uai_evid_file("../problems/Promedus_34/Promedus_34.uai.evid")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)

# Default node color
node_color = [ColorSchemes.watermelon[1] for i in 1:nv(mrf)]

# Observed node color
node_color[obsvars] .= ColorSchemes.watermelon[8]

fig = Figure(backgroundcolor = :white, resolution=(900, 600))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(ax, mrf, layout = GraphMakie.NetworkLayout.Stress(seed=1), node_color = node_color)

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
fig
```
