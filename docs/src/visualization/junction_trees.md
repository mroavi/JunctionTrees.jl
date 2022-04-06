# Visualization of junction trees

The junction trees used in JunctionTrees.jl can be conveniently visualized with
[GraphMakie.jl](https://github.com/JuliaPlots/GraphMakie.jl). This document
presents a series of examples that illustrate different possibilities of
plotting this type of graphs.

This section presents different examples of plotting junction trees.

#### Example 1

Plots a junction tree with no node or edge labels.

```@example
using JunctionTrees, CairoMakie, GraphMakie

nvars, cards, _, factors = JunctionTrees.read_uai_file("../problems/paskin/paskin.uai")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)
td = JunctionTrees.construct_td_graph(mrf, cards)

fig = Figure(backgroundcolor = :white, resolution=(900, 300))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(ax, td, layout = GraphMakie.NetworkLayout.Stress(seed=58))

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
fig
```
######

#### Example 2

Plots a junction tree with cluster IDs as node labels.

```@example
using JunctionTrees, CairoMakie, GraphMakie, Graphs, Printf

nvars, cards, _, factors = JunctionTrees.read_uai_file("../problems/paskin/paskin.uai")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)
td = JunctionTrees.construct_td_graph(mrf, cards)

fig = Figure(backgroundcolor = :white, resolution=(900, 300))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(
  ax,
  td,
  layout = GraphMakie.NetworkLayout.Stress(seed=58),
  nlabels = [@sprintf "%i" v for v in vertices(td)],
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

Plots a junction tree with cluster variables as node labels.

```@example
using JunctionTrees, CairoMakie, GraphMakie, Graphs, Printf, MetaGraphs

nvars, cards, _, factors = JunctionTrees.read_uai_file("../problems/paskin/paskin.uai")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)
td = JunctionTrees.construct_td_graph(mrf, cards)

fig = Figure(backgroundcolor = :white, resolution=(900, 400))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(
  ax,
  td,
  layout = GraphMakie.NetworkLayout.Stress(seed=58),
  nlabels = map(vertex -> get_prop(td, vertex, :vars), vertices(td)) |> x -> string.(x),
  nlabels_align = (:center,:center),
  nlabels_color = :white,
  nlabels_textsize = 14,
  node_size = 80,
)

xlims!(-1.5, 1.5)
ylims!(-0.9, 1.4)

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
fig
```
######

#### Example 4

Plots a junction tree with cluster IDs and cluster variables as node labels.

```@example
using JunctionTrees, CairoMakie, GraphMakie, Graphs, Printf, MetaGraphs

nvars, cards, _, factors = JunctionTrees.read_uai_file("../problems/paskin/paskin.uai")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)
td = JunctionTrees.construct_td_graph(mrf, cards)

# Node labels
nodeids = vertices(td) |> collect |> x -> string.(x)
varscopes = map(vertex -> get_prop(td, vertex, :vars), vertices(td)) |> x -> string.(x)
nlabels = zip(nodeids, varscopes) |> x -> map( y -> y[1]*": "*y[2], x)

fig = Figure(backgroundcolor = :white, resolution=(900, 400))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(
  ax,
  td,
  layout = GraphMakie.NetworkLayout.Stress(seed=58),
  nlabels = nlabels,
  nlabels_align = (:center,:center),
  nlabels_color = :white,
  nlabels_textsize = 14,
  node_size = 100,
)

xlims!(-1.5, 1.5)
ylims!(-0.9, 1.4)

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
fig
```
######

#### Example 5

Plots a junction tree with cluster IDs and cluster variables as node labels.
The root cluster is denoted with a fisheye lens-like circle.

```@example
using JunctionTrees, CairoMakie, GraphMakie, Graphs, Printf, MetaGraphs

nvars, cards, _, factors = JunctionTrees.read_uai_file("../problems/paskin/paskin.uai")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)
td = JunctionTrees.construct_td_graph(mrf, cards)

# Node labels
nodeids = vertices(td) |> collect |> x -> string.(x)
varscopes = map(vertex -> get_prop(td, vertex, :vars), vertices(td)) |> x -> string.(x)
nlabels = zip(nodeids, varscopes) |> x -> map( y -> y[1]*": "*y[2], x)

# Root node shape
node_marker = repeat(['●'], nv(td))
node_marker[JunctionTrees.select_rootnode(td).id] = '◉'

fig = Figure(backgroundcolor = :white, resolution=(900, 500))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(
  ax,
  td,
  layout = GraphMakie.NetworkLayout.Stress(seed=58),
  nlabels = nlabels,
  nlabels_align = (:center,:center),
  nlabels_color = :white,
  nlabels_textsize = 14,
  node_size = 160,
  node_marker = node_marker,
)

xlims!(-1.5, 1.5)
ylims!(-0.9, 1.5)

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
fig
```
######

#### Example 6

Plots a junction tree with cluster IDs and cluster variables as node labels.
The root cluster is denoted with a fisheye lens-like circle. Clusters with one
or more observed variables are colored in red.

```@example
using JunctionTrees, CairoMakie, GraphMakie, Graphs, Printf, MetaGraphs, ColorSchemes

nvars, cards, _, factors = JunctionTrees.read_uai_file("../problems/paskin/paskin.uai")
obsvars, _ = JunctionTrees.read_uai_evid_file("../problems/paskin/paskin.uai.evid")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)
td = JunctionTrees.construct_td_graph(mrf, cards)

# Node labels
nodeids = vertices(td) |> collect |> x -> string.(x)
varscopes = map(vertex -> get_prop(td, vertex, :vars), vertices(td)) |> x -> string.(x)
nlabels = zip(nodeids, varscopes) |> x -> map( y -> y[1]*": "*y[2], x)

# Root node shape
node_marker = repeat(['●'], nv(td))
node_marker[JunctionTrees.select_rootnode(td).id] = '◉'

# Default node color
node_color = [ColorSchemes.watermelon[1] for i in 1:nv(td)]

# Observed node color
node_color[obsvars] .= ColorSchemes.watermelon[8]

fig = Figure(backgroundcolor = :white, resolution=(900, 500))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(
  ax,
  td,
  layout = GraphMakie.NetworkLayout.Stress(seed=58),
  nlabels = nlabels,
  nlabels_align = (:center,:center),
  nlabels_color = :white,
  nlabels_textsize = 14,
  node_size = 160,
  node_marker = node_marker,
  node_color = node_color,
)

xlims!(-1.5, 1.5)
ylims!(-0.9, 1.4)

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
fig
```
######

#### Example 7

Plots a junction tree with cluster IDs and cluster variables as node labels and
sepsets as edge labels. The root cluster is denoted with a fisheye lens-like
circle. Clusters with one or more observed variables are colored in red.

```@example
using JunctionTrees, CairoMakie, GraphMakie, Graphs, Printf, MetaGraphs, ColorSchemes

nvars, cards, _, factors = JunctionTrees.read_uai_file("../problems/paskin/paskin.uai")
obsvars, _ = JunctionTrees.read_uai_evid_file("../problems/paskin/paskin.uai.evid")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)
td = JunctionTrees.construct_td_graph(mrf, cards)

# Node labels
nodeids = vertices(td) |> collect |> x -> string.(x)
varscopes = map(vertex -> get_prop(td, vertex, :vars), vertices(td)) |> x -> string.(x)
nlabels = zip(nodeids, varscopes) |> x -> map( y -> y[1]*": "*y[2], x)

# Root node shape
node_marker = repeat(['●'], nv(td))
node_marker[JunctionTrees.select_rootnode(td).id] = '◉'

# Default node color
node_color = [ColorSchemes.watermelon[1] for i in 1:nv(td)]

# Observed node color
node_color[obsvars] .= ColorSchemes.watermelon[8]

fig = Figure(backgroundcolor = :white, resolution=(900, 500))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(
  ax,
  td,
  layout = GraphMakie.NetworkLayout.Stress(seed=58),
  nlabels = nlabels,
  nlabels_align = (:center,:center),
  nlabels_color = :white,
  nlabels_textsize = 14,
  node_size = 160,
  node_marker = node_marker,
  node_color = node_color,
  elabels = [repr(sepset) for sepset in map(edge -> get_prop(td, edge, :sepset), edges(td))],
  elabels_textsize = 14,
  elabels_align = (:center, :bottom),
  elabels_distance = 5.0,
)

xlims!(-1.5, 1.5)
ylims!(-0.9, 1.4)

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
fig
```
######

#### Example 8

Plots a junction tree based on problem 34 of the [Promedus
benchmark](https://github.com/PACE-challenge/UAI-2014-competition-graphs).

```@example
using JunctionTrees, CairoMakie, GraphMakie, Graphs, MetaGraphs, ColorSchemes

nvars, cards, _, factors = JunctionTrees.read_uai_file("../problems/Promedus_34/Promedus_34.uai")
obsvars, _ = JunctionTrees.read_uai_evid_file("../problems/Promedus_34/Promedus_34.uai.evid")
mrf = JunctionTrees.construct_mrf_graph(nvars, factors)
td = JunctionTrees.construct_td_graph(mrf, cards)

# Default node color
node_color = [ColorSchemes.watermelon[1] for i in 1:nv(td)]

# Root node color
node_color[JunctionTrees.select_rootnode(td).id] = ColorSchemes.watermelon[3]

# Observed node color
node_color[obsvars] .= ColorSchemes.watermelon[8]

fig = Figure(backgroundcolor = :white, resolution=(900, 600))
ax = fig[1,:] = Axis(fig, backgroundcolor = :white)

graphplot!(ax, td, layout = GraphMakie.NetworkLayout.Stress(seed=1), node_color = node_color)

hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
fig
```
