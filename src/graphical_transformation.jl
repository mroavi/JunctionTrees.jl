"""
    add_vertices!(g, bags)

Construct the bags and initialize its properties.

"""
function add_vertices!(g, bags)

  # Add each bag to the graph and initialize its properties
  for (bag_id, bag) in enumerate(bags)

    # Add its corresponding vertex to the graph
    add_vertex!(g)

    # Store the bag's variables in a property
    set_prop!(g, bag_id, :vars, bag)

    # Create an empty vector of factors
    set_prop!(g, bag_id, :factors, Factor{Float64}[])

    # Create an empty vector of expressions of incoming messages
    set_prop!(g, bag_id, :in_msgs, Union{Expr, Factor{Float64}}[])

  end

  # # DEBUG: print each bag's vars
  # map(x -> string(x,": ",get_prop(g,x,:vars)), vertices(g)) |> x -> show(stdout, "text/plain", x)

end

"""
    add_edges!(g, edges)

Construct the edges and store their sepset as an edge property.

"""
function add_edges!(g, edges)

  # Add each edge to the graph and store the intersection of vars between the bags it connects
  for (edge_src, edge_dst) in edges

    # Parse and add the edge to the graph
    add_edge!(g, edge_src, edge_dst)

    # Calculate the sepset and set it as an edge's property
    vars_src = get_prop(g, edge_src, :vars)
    vars_dst = get_prop(g, edge_dst, :vars)
    sepset = intersect(vars_src, vars_dst)
    set_prop!(g, Edge(edge_src, edge_dst), :sepset, sepset)

  end

  # # DEBUG: display sepsets
  # println("\nSepset of each edge:")
  # map(edge -> get_prop(g, edge, :sepset), Graphs.edges(g)) |> display # sepset of each edge

  # # DEBUG: display empty sepsets
  # map(edge -> (edge, get_prop(g, edge, :sepset)), Graphs.edges(g)) |> x -> filter(y -> isempty(y[2]), x) |> display

end

"""
    construct_td_graph(td_filepath)

Construct a tree decomposition graph based on `td_filepath`.

The `td_filepath` file format is defined in:
https://pacechallenge.org/2017/treewidth/.

# Example
```
td_filepath = "../problems/Promedus_26/Promedus_26.td"
g = computeMarginalsExpr(td_filepath)
```
"""
function construct_td_graph(td_filepath::String)

  global g = MetaGraph()
  nbags, treewidth, nvertices, bags, edges = read_td_file(td_filepath)
  add_vertices!(g, bags)
  add_edges!(g, edges)

  return g

end

