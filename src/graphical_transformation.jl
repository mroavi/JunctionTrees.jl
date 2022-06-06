"""
$(TYPEDSIGNATURES)

Add a vertex for each bag and initialize its properties.
"""
function add_vertices!(g, bags)

  # Add each bag to the graph and initialize its properties
  for (bag_id, bag) in enumerate(bags)
    # Add its corresponding vertex to the graph
    add_vertex!(g)
    # Store the bag's variables in a property
    set_prop!(g, bag_id, :vars, bag)
  end

  # # DEBUG: print each bag's vars
  # map(x -> string(x,": ",get_prop(g,x,:vars)), vertices(g)) |> x -> show(stdout, "text/plain", x)

end

"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

Construct a tree decomposition graph based on `td_filepath`.

The `td_filepath` file format is defined in:
https://pacechallenge.org/2017/treewidth/.

# Example
```
td_filepath = "../problems/Promedus_26/Promedus_26.td"
td = compile_algo(td_filepath)
```
"""
function construct_td_graph(td_filepath::AbstractString)

  _, _, _, bags, edges = read_td_file(td_filepath)

  td = MetaGraph()
  add_vertices!(td, bags)
  add_edges!(td, edges)

  return td

end

# ------------------------------------------------------------------------------
# Triangulation
# ------------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Compute the number of edges to be added to the graph if we choose to eliminate
this vertex.
# Arguments
- `g::MetaGraph` the graph to consider.
- `v::Int64` the vertex to eliminate.
# Return
- `count::Int64` number of edges to add in the graph g if v is eliminated.
"""
function count_edges_to_be_added(g::MetaGraph, v::Int64)::Int64
  count = 0
  neighbor_vars = neighbors(g, v)
  neighbor_combinations = combinations(neighbor_vars, 2) # get all combinations of pairs of neighbors
  for (v1,v2) in neighbor_combinations
    !has_edge(g, v1, v2) && (count+=1) # increment counter if no edge
  end
  return count
end

"""
$(TYPEDSIGNATURES)

Compute the product of the cardinalities of `bag` constituent vars.
"""
function weight(g, bag::Vector{Int64}, cards)::Int64 
  map(v -> cards[v], bag) |> prod
end

"""
$(TYPEDSIGNATURES)

Return the bag induced after removing `v`, i.e. the set of vars consisting of
`v` and its neighbors.
"""
get_induced_bag(g::MetaGraph, v::Int64) = [v, neighbors(g, v)...] |> sort

"""
$(TYPEDSIGNATURES)

Compute the weight of the bag consisting of v and its neighbors.
"""
get_induced_bag_weight(g::MetaGraph, v::Int64, cards) = weight(g, get_induced_bag(g, v), cards)

"""
$(TYPEDSIGNATURES)

Calculate the "key" of `v` based on:
  1. The number of edges to be added if `v`'s neighbors were to be connected.
  2. The weight of `v` and its neighbors (also known as the induced bag).
The number of edges to be added has more priority than the weight of the
induced cluster.
The lower the number of edges to be added, the lower the key.
The lower the weight, the lower the key.
"""
function calculate_var_key(g::MetaGraph, v::Int64, cards)::Tuple{Int64, Int64}
  primary_key = count_edges_to_be_added(g, v)
  secondary_key = get_induced_bag_weight(g, v, cards)
  return (primary_key, secondary_key)
end

"""
$(TYPEDSIGNATURES)

Return whether `clique` is maximal in the `clique_set`
"""
function is_maximal(candidate_clique, cliques)
  for clique in cliques
    issubset(candidate_clique, clique) && return false
  end
  return true
end

"""
$(TYPEDSIGNATURES)

Return the maximal cliques of `g` using the minfill heuristic. The maximal
cliques of the triangulated graph correspond to the bags of the junction tree.
This implementation is based on: "Inference in Belief Networks: A Procedural
Guide" by Cecil Huang and Adnan Darwiche (1996) pg. 235.
TODO: See https://matbesancon.xyz/post/2019-05-30-vertex-safe-removal/
to implement this function using a vertex-safe version of Graphs.jl. This
would avoid having to bookkeep the node ids in the original graph after
it is modified with `rev_vertex!`.

Bookkepping example for the `paskin-example` problem:

    -----------------------------------------------------------
    | var to remove | vertices(g) | int2ext     | ext2int     |
    -----------------------------------------------------------
    | 6             | 1 2 3 4 5 6 | 1 2 3 4 5 6 | 1 2 3 4 5 6 |
    | 4             | 1 2 3 4 5   | 1 2 3 4 5   | 1 2 3 4 5 0 |
    | 5             | 1 2 3 4     | 1 2 3 5     | 1 2 3 0 4 0 |
    | 2             | 1 2 3       | 1 2 3       | 1 2 3 0 0 0 |
    | 1             | 1 2         | 1 3         | 1 0 2 0 0 0 |
    | 3             | 1           | 3           | 0 0 1 0 0 0 |
    -----------------------------------------------------------

Variables preceded with `_` use the internal variable indexation. Those without
use the external/original variable indexation.
"""
function form_bags(g, cards)

  # Vector to be returned with the maximal cliques (or bags) of `g`
  maximal_cliques = Vector{Vector{Int64}}()

  # Make a deep copy of g
  g2 = deepcopy(g)

  # Create the named tuples to be inserted into the binary heap
  vars = map(v -> (var=v, key=calculate_var_key(g2, v, cards)), vertices(g2))

  # Create a min-binary heap with the `key` element of the named tuple as key
  heap = MutableBinaryHeap(Base.By(x -> x.key, Base.Forward), vars)

  # Keeps track of node ids in the original graph after it is modified with `rem_vertex!`
  int2ext = vertices(g2) |> collect # maps internal vertex indexation to the external/original one
  ext2int = vertices(g2) |> collect # maps external vertex indexation to the internal used by Graphs.jl

  # While `g2` has nodes
  while nv(g2) > 0

    # # DEBUG: Plot the graph
    # plotgraph(g2)

    # # # DEBUG: Plot the graph
    # plotgraph(g2, nlabels=[@sprintf "%i" int2ext[v] for v in vertices(g2)])

    # Get the next variable to be removed from the binary heap according to the `key`
    var_to_remove = pop!(heap).var

    # Connect all neighbors of the var to be removed with each other
    _neighbor_vars = neighbors(g2, ext2int[var_to_remove]) |> copy
    _neighbor_combinations = combinations(_neighbor_vars, 2) # get all combinations of pairs of neighbors
    for (_v1,_v2) in _neighbor_combinations
      add_edge!(g2, _v1, _v2)
    end

    # Convert the internal var representatin to the external/original one for each neighbor
    neighbor_vars = map(neighbor_var -> int2ext[neighbor_var], _neighbor_vars)

    # The var to be removed together with its neighbors form a maximal clique
    # Push it to the array of maximal cliques to be returned if it is maximal
    clique = [var_to_remove, neighbor_vars...] |> sort
    is_maximal(clique, maximal_cliques) && push!(maximal_cliques, clique)

    # Bookkepping
    # @show int2ext
    # @show ext2int
    idx_var_to_remove = ext2int[var_to_remove]
    int2ext[idx_var_to_remove] = int2ext[end]
    ext2int[int2ext[end]] = idx_var_to_remove
    ext2int[var_to_remove] = 0
    pop!(int2ext)

    # Remove vertex (automatically removes the edges connected to it)
    rem_vertex!(g2, idx_var_to_remove)

    # Update the key of the neighbors of the var to be removed
    map(v -> update!(heap, v, (var=v, key=calculate_var_key(g2, ext2int[v], cards))), neighbor_vars)

  end

  return maximal_cliques

end

# ------------------------------------------------------------------------------
# Connection of bags 
# ------------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Return the number of vars in the sepset.
"""
function mass(g::MetaGraph, sepset::Vector{Int64})::Int64 
  return length(sepset)
end

"""
$(TYPEDSIGNATURES)

Compute the product of the cardinalities of `bag` constituent vars.
"""
function cost(g::MetaGraph, sepset::Vector{Int64}, cards)::Int64 
  isempty(sepset) && return 0
  return map(v -> cards[v], sepset) |> prod
end

"""
$(TYPEDSIGNATURES)

Calculate the "key" of `sepset` based on:
  1. Mass: The number of variables in `sepset`.
  2. Cost: The product of the cardinality of each variable in `sepset`.
The number of variables in the sepset has higher priority than the product of
their cardinality.
The higher the mass, the lower the key.
The lower the cost, the lower the key.
"""
function calculate_sepset_key(g::MetaGraph, sepset::Vector{Int64}, cards)::Tuple{Int64, Int64}
  primary_key = -mass(g, sepset) 
  secondary_key = cost(g, sepset, cards)
  return (primary_key, secondary_key)
end

"""
$(TYPEDSIGNATURES)

Calculate the key for each element that will be inserted into the binary heap
and wraps the edge, its sepset and the key into a tuple.
"""
function preprocess_heap_elements(g::MetaGraph, bag1, bag2, cards)
  endpoints=(bag1[1], bag2[1])
  sepset = bag1[2] ∩ bag2[2]
  key = calculate_sepset_key(g, sepset, cards)
  (endpoints=endpoints, sepset=sepset, key=key)
end


"""
$(TYPEDSIGNATURES)

Return whether vertices `v1` and `v2` are connected.
"""
function are_connected(g, v1, v2)
  # Mark all the vertices as not visited
  visited = falses(nv(g))
  # Create a queue for BFS
  q = Queue{Int}()
  # Mark the root vertex as visited and enqueue it
  visited[v1] = true
  enqueue!(q, v1)
  # While the queue is not empty
  while !isempty(q)
    # Get the current vertex
    v = dequeue!(q)
    # If `v2` is being visited, then stop and return true
    v == v2 && return true
    # Get the neighbors of the current vertex
    ns = neighbors(g, v)
    # For each neighbor
    for n in ns
      # Check whether the current neighbor has been visited
      if visited[n] == false
        # No, it hasn't. Then mark it as visited and enqueue it
        visited[n] = true
        enqueue!(q, n)
      end
    end
  end
  return false
end

"""
$(TYPEDSIGNATURES)

Connect the bags such that the running intersection property is satisfied.
"""
function connect_bags!(td::MetaGraph, mrf::MetaGraph, bags::Vector{Vector{Int64}}, cards)

  # Create the sepset elements that will be inserted into the binary heap
  bags_enumerated = enumerate(bags) |> collect
  bag_combinations = combinations(bags_enumerated, 2) 
  sepsets = map(bags -> preprocess_heap_elements(mrf, bags[1], bags[2], cards), bag_combinations)

  # Create a min-binary heap with the `key` element of the named tuple as key
  heap = MutableBinaryHeap(Base.By(x -> x.key, Base.Forward), sepsets)

  # Create an array to temporarily hold edges that cannot be added to the junction tree
  temp = Vector{eltype(heap)}()

  # While the junction tree has less than the number of bags - 1 edges
  while ne(td) < (length(bags) - 1)

    # Get the next variable to be removed from the binary heap according to the `key`
    edge = pop!(heap)

    # Are the endpoints of the edge contained in different components?
    if are_connected(td, edge.endpoints...) == false

      # No, then add the edge and its sepset to the junction tree
      add_edge!(td, Edge(edge.endpoints))
      set_prop!(td, Edge(edge.endpoints), :sepset, edge.sepset)

      # # DEBUG:
      # @show edge

      # Add all values in `temp` back to the heap
      while !isempty(temp)
        pop!(temp) |> edge -> push!(heap, edge)
      end

    else
      # Yes, then temprarilly store the edge in an array
      push!(temp, edge)
    end

  end

  # # DEBUG:
  # @show temp

end

"""
$(TYPEDSIGNATURES)

Transform the given Markov random field into a junction tree.
This implementation is based on "Inference in Belief Networks: A Procedural
Guide" by Cecil Huang and Adnan Darwiche (1996) pg. 235.
"""
function construct_td_graph(mrf, cards)

  bags = form_bags(mrf, cards)

  # Construct a MetaGraph for the junction tree
  td = MetaGraph(length(bags))

  # Connect the bags obtained above in an optimal manner to form a junction tree
  connect_bags!(td, mrf, bags, cards)

  # Store the variables each bag depends on in a property
  for (i, bag) in enumerate(bags)
    set_prop!(td, i, :vars, bag)
  end

  # # Store the number of variables in the junction tree graph itself
  # get_prop(mrf, :nvars) |> x -> set_prop!(td, :nvars, x)

  return td

end
