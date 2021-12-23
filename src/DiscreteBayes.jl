module DiscreteBayes

import Base:
  eltype,
  IteratorEltype,
  parent

export computeMarginalsExpr,
  Factor,
  product,
  marg,
  redu,
  norm,
  getGraph

export LastStage, ForwardPass, BackwardPass, JointMarginals, UnnormalizedMarginals, Marginals

using Graphs, MetaGraphs, AbstractTrees, MacroTools

include("factors.jl")
include("tree.jl")

getGraph() = @isdefined(g) ? g : nothing

"""
    constructTreeDecompositionAbstractTree!(g::MetaGraph, root::Node, parent::Node=root)

Construct a tree decomposition abstract tree based on the graph `g` using
`root` as the root node.

# Example
```
using LightGraphs

g = double_binary_tree(3)
root = Node(1)
constructTreeDecompositionAbstractTree!(g, root)
print_tree(root)
```

"""
function constructTreeDecompositionAbstractTree!(g::MetaGraph, root::Node, parent::Node=root)

  # Is the parent node the root?
  if root === parent
    # Yes, then include all parent's neighbors in the children set
    children_ids = neighbors(g, parent.id)
  else
    # No, then include all parent's neighbors except its parent in the children set
    grandparent = Base.parent(root, parent)
    children_ids = neighbors(g, parent.id) |> x -> setdiff(x, grandparent.id)
  end

  # Base case
  length(children_ids) == 0 && return

  children = addchildren!(children_ids, parent)
  map(child -> constructTreeDecompositionAbstractTree!(g, root, child), children)

  return

end

@enum PartialEval EvalMsg DoNotEvalMsg EvalProductOnly

# TODO: If an observed var is present in all clusters then it will not get
# marginalized in any message computation and hence a reduction statement will
# never be introduced for it.
"""
    partial_eval_prop_update!(g, curr_node, prev_node, obs_bag_var, obs_var_marginalized)

Recursive function that invalidates messages in the graph `g` that cannot be
partially evaluated at compile time. This marking is done using MetaGraphs'
properties for each edge in the graph.

"""
function partial_eval_prop_update!(g, curr_node, prev_node, obs_bag_var, obs_var_marginalized)
  # Get the neighbors except the previous node
  next_nodes = neighbors(g, curr_node) |> x -> setdiff(x, prev_node)
  # Base case
  isempty(next_nodes) && return
  # Check messages that go from current node to each node in `next_nodes`
  for next_node in next_nodes
    # Construct property name
    prop_name = Symbol("pre_eval_msg_", curr_node, "_", next_node)
    # Has the observed variable been marginalized already?
    if obs_var_marginalized
      # Yes, then invalidate partial evaluation for this message
      set_prop!(g, curr_node, next_node, prop_name, DoNotEvalMsg)
      # And call this function recursively with `true` as arg for `obs_var_marginalized`
      partial_eval_prop_update!(g, next_node, curr_node, obs_bag_var, true)
    else
      # No, then first mark the current node as not consistent so marginals are not extracted from it
      set_prop!(g, curr_node, :isconsistent, false)
      # Then check whether the obs var is marginalized in the current msg
      bag_vars = get_prop(g, curr_node, :vars)
      edge_sepset = get_prop(g, curr_node, next_node, :sepset)
      mar_vars = setdiff(bag_vars, edge_sepset)
      # Is the obs var marginalized in the current message?
      if obs_bag_var in mar_vars 
        # Yes, and hasn't the current message been invalidated by another obs node?
        partial_eval = get_prop(g, curr_node, next_node, prop_name)
        if partial_eval != DoNotEvalMsg
          # True, then only partially evaluate the product of the current message
          set_prop!(g, curr_node, next_node, prop_name, EvalProductOnly)
        end
        # Save the marginalized obs var together with the msg dir (edge dir) as a prop of the current edge
        # FIXME: support several mar obs vars per edge by appending obs_bag_var to the var field in the tuple
        set_prop!(g, curr_node, next_node, :mar_obs_vars, (src=curr_node, dst=next_node, vars=Tuple(obs_bag_var))) 
        # Call this function recursively passing passing `true` as arg for `obs_var_marginalized`
        partial_eval_prop_update!(g, next_node, curr_node, obs_bag_var, true)
      else
        # No, then mark the edge between the current node and the next node as
        # not consistent so marginals are not extracted from its sepset
        set_prop!(g, curr_node, next_node, :isconsistent, false)
        # Then call this function recursively passing passing `false` as arg for `obs_var_marginalized`
        partial_eval_prop_update!(g, next_node, curr_node, obs_bag_var, false)
      end
    end
  end
end

"""
    partial_eval_analysis!(g)

Analyzes which messages can be computed during the compilation stage in the
graph `g`.

"""
function partial_eval_analysis!(g)
  # Initialize each edge's `pre_eval_msg_` property with `EvalMsg` for all messages (two per edge)
  for edge in edges(g)
    prop_name = Symbol("pre_eval_msg_", edge.src, "_", edge.dst)
    set_prop!(g, edge.src, edge.dst, prop_name, EvalMsg)
    prop_name = Symbol("pre_eval_msg_", edge.dst, "_", edge.src)
    set_prop!(g, edge.src, edge.dst, prop_name, EvalMsg)
  end
  # Initialize each bag and edge with a consistency flag set to `true`
  # This flag specifies whether marginals can be extracted from a bag/edge after the evidence has been entered
  map(bag -> set_prop!(g, bag, :isconsistent, true), vertices(g))
  map(edge -> set_prop!(g, edge, :isconsistent, true), edges(g))
  # Get a list of the bags that have one or more observed vars
  obs_bags = filter_vertices(g, :obsvars)
  # For each observed var in a bag, traverse the graph from the current bag to all 
  # other bags and analyze whether the msg on each edge can be partially evaluated or not
  # TODO: account for the fact that empty sepsets stops the influence of observed vars
  for obs_bag in obs_bags
    # Get the observed vars in the current bag
    obs_bag_vars = get_prop(g, obs_bag, :obsvars)
    # Start the marking process for each observed var (this handles the case of several obs vars per bag)
    for obs_bag_var in obs_bag_vars
      partial_eval_prop_update!(g, obs_bag, [], obs_bag_var, false)
    end
  end
end

"""
    partially_evaluate(g, before_pass_msgs)

Partially evaluate messages that do not depend on observed variables.

"""
function partially_evaluate(g, before_pass_msgs)

  after_pass_msgs = quote end |> rmlines

  for before_pass_msg in before_pass_msgs.args
    if @capture(before_pass_msg, var_ = f_(fargs__)) # parse the current msg (Note: this filters the line number nodes)
      src, dst = split(string(var), "_") |> x -> x[2:3] |> x -> parse.(Int, x) # get msg src and dst
      prop_name = Symbol("pre_eval_msg_", src, "_", dst) # construct property name
      pe_msg_prop = get_prop(g, Edge(src, dst), prop_name) # get the property from graph
      # Evaluate the entire message?
      if pe_msg_prop == EvalMsg
        # Yes, then eval the message and add the resulting factor to the new expr
        msg_evaled = eval(before_pass_msg)
        after_pass_msg = :($var = $msg_evaled) # create assignment expr with evaled msg
      # No, then evaluate the product contained in the message?
      elseif pe_msg_prop == EvalProductOnly
        # Is the first argument of the msg a product operation?
        if @capture(fargs[1], product(pargs__))
          # Yes, then evaluate it, wrap it in the marg operation and add the msg to the new expr
          prod_evaled = :(product($(pargs...))) |> eval # eval the product 
          after_pass_msg = :($var = $f($prod_evaled, $(fargs[2:end]...))) # wrap the evaled prod in the msg
        else
          # No, then do not modify the current message
          after_pass_msg = before_pass_msg
        end
      # No, then do not evaluate anything?
      elseif pe_msg_prop == DoNotEvalMsg
        # Yes, then do not modify the current message
        after_pass_msg = before_pass_msg
      end
      # Push the msg to the new expr
      push!(after_pass_msgs.args, after_pass_msg)
      # DEBUG
      # println("Before PE: ", before_pass_msg, "\n", "After  PE: ", after_pass_msg, "\n")
    end
  end
  return after_pass_msgs
end

"""
    inject_redus_in_msgs(g, before_pass_msgs, obsvars, obsvals)

Inject a reduction statement for each observed variable. Each reduction takes
the observed variable and it's corresponding observed value. The reduction
statements are introduced as late as possible, i.e. just before the observed
variable is marginalized.

"""
function inject_redus_in_msgs(g, before_pass_msgs, obsvars, obsvals)

  after_pass_msgs = quote end |> rmlines

  # Get the messages where one or more observed vars are marginalized
  mar_obs_msgs = filter_edges(g, :mar_obs_vars) |> # get a list of edges over which msgs that marginalize obs var pass
    mar_obs_edges -> map(x -> get_prop(g, x, :mar_obs_vars), mar_obs_edges) |> # get the property value
    mar_obs_msgs -> map(x -> Edge(x.src, x.dst), mar_obs_msgs) # create and store edges with msg direction info

  # # DEBUG: display the `:mar_obs_var` property for those edges that have it
  # filter_edges(g, :mar_obs_vars) |> mar_obs_edges -> map(x -> get_prop(g, x, :mar_obs_vars), mar_obs_edges) |> display

  for before_pass_msg in before_pass_msgs.args
    if @capture(before_pass_msg, var_ = f_(fargs__)) # parse the current msg (Note: this filters the line number nodes)
      src, dst = split(string(var), "_") |> x -> x[2:3] |> x -> parse.(Int, x) # get msg src and dst
      # Is the current msg in the set of msgs that marginalize one or more observed vars?
      # AND is the sepset of the edge through which this msg passes not empty?
      msg = Edge(src, dst) # create a directed edge that represents the current message
      if (msg in mar_obs_msgs) && (get_prop(g, msg, :sepset) |> !isempty)
        # Yes, then add a redu statement right before the marginalization
        mar_obs_vars = get_prop(g, msg, :mar_obs_vars) |> x -> x.vars # get the marginalized observed vars
        indx_mar_obs_vars = indexin(mar_obs_vars, obsvars) # find index of each elem in mar_obs_vars in the obsvars array
        mar_obs_vals = map(i -> :(obsvals[$i]), indx_mar_obs_vars) |> Tuple
        redu_expr = :(redu($(fargs[1]), $(mar_obs_vars), ($(mar_obs_vals...),))) # wrap the evaled prod in the redu expr
        after_pass_msg = :($var = $f($redu_expr, $(fargs[2:end]...))) # wrap the redu expr in the msg
      else
        # No, then do not add a redu statement
        after_pass_msg = before_pass_msg
      end
    else
      # The current msg is an evaled factor expr. Add it unmodified to the resulting expr arr
      after_pass_msg = before_pass_msg
    end
    push!(after_pass_msgs.args, after_pass_msg)
  end

  return after_pass_msgs

end

"""
    inject_redus_in_pots(g, before_pass_pots, obsvars, obsvals)

Inject a reduction statement for observed variables contained inside isolated
bags. An isolated bag is a leaf bag connected to the rest of the tree via one
empty sepset. Each reduction takes the observed variable and it's corresponding
observed value. 

"""
function inject_redus_in_pots(g, before_pass_pots, obsvars, obsvals)
  after_pass_pots = quote end |> rmlines
  for before_pass_pot in before_pass_pots.args
    if @capture(before_pass_pot, var_ = factor_)
      bag = split(string(var), "_") |> x -> x[2] |> x -> parse(Int, x) # get the bag id from the expr
      # Does the current bag contain an observed variable?
      # AND is the current bag isolated? (i.e. a leaf node connected to the rest of the tree via one empty sepset)
      bag_neighbors = neighbors(g, bag)
      if has_prop(g, bag, :obsvars) && length(bag_neighbors) == 1 && (get_prop(g, bag, bag_neighbors[1], :sepset) |> isempty)
        # Yes, then allow marginals to be extracted from this isolated bag
        set_prop!(g, bag, :isconsistent, true)
        # Add a redu statement to the current expression
        bag_obsvars = get_prop(g, bag, :obsvars) |> Tuple
        indx_bag_obsvars = indexin(bag_obsvars, obsvars) # find index of each elem in mar_obsvars in the obsvars array
        bag_obsvals = map(i -> :(obsvals[$i]), indx_bag_obsvars) |> Tuple
        redu_expr = :(redu($(factor), $(bag_obsvars), ($(bag_obsvals...),))) # wrap the evaled prod in the redu expr
        after_pass_pot = :($var = $redu_expr) # wrap the redu expr in the msg
      else
        # No, then do not add a redu statement
        after_pass_pot = before_pass_pot
      end
      push!(after_pass_pots.args, after_pass_pot)
    end
  end
  return after_pass_pots
end

"""
    inject_redus(g, before_pass_pots, obsvars, obsvals)

Inject a reduction expression to potentials that contain observed variables.

"""
function inject_redus(g, before_pass_pots, obsvars, obsvals)
  after_pass_pots = quote end |> rmlines
  for before_pass_pot in before_pass_pots.args
    if @capture(before_pass_pot, var_ = factor_)
      bag = split(string(var), "_") |> x -> x[2] |> x -> parse(Int, x) # get the bag id from the expr
      # Does the current bag contain an observed variable?
      if has_prop(g, bag, :obsvars)
        # Yes, then reduce the potential based on the observed vars and values
        bag_obsvars = get_prop(g, bag, :obsvars) |> Tuple
        indx_bag_obsvars = indexin(bag_obsvars, obsvars) # find index of each elem in mar_obsvars in the obsvars array
        bag_obsvals = map(i -> :(obsvals[$i]), indx_bag_obsvars) |> Tuple
        redu_expr = :(redu($(factor), $(bag_obsvars), ($(bag_obsvals...),))) # wrap the evaled prod in the redu expr
        after_pass_pot = :($var = $redu_expr) # wrap the redu expr in the msg
      else
        # No, then do not add a redu statement
        after_pass_pot = before_pass_pot
      end
      push!(after_pass_pots.args, after_pass_pot)
    end
  end
  return after_pass_pots
end

"""
    generate_function_expression(function_name, sig, variables, body)

Generates a function expression using Julia's metaprogramming capabilities

# Example
```
function_name = :foo
sig = (Int, Float64, Int32)
variables = [:a, :q, :d]
body = :(a + q * d)

ex = generate_function_expression(function_name, sig, variables, body)

eval(ex)

foo(1, 2.0, Int32(3))
```
"""
function generate_function_expression(function_name, sig, variables, body)
	Expr(:function, 
		Expr(:call,
			function_name,
			[Expr(:(::), s, t) for (s, t) in zip(variables, sig)]...),
		body
	)
end

"""
    read_td_file(td_filepath)

Read the td file.

"""
function read_td_file(td_filepath)

  # Read the td file into an array of lines
  rawlines = open(td_filepath) do file
    readlines(file)
  end

  # Filter out comments
  lines = filter(x -> !startswith(x, "c"), rawlines)

  # Extract number of bags, treewidth+1 and number of vertices from solution line
  nbags, treewidth, nvertices = split(lines[1]) |> soln_line->soln_line[3:5] |> x->parse.(Int, x)

  # # DEBUG
  # @show nbags, treewidth, nvertices

  return lines, nbags, treewidth, nvertices

end

"""
  read_evid_file(uai_evid_filepath)

Read the uai evid file if the passed file name is not an empty string.

"""
function read_evid_file(uai_evid_filepath)

  if !isempty(uai_evid_filepath)

    # Read the uai evid file into an array of lines
    line = open(uai_evid_filepath) do file
      readlines(file)
    end

    @assert length(line) == 1

    # Extract number of observed vars, and their id together with their corresponding value
    nobsvars, rest = split(line[1]) |> x -> parse.(Int, x) |> x -> (x[1], x[2:end])
    observations = reshape(rest, 2, :)

    # Convert to 1-based indexing
    obsvars = observations[1,:] .+ 1
    obsvals = observations[2,:] .+ 1

    @assert nobsvars == length(obsvars)
  else
    # No evidence
    obsvars = []
    obsvals = []
  end

  # # DEBUG
  # print("  "); @show obsvars
  # print("  "); @show obsvals

  return obsvars, obsvals

end

"""
    add_vertices!(g, lines, nbags, obsvars)

Construct the bags

"""
function add_vertices!(g, lines, nbags, obsvars)

  # Extract bag definition lines
  bag_lines = lines[2:(2+nbags-1)]

  # Add each bag to the graph and initialize its properties
  for bag in bag_lines

    bag_arr = split(bag)

    @assert bag_arr[1] == "b"

    # Add its corresponding vertex to the graph
    add_vertex!(g)

    # Verify that the bag id (second element in line) is the same as the index of last added vertex
    bag_id = bag_arr[2] |> x -> parse(Int, x)
    @assert nv(g) == bag_id

    # Store the variables the bag (cluster) depends on in a property
    bag_vertices = bag_arr[3:end] |> x -> parse.(Int, x)
    set_prop!(g, bag_id, :vars, bag_vertices)

    # If any, store the bag's observed vars in a property
    bag_obsvars = intersect(bag_vertices, obsvars)
    !isempty(bag_obsvars) && set_prop!(g, bag_id, :obsvars, bag_obsvars)

    # Create an empty vector of factors
    set_prop!(g, bag_id, :factors, Factor{Float64}[])

    # Create an empty vector of expressions of incoming messages
    set_prop!(g, bag_id, :in_msgs, Union{Expr, Factor{Float64}}[])

  end

  # # DEBUG
  # observations # observed vars with their corresponding value (vars on first row, vals on second row)
  # map(x -> string(x,": ",get_prop(g,x,:vars)), vertices(g)) |> x -> show(stdout, "text/plain", x)
  # @show filter_vertices(g, :obsvars) |> collect # bags that contain at least one observed var

end

"""
    add_edges!(g, lines)

Construct the edges.

"""
function add_edges!(g, lines)

  # Extract edge definition lines
  edge_lines = lines[(2+nv(g)):end]

  # Add each edge to the graph and store the intersection of vars between the bags it connects
  for edge in edge_lines

    # Parse and add the edge to the graph
    edge_src, edge_dst = split(edge) |> x -> parse.(Int, x)
    add_edge!(g, edge_src, edge_dst)

    # Calculate the sepset and set it as an edge's property
    vars_src = get_prop(g, edge_src, :vars)
    vars_dst = get_prop(g, edge_dst, :vars)
    sepset = intersect(vars_src, vars_dst)
    set_prop!(g, Edge(edge_src, edge_dst), :sepset, sepset)

  end

  # # DEBUG
  # println("\nSepset of each edge:")
  # map(edge -> get_prop(g, edge, :sepset), edges(g)) |> display # sepset of each edge

  # # DEBUG: display empty sepsets
  # map(edge -> (edge, get_prop(g, edge, :sepset)), edges(g)) |> x -> filter(y -> isempty(y[2]), x) |> display

end

"""
    mark_leaves!(g)

Mark which nodes of the graph correspond to leaves using a property.

"""
function mark_leaves!(g)

  map(x -> length(neighbors(g, x)), vertices(g)) |>   # number of neighbors for each bag
    x -> findall(isone, x) |>                         # indices of the leaves
    x -> map(y -> set_prop!(g, y, :isleaf, true), x)  # set a property for the leaf bags

  # # DEBUG
  # println("\nLeaf bags:")
  # filter_vertices(g, :isleaf) |> collect |> display

end

"""
    construct_td_graph(td_filepath, uai_filepath, uai_evid_filepath = "")

Construct a tree decomposition graph based on `td_filepath`.
Mark the observed variables according to `uai_evid_filepath`.

The `td_filepath` file format is defined in:
https://pacechallenge.org/2017/treewidth/.

The `uai_evid_filepath` file format is defined in :
http://www.hlt.utdallas.edu/~vgogate/uai14-competition/evidformat.html

# Example
```
td_filepath       = "../problems/Promedus_26/Promedus_26.td"
uai_evid_filepath = "../problems/Promedus_26/Promedus_26.evid"
g = computeMarginalsExpr(td_filepath, uai_evid_filepath)
```
"""
function construct_td_graph(td_filepath, uai_evid_filepath = "")

  global g = MetaGraph()
  lines, nbags, treewidth, nvertices = read_td_file(td_filepath)
  obsvars, obsvals = read_evid_file(uai_evid_filepath)
  add_vertices!(g, lines, nbags, obsvars)
  add_edges!(g, lines)
  # mark_leaves!(g)

  return g, obsvars, obsvals

end


"""
    read_uai_file(uai_filepath)

Read the factors from the UAI file.

The `uai_filepath` file format is defined in:
http://www.hlt.utdallas.edu/~vgogate/uai14-competition/modelformat.html

"""
function read_uai_file(uai_filepath)

  # Read the uai file into an array of lines
  rawlines = open(uai_filepath) do file
    readlines(file)
  end

  # Filter out empty lines
  lines = filter(!isempty, rawlines)

  nvars   = lines[2] |> x -> parse.(Int, x)
  card    = lines[3] |> split |> x -> parse.(Int, x)
  ntables = lines[4] |> x -> parse.(Int, x)

  # @assert nvars == nvertices

  scopes =
    lines[5:(5+ntables-1)] |>             # extract the factor scope definition lines
    x -> map(y -> split(y), x) |>         # split each line using blank space as delimeter
    x -> map(y -> map(z -> parse(Int, z), y), x) |> # parse each string element as an integer
    x -> map(y -> y[2:end], x) |>         # drop first element of each inner array
    x -> map(y -> map( z -> z +1, y), x) |> # convert to 1-based index
    x -> map(reverse, x)                  # order vars in ascending order (least significant first)

  tables1 =
    lines[(5+ntables):end] |>             # extract the probability tables definition lines
    x -> map(y -> y * " ", x) |>          # append a "space" to the end of each element
    x -> reduce(*, x) |>                  # concatenate all string elements
    x -> split(x)                         # split the array using blank space as delimeter

  tables2 = Array{Float64,1}[]

  let i = 1
    while i <= length(tables1)
      nelements = tables1[i] |> x -> parse(Int, x)
      tables1[i+1:i+nelements] |> x -> parse.(Float64, x) |> x -> push!(tables2, x)
      i += nelements + 1
    end
  end

  tables =
    zip(tables2, map(scope -> card[scope], scopes)) |> # pair each table with its card vector
    x -> map(y -> reshape(y[1], Tuple(y[2])), x) # reshape each factor according to its card

  # Sort scope vars in ascending order and permute table dims accordingly
  scopes_sorted = map(sort, scopes)
  tables_sorted = map(indexin, scopes_sorted, scopes) |> x -> map(permutedims, tables, x)

  # Wrap the tables with their corresponding scopes in an array of Factor type
  factors = [Factor{Float64,length(scope)}(Tuple(scope), table) for (scope, table) in zip(scopes_sorted, tables_sorted)]

  return factors, nvars

end


"""
    assign_factors!(g, factors, smart_root_selection)

Assign each factor to a cluster that covers its variables.

"""
function assign_factors!(g, factors, smart_root_selection)

  if smart_root_selection
    root = map(v -> get_prop(g, v, :vars) |> length, vertices(g)) |> findmax |> x -> Node(x[2])
  else
    root = Node(1) # arbitrarily chose Node 1 as root
  end

  # root = Node(339) # optimal for problem 14
  set_prop!(g, root.id, :isroot, true)

  # Construct an abstract tree using AbstractTrees.jl
  constructTreeDecompositionAbstractTree!(g, root)

  # # DEBUG
  # @show root.id
  # println("\nCluster tree:")
  # print_tree(root)

  # Traverse the tree in postorder and assign each factor to a bag (cluster) that
  # 1) covers its vars and 2) is closest to a leaf
  for factor in factors
    for bag in PostOrderDFS(root)
      if issubset(factor.vars, get_prop(g, bag.id, :vars))
        push!(get_prop(g, bag.id, :factors), factor)
        break
      end
    end
  end

  # # DEBUG
  # map(vertex -> get_prop(g, vertex, :vars), vertices(g)) # vars on which each bag depends on
  # map(vertex -> get_prop(g, vertex, :factors), vertices(g)) # factors assigned to each bag

  return root

end

"""
    compile_bag_potentials(g)

Compile each bag's potential into a Julia expression.

"""
function compile_bag_potentials(g)

  pots = quote end |> rmlines

  # For each bag
  for bag in vertices(g)
    # Are there any factors assigned to the current bag?
    bag_factors = get_prop(g, bag, :factors)
    if isempty(bag_factors)
      # No, then assign a "unit" potential to this bag
      pot = Factor{Float64,0}((), Array{Float64,0}(undef))
    else
      # Yes, then compute the product of all potentials assigned to the current bag
      pot = product(bag_factors...)
    end
    pot_var_name = Symbol("pot_", bag)
    push!(pots.args, :($pot_var_name = $pot))
  end

  # # DEBUG
  # map(vertex -> (vertex, get_prop(g, vertex, :pot)), vertices(g)) # potential of each bag

  # DEBUG
  # println(pots)
  # eval(pots)

  return pots
end

"""
    initialize_td_graph!(g, uai_filepath, smart_root_selection)

Initialize the td graph by assigning the different factors to one bag
that covers its scope.

"""
function initialize_td_graph!(g, uai_filepath, smart_root_selection)

  factors, nvars = read_uai_file(uai_filepath)
  root = assign_factors!(g, factors, smart_root_selection)
  pots = compile_bag_potentials(g)

  return root, pots, nvars

end


"""
    compile_forward_pass!(g, root)

Compile the upstream messages.

"""
function compile_forward_pass!(g, root)

  forward_pass = quote end |> rmlines

  # Visit each bag in postorder and compute its upstream message
  for bag in PostOrderDFS(root)

    # Get parent bag from abstract tree
    parent_bag = Base.parent(root, bag)

    # Exit loop if the current bag is the root (i.e. if the current bag has no parent)
    isnothing(parent_bag) && break

    # Compute the joint distribution of all incoming messages and the potential
    pot_var_name = Symbol("pot_", bag.id)
    # Is this bag a leaf?
    if isempty(bag.children)
      # Yes, then only extract the potential
      joint = pot_var_name
    else
      # No, then construct an expr of the factor product between incomming msgs and potential
      joint =
        map(child -> child.id, bag.children) |> # get the children ids of the current bag
        children_ids -> map(child_id -> get_prop(g, child_id, bag.id, :up_msg), children_ids) |>
        in_msgs -> map(in_msg -> in_msg.args[1], in_msgs) |> # get the msg variable name from Expr
        in_msgs_var_names -> vcat(in_msgs_var_names, pot_var_name) |> # concat in msgs and pot
        in_msgs_and_potential -> :(product($(in_msgs_and_potential...))) #splatting interpol
    end

    # Get the variables that need to be marginalized
    bag_vars = get_prop(g, bag.id, :vars)
    out_edge_sepset = get_prop(g, bag.id, parent_bag.id, :sepset)
    mar_vars = setdiff(bag_vars, out_edge_sepset)

    # Marginalize vars to compute the upstream message
    msg_var_name = Symbol("msg_", bag.id, "_", parent_bag.id)
    up_msg = :($msg_var_name = marg($joint, $(mar_vars...)))

    # Set the resulting factor, the upstream message, as an edge property
    set_prop!(g, bag.id, parent_bag.id, :up_msg, up_msg)

    # Store message in parent's bag incoming messages array
    push!(get_prop(g, parent_bag.id, :in_msgs), up_msg)

    # Push the current up message expression to the algo expression
    push!(forward_pass.args, up_msg)

    # # DEBUG
    # println(up_msg)

  end

  # # DEBUG
  # map(bag -> bag.id , PostOrderDFS(root)) |> 
  #   x -> println("\nForward pass visiting order: \n", x)

  # @show forward_pass
  # @time eval(forward_pass) 
  # @btime eval(forward_pass) 

  return forward_pass

end

"""
    compile_backward_pass!(g, root)

Compile the downstream messages.

"""
function compile_backward_pass!(g, root)

  backward_pass = quote end |> rmlines

  # Visit each bag in preorder and compute the messages through all edges other
  # than the one connecting it to the parent
  for bag in PreOrderDFS(root)

    # Get parent bag from abstract tree
    parent_node = Base.parent(root, bag)

    # Compute and send a message to each child
    for child in bag.children

      # Compute joint btwn the incoming msgs (coming from parent and siblings) and bag's potential
      pot_var_name = Symbol("pot_", bag.id)
      siblings = setdiff(bag.children, [child]) # get a list of the siblings (all other children)

      # Four possible scenarios:
      if isnothing(parent_node) && isempty(siblings)
        # current bag has no parent, current child has no sibling(s)
        joint = pot_var_name
      elseif !isnothing(parent_node) && isempty(siblings)
        # current bag has parent, current child has no sibling(s)
        parent_msg = get_prop(g, parent_node.id, bag.id, :down_msg) |> # get down msg from prop
          down_msg -> down_msg.args[1] # get the msg variable name
        joint = :(product($parent_msg, $pot_var_name))
      elseif isnothing(parent_node) && !isempty(siblings)
        # current bag has no parent, current child has sibling(s)
        joint =
          map(sibling -> get_prop(g, sibling.id, bag.id, :up_msg), siblings) |> # get sibling msgs
          sibling_msgs -> map(sibling_msg -> sibling_msg.args[1], sibling_msgs) |> # get var names from Exprs
          sibling_msgs_var_names -> vcat(sibling_msgs_var_names, pot_var_name) |> # concat sibling msgs and potential
          sibling_msgs_and_potential -> :(product($(sibling_msgs_and_potential...)))
      else
        # current bag has parent, current child has sibling(s)
        parent_msg = get_prop(g, parent_node.id, bag.id, :down_msg)
        joint =
          map(sibling -> get_prop(g, sibling.id, bag.id, :up_msg), siblings) |> # get sibling msgs
          sibling_msgs -> map(sibling_msg -> sibling_msg.args[1], sibling_msgs) |> # get var names
          sibling_msgs_var_names -> vcat(sibling_msgs_var_names, parent_msg.args[1], pot_var_name) |> # concat all factors
          all_factors -> :(product($(all_factors...)))
      end

      # Get the variables that need to be marginalized
      bag_vars = get_prop(g, vertices(g)[bag.id], :vars)
      out_edge_sepset = get_prop(g, bag.id, child.id, :sepset)
      mar_vars = setdiff(bag_vars, out_edge_sepset)

      # Marginalize vars
      msg_var_name = Symbol("msg_", bag.id, "_", child.id)
      down_msg = :($msg_var_name = marg($joint, $(mar_vars...)))

      # Set the resulting factor, the downstream message, as an edge property
      set_prop!(g, bag.id, child.id, :down_msg, down_msg)

      # Store the message in childs's incoming messages array
      push!(get_prop(g, child.id, :in_msgs), down_msg)

      # Push the up message expression to the backward pass expression
      push!(backward_pass.args, down_msg)

      # # DEBUG
      # println(down_msg)

    end

  end

  # # DEBUG
  # map(bag -> bag.id , PreOrderDFS(root)) |> 
  #   x -> println("Backward pass visiting order: \n", x)

  # @show backward_pass

  return backward_pass

end

"""
    compile_message_propagation!(g, root)

Compile the forward and backward passes of messages.

"""
function compile_message_propagation!(g, root)

  forward_pass = compile_forward_pass!(g, root)
  backward_pass = compile_backward_pass!(g, root)

  return forward_pass, backward_pass

end

"""
    compile_unnormalized_marginals(g, nvars, partial_evaluation)

Compile marginalization statements for each variable from a sepset if possible
and otherwise from a bag.

"""
function compile_unnormalized_marginals(g, nvars, partial_evaluation)

  edge_marginals = quote end |> rmlines
  bag_marginals = quote end |> rmlines
  unnormalized_marginals = quote end |> rmlines

  # Order edges in ascending order according to the number of vars in the sepset
  edges_ordered = map(edge -> (edge, length(get_prop(g, edge, :sepset))), edges(g)) |>
    x -> sort(x, by=y -> y[end]) |>
    x -> map(y -> y[begin], x)

  # Order bags in ascending order according to their number of vars
  bags_ordered =
    map(bag_id -> (bag_id, length(get_prop(g, bag_id, :vars))), 1:nv(g)) |>
    x -> sort(x, by=y -> y[end]) |>
    x -> map(y -> y[begin], x)

  # For each var in the model
  for var in 1:nvars 
    unnorm_mar_var_name = Symbol("unnorm_mar_", var) # variable name for the marginal of the current var

    # 1. Search current var in sepsets
    for edge in edges_ordered
      sepset = get_prop(g, edge, :sepset)
      sepset_is_consistent = partial_evaluation ? get_prop(g, edge, :isconsistent) : true
      # Is the current var in the current edge sepset AND is the sepset consistent after evidence has been entered?
      if var in sepset && sepset_is_consistent

        # Yes, then 1.1 add edge marginal expr to algo (if not done already) 
        # and 1.2 add an expr that marginalizes the other vars (unnorm_mar_)
        edge_mar_var_name = Symbol("edge_mar_", min(edge.src,edge.dst),"_", max(edge.src,edge.dst))
        up_msg = get_prop(g, edge, :up_msg)
        down_msg = get_prop(g, edge, :down_msg)
        edge_marginal = :($edge_mar_var_name = product($(up_msg.args[1]), $(down_msg.args[1])))

        # 1.1 Has this edge marginal expr already been added to the algo?
        if !has_prop(g, edge, :marg)
          # No, then add it
          push!(edge_marginals.args, edge_marginal)
          # Mark as added
          set_prop!(g, edge, :marg, edge_marginal)
        end

        # 1.2 Marginalize the other vars (if any) and store the resulting expr in the algo
        mar_vars = setdiff(sepset, var)
        if isempty(mar_vars) 
          push!(unnormalized_marginals.args, :($unnorm_mar_var_name = $edge_mar_var_name))
        else
          push!(unnormalized_marginals.args, :($unnorm_mar_var_name = marg($(edge_marginal.args[1]), $(mar_vars...))))
        end

        @goto continue_with_next_var # used to "break" from two nested loops
      end
    end

    # 2. The current var was not found in any edge, then search for it in the bags
    for bag in bags_ordered
      bag_vars = get_prop(g, bag, :vars)
      bag_is_consistent = partial_evaluation ? get_prop(g, bag, :isconsistent) : true
      # Is the current var in the current bag and is the bag consistent after evidence has been entered??
      if var in bag_vars && bag_is_consistent

        # Yes, then 2.1 add bag marginal expr to algo (if not done already) and 2.2 marginalize the other vars
        bag_mar_var_name = Symbol("bag_mar_", bag)
        pot_var_name = Symbol("pot_", bag)

        in_msgs_var_names =
          get_prop(g, bag, :in_msgs) |>
          in_msgs -> map(in_msg -> in_msg.args[1], in_msgs) # get in msg variable names

        bag_marginal =
          vcat(in_msgs_var_names, pot_var_name) |>
          x -> :($bag_mar_var_name = product($(x...)))

        # 2.1 Has this bag marginal expr already been added to the algo?
        if !has_prop(g, bag, :marg)
          push!(bag_marginals.args, bag_marginal) # no, then add it
          set_prop!(g, bag, :marg, bag_marginal) # mark as added
        end

        # 2.2 Marginalize the other vars (if any) and store the resulting expr in the algo
        mar_vars = setdiff(bag_vars, var)
        if isempty(mar_vars) 
          push!(unnormalized_marginals.args, :($unnorm_mar_var_name = $bag_mar_var_name))
        else
          push!(unnormalized_marginals.args, :($unnorm_mar_var_name = marg($(bag_marginal.args[1]), $(mar_vars...))))
        end

        @goto continue_with_next_var
      end
    end

    @label continue_with_next_var
  end

  # # DEBUG
  # @show edge_marginals
  # @show bag_marginals
  # @show unnormalized_marginals

  return edge_marginals, bag_marginals, unnormalized_marginals

end

"""
    compile_normalized_marginals(unnormalized_marginals)

Compile the normalized marginal expressions for each variable in the model.

"""
function compile_normalized_marginals(unnormalized_marginals)

  # TODO: create empty expr array and return it and modify the code at the bottom of DiscreteBayes.jl

  # Normalize all marginals
  normalize_marginals_expr =
    map(x -> x.args[1], unnormalized_marginals.args) |> # get the variable name
    x -> :(norm.([$(x...)])) # create an expression of vector form than normalizes each mar

  return normalize_marginals_expr

end

@enum LastStage ForwardPass BackwardPass JointMarginals UnnormalizedMarginals Marginals

"""
    computeMarginalsExpr(td_filepath,
                         uai_filepath,
                         uai_evid_filepath = "";
                         partial_evaluation = false,
                         last_stage::LastStage = Marginals)

Returns and expression containing the computations to compute the marginals
of all the variables in the model using the junction tree algorithm.

"""
function computeMarginalsExpr(td_filepath,
                              uai_filepath,
                              uai_evid_filepath = ""; 
                              partial_evaluation = false,
                              last_stage::LastStage = Marginals,
                              smart_root_selection = true,
                             )

  g, obsvars, obsvals = construct_td_graph(td_filepath, uai_evid_filepath)
  root, pots, nvars = initialize_td_graph!(g, uai_filepath, smart_root_selection)
  forward_pass, backward_pass = compile_message_propagation!(g, root)

  # ==============================================================================
  ## Partial evaluation
  # ==============================================================================

  if partial_evaluation

    partial_eval_analysis!(g)

    # # DEBUG: Print `pre_eval_msg_` for each edge in forward pass order
    # for bag in PostOrderDFS(root)
    #   parent_bag = Base.parent(root, bag)
    #   isnothing(parent_bag) && break
    #   prop_name = Symbol("pre_eval_msg_", bag.id, "_", parent_bag.id)
    #   println(prop_name, ": ", get_prop(g, bag.id, parent_bag.id, prop_name))
    # end

    # # DEBUG: Print `pre_eval_msg_` for each edge in backward pass order
    # for bag in PreOrderDFS(root)
    #   for child in bag.children
    #     prop_name = Symbol("pre_eval_msg_", bag.id, "_", child.id)
    #     println(prop_name, ": ", get_prop(g, bag.id, child.id, prop_name))
    #   end
    # end

    # # DEBUG: print the isconsistent flag for each bag and edge
    # map(vertex -> ("Bag $vertex", get_prop(g, vertex, :isconsistent)), vertices(g)) |>
    #   x -> show(IOContext(stdout, :limit=>false), MIME"text/plain"(), x)
    # map(edge -> (edge, get_prop(g, edge, :isconsistent)), edges(g)) |>
    #   x -> show(IOContext(stdout, :limit=>false), MIME"text/plain"(), x)

    eval(pots)
    forward_pass_partially_evaled = partially_evaluate(g, forward_pass)

    # # DEBUG
    # @show forward_pass
    # @show forward_pass_partially_evaled

    backward_pass_partially_evaled = partially_evaluate(g, backward_pass)

    # # DEBUG
    # @show backward_pass
    # @show backward_pass_partially_evaled

  end


  # ==============================================================================
  ## Inject reduce statements
  # ==============================================================================

  if partial_evaluation

    pots_redu = inject_redus_in_pots(g, pots, obsvars, obsvals)

    # # DEBUG
    # @show pots
    # @show pots_redu

    forward_pass_pe_redu = inject_redus_in_msgs(g, forward_pass_partially_evaled, obsvars, obsvals)

    # # DEBUG
    # @show forward_pass
    # @show forward_pass_partially_evaled
    # @show forward_pass_pe_redu

    backward_pass_pe_redu = inject_redus_in_msgs(g, backward_pass_partially_evaled, obsvars, obsvals)

    # # DEBUG
    # @show backward_pass
    # @show backward_pass_partially_evaled
    # @show backward_pass_pe_redu

  else

    pots_redu = inject_redus(g, pots, obsvars, obsvals)

    # # DEBUG
    # @show pots
    # @show pots_redu

  end

  edge_marginals, bag_marginals, unnormalized_marginals = compile_unnormalized_marginals(g, nvars, partial_evaluation)

  normalize_marginals_expr = compile_normalized_marginals(unnormalized_marginals)
  
  # # ==============================================================================
  # ## DEBUG: save the message order in edge property
  # # ==============================================================================
  # for (i, bag) in enumerate(PostOrderDFS(root))
  #   parent_bag = Base.parent(root, bag)
  #   isnothing(parent_bag) && break
  #   set_prop!(g, Edge(bag.id, parent_bag.id), :up_msg_order, i)
  # end
  # for (i, bag) in enumerate(PreOrderDFS(root))
  #   for (j, child) in enumerate(bag.children)
  #     set_prop!(g, Edge(bag.id, child.id), :down_msg_order, i+j-1)
  #   end
  # end

  # ==============================================================================
  ## Finalize algorithm
  # ==============================================================================

  # Concatenate the different expressions corresponding to the Junction Tree algo steps
  if last_stage == ForwardPass
    algo = Expr(:block, vcat(pots_redu,
                             partial_evaluation ? forward_pass_pe_redu : forward_pass,
                            )...)
  elseif last_stage == BackwardPass
    algo = Expr(:block, vcat(pots_redu,
                             partial_evaluation ? forward_pass_pe_redu : forward_pass,
                             partial_evaluation ? backward_pass_pe_redu : backward_pass,
                            )...)
  elseif last_stage == JointMarginals
    algo = Expr(:block, vcat(pots_redu,
                             partial_evaluation ? forward_pass_pe_redu : forward_pass,
                             partial_evaluation ? backward_pass_pe_redu : backward_pass,
														 edge_marginals,
														 bag_marginals,
                            )...)
  elseif last_stage == UnnormalizedMarginals
    algo = Expr(:block, vcat(pots_redu,
                             partial_evaluation ? forward_pass_pe_redu : forward_pass,
                             partial_evaluation ? backward_pass_pe_redu : backward_pass,
														 edge_marginals,
														 bag_marginals,
														 unnormalized_marginals,
                            )...)
  elseif last_stage == Marginals
    algo = Expr(:block, vcat(pots_redu,
                             partial_evaluation ? forward_pass_pe_redu : forward_pass,
                             partial_evaluation ? backward_pass_pe_redu : backward_pass,
														 edge_marginals,
														 bag_marginals,
														 unnormalized_marginals,
														 Expr(:block, normalize_marginals_expr),
                            )...)
  end

  # # DEBUG
  # println(algo)

  # # DEBUG
  # @btime eval(algo)

  function_name = :compute_marginals
  sig = (Tuple, Tuple)
  variables = [:obsvars, :obsvals]
  body = algo

  return generate_function_expression(function_name, sig, variables, body)

end

end # module

