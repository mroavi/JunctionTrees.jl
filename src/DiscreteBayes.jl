module DiscreteBayes
##
import Base:
  eltype,
  IteratorEltype,
  parent

export computeMarginalsExpr,
  Factor,
  product!,
  init_prod,
  marg!,
  init_marg,
  redu,
  norm

using LightGraphs, MetaGraphs, AbstractTrees, CommonSubexpressions, StaticArrays

include("factors.jl")
include("tree.jl")

"""
    constructTreeDecompositionAbstractTree!(g, root)

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
function constructTreeDecompositionAbstractTree!(g, root, parent=root)

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

"""
    hasObservedNode(g, root)

    Returns a boolean stating whether the input (sub)tree has at least one observed
    node.
"""
function hasObservedNode(g, root)

  for node in PostOrderDFS(root)
    has_prop(g, node.id, :isobserved) && return true
  end

  return false

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

# -----------------------------------------------------------------------------
# TODO: erase me. TEMP: handy while developing
Base.show(io::IO, x::Array{Float64}) = print(io, "[..]")
# problem = "Promedus_11"
# problem = "Promedus_26"
# problem = "01-example-paskin"
# problem = "03-merlin-simple6"
problem = "05-mrv"

problem_dir = joinpath(homedir(), "repos/partial-evaluation/problems/"*problem*"/")

td_filepath = problem_dir*problem*".td"
uai_filepath = problem_dir*problem*".uai"
uai_evid_filepath = problem_dir*problem*".uai.evid"
# -----------------------------------------------------------------------------

##
"""
    computeMarginalsExpr(td_filepath, uai_filepath, uai_evid_filepath)

Construct a tree decomposition graph based on `td_filepath`.
Assign the factor tables defined in `uai_filepath` to one bag (cluster)
and mark the observed variables according to `uai_evid_filepath`.

The `td_filepath` file format is defined in:
https://pacechallenge.org/2017/treewidth/.

The `uai_filepath` file format is defined in:
http://www.hlt.utdallas.edu/~vgogate/uai14-competition/modelformat.html

The `uai_evid_filepath` file format is defined in :
http://www.hlt.utdallas.edu/~vgogate/uai14-competition/evidformat.html

# Example
```
td_filepath       = "../problems/Promedus_26/Promedus_26.td"
uai_filepath      = "../problems/Promedus_26/Promedus_26.uai"
uai_evid_filepath = "../problems/Promedus_26/Promedus_26.evid"
g = computeMarginalsExpr(td_filepath, uai_evid_filepath)
```
"""
function computeMarginalsExpr(td_filepath, uai_filepath, uai_evid_filepath)

  ## Empty expressions that will be filled with nested expressions that
  # initialize and run the junction tree algorithm
  init_expr = quote end
  run_expr = quote end

  # Read the td file into an array of lines
  rawlines = open(td_filepath) do file
    readlines(file)
  end

  # Filter out comments
  lines = filter(x -> !startswith(x, "c"), rawlines)

  # Extract number of bags, treewidth+1 and number of vertices from solution line
  nbags, treewidth, nvertices = split(lines[1]) |> soln_line->soln_line[3:5] |> x->parse.(Int, x)

  # Read the uai evid file into an array of lines
  line = open(uai_evid_filepath) do file
    readlines(file)
  end

  @assert length(line) == 1

  # Extract number of observed vars, and their id together with their corresponding value
  nobsvars, rest = split(line[1]) |> x -> parse.(Int, x) |> x -> (x[1], x[2:end])
  obsvars = reshape(rest, 2, :)

  # Convert to 1-based indexing
  obsvars[1,:] = obsvars[1,:] .+ 1

  @assert nobsvars == size(obsvars)[2]

  # Initialize an empty MetaGraph
  g = MetaGraph()

  # ==============================================================================
  # Construct the bags
  # ==============================================================================

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

    # Mark bags (clusters) that contain at least one observed var
    has_observed_var = intersect(bag_vertices, obsvars[1,:]) |> !isempty
    has_observed_var && set_prop!(g, bag_id, :isobserved, true)

    # Create an empty vector of factors
    set_prop!(g, bag_id, :factors, Factor{Float64}[])

    # Create an empty vector of expressions of incoming messages
    set_prop!(g, bag_id, :in_msgs, Union{Expr, Factor{Float64}}[])

  end

  # # DEBUG
  # obsvars # observed vars with their corresponding value (vars on first row, vals on second row)
  # map(x -> string(x,": ",get_prop(g,x,:vars)), vertices(g)) |> x -> show(stdout, "text/plain", x)
  # filter_vertices(g, :isobserved) |> collect # bags that contain at least one observed var

  # ==============================================================================
  # Construct the edges
  # ==============================================================================

  # Extract edge definition lines
  edge_lines = lines[(2+nbags):end]

  # Add each edge to the graph and store the intersection of vars between the bags it connects
  for edge in edge_lines

    # Parse and add the edge to the graph
    edge_src, edge_dst = split(edge) |> x -> parse.(Int, x)
    add_edge!(g, edge_src, edge_dst)

    # Find and set the sepset as edge's property
    vars_src = get_prop(g, edge_src, :vars)
    vars_dst = get_prop(g, edge_dst, :vars)
    sepset = intersect(vars_src, vars_dst)
    set_prop!(g, Edge(edge_src, edge_dst), :sepset, sepset)

  end

  # # DEBUG
  # println("\nSepset of each edge:")
  # map(edge -> get_prop(g, edge, :sepset), edges(g)) |> display # sepset of each edge

  # ==============================================================================
  # leaves
  # ==============================================================================

  map(x -> length(neighbors(g, x)), vertices(g)) |>   # number of neighbors for each bag
    x -> findall(isone, x) |>                         # indices of the leaves
    x -> map(y -> set_prop!(g, y, :isleaf, true), x)  # set a property for the leaf bags

  # # DEBUG
  # println("\nLeaf bags:")
  # filter_vertices(g, :isleaf) |> collect |> display

  # ==============================================================================
  # Read the factors
  # ==============================================================================

  # Read the uai file into an array of lines
  rawlines = open(uai_filepath) do file
    readlines(file)
  end

  # Filter out empty lines
  lines = filter(!isempty, rawlines)

  nvars   = lines[2] |> x -> parse.(Int, x)
  card    = lines[3] |> split |> x -> parse.(Int, x)
  ntables = lines[4] |> x -> parse.(Int, x)

  @assert nvars == nvertices

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

  # ==============================================================================
  # # Assign each factor to a cluster
  # ==============================================================================

  # Construct an abstract tree using AbstractTrees.jl
  root = Node(1) # arbitrarily choosing Node 1 as root
  constructTreeDecompositionAbstractTree!(g, root)

  # # DEBUG
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

  # ==============================================================================
  # Compute potentials for each bag and store them in a bag property
  # ==============================================================================

  for bag in vertices(g)
    bag_factors = get_prop(g, bag, :factors)
    if isempty(bag_factors)
      potential = Factor{Float64,0}((), [1.0])
    else
      potential = product(bag_factors...)
    end
    set_prop!(g, bag, :potential, potential)
  end

  # # DEBUG
  # map(vertex -> (vertex, get_prop(g, vertex, :potential)), vertices(g)) # potential of each bag

  # ==============================================================================
  # # Initialize the upstream messages
  # ==============================================================================

  # Visit each bag in postorder and initialize its upstream message
  for bag in PostOrderDFS(root)

    # Get parent bag from abstract tree
    parent_bag = Base.parent(root, bag)

    # Exit loop if the current bag is the root (i.e. if the current bag has no parent)
    isnothing(parent_bag) && break

    # Used to construct an assignment expr of the factor product between incomming msgs and potential
    _msg_var_name = Symbol("_msg_", bag.id, "_", parent_bag.id)
    _msg_in_card_new = Symbol("_msg_", bag.id, "_", parent_bag.id, "_in_card_new")

    # Initialize the joint distribution of all incoming messages and the potential
    potential = get_prop(g, bag.id, :potential)
    # Is this bag a leaf?
    if isempty(bag.children)
      # Yes, then only extract the potential
      potential_card = (size(potential.vals),) # wrap potential card in tuple
      _up_msg = :(($_msg_var_name,$_msg_in_card_new) = ($potential,$potential_card))
    else
      # No, then construct an expr of the factor product between incomming msgs and potential
      _up_msg =
        map(child -> child.id, bag.children) |> # get the children ids of the current bag
        children_ids -> map(child_id -> get_prop(g, child_id, bag.id, :up_msg), children_ids) |>
        in_msgs -> map(in_msg -> in_msg.args[1].args[1], in_msgs) |> # get the msg variable name from Expr
        in_msgs_var_names -> vcat(in_msgs_var_names, potential) |> # concat in msgs and potential
        in_msgs_and_pot -> :(($_msg_var_name,$_msg_in_card_new) = init_prod($(in_msgs_and_pot...))) # splatting interpol
    end

    # Push the the assignment expression defined above to the init_expr expression
    push!(init_expr.args, _up_msg)

    # Get the variables that need to be marginalized
    bag_vars = get_prop(g, bag.id, :vars)
    out_edge_sepset = get_prop(g, bag.id, parent_bag.id, :sepset)
    mar_vars = setdiff(bag_vars, out_edge_sepset)

    # Construct an assignment expression for the up msg by marginilizing the necessary vars
    msg_var_name = Symbol("msg_", bag.id, "_", parent_bag.id)
    msg_r_vals = Symbol("msg_", bag.id, "_", parent_bag.id, "_r_vals")
    msg_mar_dims = Symbol("msg_", bag.id, "_", parent_bag.id, "_mar_dims")
    up_msg = :(($msg_var_name,$msg_r_vals,$msg_mar_dims) = init_marg($(_up_msg.args[1].args[1]), $(mar_vars...)))

    # Set the resulting factor, the upstream message, as an edge property
    set_prop!(g, bag.id, parent_bag.id, :up_msg, up_msg)

    # Store message in parent's bag incoming messages array
    push!(get_prop(g, parent_bag.id, :in_msgs), up_msg)

    # Push the current up message alloc expression to the init_expr expression
    push!(init_expr.args, up_msg)

    # # DEBUG
    # println(up_msg)

  end

  # # DEBUG
  # println(init_expr)

  # # DEBUG
  # @time eval(init_expr)
  # @btime eval(init_expr) 

  # # DEBUG
  # map(bag -> bag.id , PostOrderDFS(root)) |> 
  #   x -> println("\nForward pass visiting order: \n", x)

  # ==============================================================================
  # Initialize the downstream messages
  # ==============================================================================

  # Visit each bag in preorder and initialize the messages through all edges other
  # than the one connecting it to the parent
  for bag in PreOrderDFS(root)

    # Get parent bag from abstract tree (if the bag has no parent `nothing` is returned)
    parent_node = Base.parent(root, bag)

    # Compute and send a message to each child
    for child in bag.children

      # Used to construct an assignment expr of the factor product between in msgs and potential
      _msg_var_name = Symbol("_msg_", bag.id, "_", child.id)
      _msg_in_card_new = Symbol("_msg_", bag.id, "_", child.id, "_in_card_new")

      # Compute joint btwn the incoming msgs (coming from parent and siblings) and bag's potential
      potential = get_prop(g, bag.id, :potential)
      siblings = setdiff(bag.children, [child]) # get a list of the siblings (all other children)

      # Four possible scenarios:
      if isnothing(parent_node) && isempty(siblings)
        # current bag has no parent, current child has no sibling(s)
        potential_card = (size(potential.vals),) # wrap potential card in tuple
        _down_msg = :(($_msg_var_name,$_msg_in_card_new) = ($potential,$potential_card))
      elseif !isnothing(parent_node) && isempty(siblings)
        # current bag has parent, current child has no sibling(s)
        parent_msg = get_prop(g, parent_node.id, bag.id, :down_msg) |> # get down msg from prop
          down_msg -> down_msg.args[1].args[1] # get the msg variable name
        _down_msg = :(($_msg_var_name,$_msg_in_card_new) = init_prod($parent_msg, $potential))
      elseif isnothing(parent_node) && !isempty(siblings)
        # current bag has no parent, current child has sibling(s)
        _down_msg =
          map(sibling -> get_prop(g, sibling.id, bag.id, :up_msg), siblings) |> # get sibling msgs
          sibling_msgs -> map(sibling_msg -> sibling_msg.args[1].args[1], sibling_msgs) |> # get var names from Exprs
          sibling_msgs_var_names -> vcat(sibling_msgs_var_names, potential) |> # concat sibling msgs and potential
          sibling_msgs_and_pot -> :(($_msg_var_name,$_msg_in_card_new) = init_prod($(sibling_msgs_and_pot...)))
      else
        # current bag has parent, current child has sibling(s)
        parent_msg = get_prop(g, parent_node.id, bag.id, :down_msg)
        _down_msg =
          map(sibling -> get_prop(g, sibling.id, bag.id, :up_msg), siblings) |> # get sibling msgs
          sibling_msgs -> map(sibling_msg -> sibling_msg.args[1].args[1], sibling_msgs) |> # get var names
          sibling_msgs_var_names -> vcat(parent_msg.args[1].args[1], sibling_msgs_var_names, potential) |> # concat all
          all_factors -> :(($_msg_var_name,$_msg_in_card_new) = init_prod($(all_factors...)))
      end

      # Push the the assignment expression defined above to the init_expr expression
      push!(init_expr.args, _down_msg)

      # Get the variables that need to be marginalized
      bag_vars = get_prop(g, vertices(g)[bag.id], :vars)
      out_edge_sepset = get_prop(g, bag.id, child.id, :sepset)
      mar_vars = setdiff(bag_vars, out_edge_sepset)

      # Construct an assignment expression for the up msg by marginilizing the necessary vars
      msg_var_name = Symbol("msg_", bag.id, "_", child.id)
      msg_r_vals = Symbol("msg_", bag.id, "_", child.id, "_r_vals")
      msg_mar_dims = Symbol("msg_", bag.id, "_", child.id, "_mar_dims")
      down_msg = :(($msg_var_name,$msg_r_vals,$msg_mar_dims) = init_marg($(_down_msg.args[1].args[1]), $(mar_vars...)))

      # Set the resulting factor, the downstream message, as an edge property
      set_prop!(g, bag.id, child.id, :down_msg, down_msg)

      # Store the message in childs's incoming messages array
      push!(get_prop(g, child.id, :in_msgs), down_msg)

      # Push the up message expression to the run_expr expression
      push!(init_expr.args, down_msg)

      # # DEBUG
      # println(down_msg)

    end
  end

  # # DEBUG
  # print(init_expr)

  # # DEBUG
  # @time eval(init_expr)
  # @btime eval(init_expr) 

  # # DEBUG
  # map(bag -> bag.id , PreOrderDFS(root)) |> 
  #   x -> println("Backward pass visiting order: \n", x)

  # ==============================================================================
  # # Compute the upstream message
  # ==============================================================================

  # Visit each bag in postorder and compute its upstream message
  for bag in PostOrderDFS(root)

    # Get parent bag from abstract tree
    parent_bag = Base.parent(root, bag)

    # Exit loop if the current bag is the root (i.e. if the current bag has no parent)
    isnothing(parent_bag) && break

    # Used to construct an assignment expr of the factor product between incomming msgs and potential
    _msg_var_name = Symbol("_msg_", bag.id, "_", parent_bag.id)
    _msg_in_card_new = Symbol("_msg_", bag.id, "_", parent_bag.id, "_in_card_new")

    # Compute the joint distribution of all incoming messages and the potential
    potential = get_prop(g, bag.id, :potential)
    # Is this bag a leaf?
    if isempty(bag.children)
      # Yes, then only extract the potential
      _up_msg = :(product!($_msg_var_name, $_msg_in_card_new, ($potential,)))
    else
      # No, then construct an expr of the factor product between incomming msgs and potential
      _up_msg =
        map(child -> child.id, bag.children) |> # get the children ids of the current bag
        children_ids -> map(child_id -> get_prop(g, child_id, bag.id, :up_msg), children_ids) |>
        in_msgs -> map(in_msg -> in_msg.args[2], in_msgs) |> # get the msg variable name from Expr
        in_msgs_var_names -> vcat(in_msgs_var_names, potential) |> # concat in msgs and potential
        in_msgs_and_pot -> :(product!($_msg_var_name, $_msg_in_card_new, ($(in_msgs_and_pot...),))) #splatting interpol
    end

    # Push the the assignment expression defined above to the init_expr expression
    push!(run_expr.args, _up_msg)

    # Get the variables that need to be marginalized
    bag_vars = get_prop(g, bag.id, :vars)
    out_edge_sepset = get_prop(g, bag.id, parent_bag.id, :sepset)
    mar_vars = setdiff(bag_vars, out_edge_sepset)

    # Marginalize vars to compute the upstream message
    msg_var_name = Symbol("msg_", bag.id, "_", parent_bag.id)
    _msg_var_name = Symbol("_", msg_var_name)
    msg_r_vals = Symbol(msg_var_name, "_r_vals")
    msg_mar_dims = Symbol(msg_var_name, "_mar_dims")
    up_msg = :(marg!($msg_var_name, $_msg_var_name, $msg_r_vals, $msg_mar_dims))

    # # ----------------------------- PARTIAL EVALUATION ------------------------
    # # Does the subtree with the current bag as root have any bag with an observed var?
    # if hasObservedNode(g, bag)
    #   # Yes, the evaluate the upstream message
    #   up_msg = eval(up_msg)
    # end
    # # -------------------------------------------------------------------------

    # Set the resulting factor, the upstream message, as an edge property
    set_prop!(g, bag.id, parent_bag.id, :up_msg, up_msg)

    # Store message in parent's bag incoming messages array
    push!(get_prop(g, parent_bag.id, :in_msgs), up_msg)

    # Push the current up message expression to the run_expr expression
    push!(run_expr.args, up_msg)

    # # DEBUG
    # println(up_msg)

  end

  # # DEBUG
  # println(init_expr)
  # println(run_expr)

  # # # DEBUG
  # @btime eval(init_expr)
  # @btime eval(run_expr) 

  # # DEBUG
  # map(bag -> bag.id , PostOrderDFS(root)) |> 
  #   x -> println("\nForward pass visiting order: \n", x)

  # ==============================================================================
  # Compute the downstream messages
  # ==============================================================================

  # Visit each bag in preorder and compute the messages through all edges other
  # than the one connecting it to the parent
  for bag in PreOrderDFS(root)

    # Get parent bag from abstract tree
    parent_node = Base.parent(root, bag)

    # Compute and send a message to each child
    for child in bag.children

      # Used to construct an assignment expr of the product between incomming msgs and potential
      _msg_var_name = Symbol("_msg_", bag.id, "_", child.id)
      _msg_in_card_new = Symbol("_msg_", bag.id, "_", child.id, "_in_card_new")

      # Compute joint btwn the incoming msgs (coming from parent and siblings) and bag's potential
      potential = get_prop(g, bag.id, :potential)
      siblings = setdiff(bag.children, [child]) # get a list of the siblings (all other children)

      # Four possible scenarios:
      if isnothing(parent_node) && isempty(siblings)
        # current bag has no parent, current child has no sibling(s)
        _down_msg = :(product!($_msg_var_name, $_msg_in_card_new, ($potential,)))
      elseif !isnothing(parent_node) && isempty(siblings)
        # current bag has parent, current child has no sibling(s)
        parent_msg = get_prop(g, parent_node.id, bag.id, :down_msg) |> # get down msg from prop
          down_msg -> down_msg.args[2] # get the msg variable name
        _down_msg = :(product!($_msg_var_name, $_msg_in_card_new, ($parent_msg, $potential)))
      elseif isnothing(parent_node) && !isempty(siblings)
        # current bag has no parent, current child has sibling(s)
        _down_msg =
          map(sibling -> get_prop(g, sibling.id, bag.id, :up_msg), siblings) |> # get sibling msgs
          sibling_msgs -> map(sibling_msg -> sibling_msg.args[2], sibling_msgs) |> # get var names from Exprs
          sibling_msgs_var_names -> vcat(sibling_msgs_var_names, potential) |> # concat sibling msgs and potential
          sibling_msgs_and_potential -> :(product!($_msg_var_name, $_msg_in_card_new, ($(sibling_msgs_and_potential...),)))
      else
        # current bag has parent, current child has sibling(s)
        parent_msg = get_prop(g, parent_node.id, bag.id, :down_msg)
        _down_msg =
          map(sibling -> get_prop(g, sibling.id, bag.id, :up_msg), siblings) |> # get sibling msgs
          sibling_msgs -> map(sibling_msg -> sibling_msg.args[2], sibling_msgs) |> # get var names
          sibling_msgs_var_names -> vcat(parent_msg.args[2], sibling_msgs_var_names, potential) |> # concat all factors
          all_factors -> :(product!($_msg_var_name, $_msg_in_card_new, ($(all_factors...),)))
      end

      # Push the the assignment expression defined above to the init_expr expression
      push!(run_expr.args, _down_msg)

      # Get the variables that need to be marginalized
      bag_vars = get_prop(g, vertices(g)[bag.id], :vars)
      out_edge_sepset = get_prop(g, bag.id, child.id, :sepset)
      mar_vars = setdiff(bag_vars, out_edge_sepset)

      # Marginalize vars to compute the downstream message
      msg_var_name = Symbol("msg_", bag.id, "_", child.id)
      _msg_var_name = Symbol("_", msg_var_name)
      msg_r_vals = Symbol(msg_var_name, "_r_vals")
      msg_mar_dims = Symbol(msg_var_name, "_mar_dims")
      down_msg = :(marg!($msg_var_name, $_msg_var_name, $msg_r_vals, $msg_mar_dims))

      # Set the resulting factor, the downstream message, as an edge property
      set_prop!(g, bag.id, child.id, :down_msg, down_msg)

      # Store the message in childs's incoming messages array
      push!(get_prop(g, child.id, :in_msgs), down_msg)

      # Push the up message expression to the run_expr expression
      push!(run_expr.args, down_msg)

      # # DEBUG
      # println(down_msg)

    end

  end

  # # DEBUG
  # println(init_expr)
  # println(run_expr)

  # # # # DEBUG
  # @time eval(init_expr)
  # @time eval(run_expr) 

  # # DEBUG
  # map(bag -> bag.id , PreOrderDFS(root)) |> 
  #   x -> println("Backward pass visiting order: \n", x)

  # ==============================================================================
  # # Compute bag marginals
  # ==============================================================================

  # Used to store the unnormalized bag marginals
  bag_marginals = Vector{Expr}(undef, nbags)

  # A bag marginal is the factor product of incomming msgs and potential
  # TODO: Compute marginals more efficiently: extract them from sepsets, not from bags
  for bag in PostOrderDFS(root)
    # Bag marginal expression
    potential = get_prop(g, bag.id, :potential)

    bag_mar_var_name = Symbol("mar_bag_", bag.id)
    bag_mar_in_card_new = Symbol("mar_bag_", bag.id, "_in_card_new")

    neighbors(g, bag.id) |>
      neighbors -> map(neighbor -> Symbol("msg_", neighbor, "_", bag.id), neighbors) |>
      in_msgs -> :(($bag_mar_var_name,$bag_mar_in_card_new) = init_prod($(in_msgs...), $potential)) |>
      bag_marginal_expr -> push!(init_expr.args, bag_marginal_expr)

    neighbors(g, bag.id) |>
      neighbors -> map(neighbor -> Symbol("msg_", neighbor, "_", bag.id), neighbors) |>
      in_msgs -> :(product!($bag_mar_var_name,$bag_mar_in_card_new,($(in_msgs...), $potential))) |>
      bag_marginal_expr -> push!(run_expr.args, bag_marginal_expr)
  end

  # # DEBUG
  # println(init_expr)
  # println(run_expr)

  # # DEBUG
  # @btime eval(init_expr)
  # @btime eval(run_expr) 

  # ==============================================================================
  # # Compute marginals
  # ==============================================================================

  # Traverse bags in ascending order according to their number of vars
  bag_traversal_order =
    map(bag_id -> (bag_id, length(get_prop(g, bag_id, :vars))), 1:nbags) |> # (bag_id, nvars)
    nvars_per_bag -> sort(nvars_per_bag, by=x -> x[end]) |> # sort nested tuples by nvars
    sorted_nvars_per_bag -> map(x -> x[begin], sorted_nvars_per_bag) # take the bag_id only

  # Used to store the unnormalized marginal expressions
  init_unnormalized_marginals = Vector{Expr}(undef, nvars)
  unnormalized_marginals = Vector{Expr}(undef, nvars)

  for bag_id in bag_traversal_order
    # Marginal expressions for each variable
    bag_vars = get_prop(g, bag_id, :vars)
    for var in bag_vars

      isassigned(init_unnormalized_marginals, var) && continue
      mar_vars = setdiff(bag_vars, var)

      # Construct an init assignment expr for the bag marginal by marginilizing the necessary vars
      unnorm_mar_var_name = Symbol("unnorm_mar_", var)
      unnorm_mar_r_vals = Symbol("unnorm_mar_", var, "_r_vals")
      unnorm_mar_dims = Symbol("unnorm_mar_", var, "_mar_dims")
      bag_mar_var_name = Symbol("mar_bag_", bag_id)
      init_unnormalized_marginals[var] = 
        :(($unnorm_mar_var_name,$unnorm_mar_r_vals,$unnorm_mar_dims) = 
          init_marg($bag_mar_var_name, $(mar_vars...)))

      # Construct an assignment expr for the bag marginal by marginilizing the necessary vars
      unnormalized_marginals[var] = 
        :(marg!($unnorm_mar_var_name, $bag_mar_var_name, $unnorm_mar_r_vals, $unnorm_mar_dims))
    end
  end

  # Push each unnormalized marginal expression to the init_expr and run_expr expressions
  map(x -> push!(init_expr.args, x), init_unnormalized_marginals)
  map(x -> push!(run_expr.args, x), unnormalized_marginals)

  # # DEBUG
  # map(unnormalized_marginal -> println(unnormalized_marginal), unnormalized_marginals)

  # Normalize all marginals
  normalize_marginals_expr =
    map(unnorm_marg_id -> Symbol("unnorm_mar_", unnorm_marg_id), 1:nvars) |> # construct the var name
    unnorm_mar_var_name -> :(norm.([$(unnorm_mar_var_name...)])) # create an expr of than normalizes each mar

  push!(run_expr.args, normalize_marginals_expr)

  # # DEBUG
  # println(init_expr)
  # println(run_expr)

  # # DEBUG
  # @btime eval(init_expr)
  # @btime eval(run_expr) 

;
  # ==============================================================================
  ## Common subexpression elimination
  # ==============================================================================
  # algo_cse = CommonSubexpressions.binarize(run_expr) |> cse

  # # DEBUG
  # print(algo_cse)
;
  ##

  function_name = :init_algo
  sig = ()
  variables = []
  body = init_expr

  init_function_expr = generate_function_expression(function_name, sig, variables, body)

  function_name = :run_algo
  sig = ()
  variables = []
  body = run_expr

  run_function_expr = generate_function_expression(function_name, sig, variables, body)

  return init_function_expr, run_function_expr
  # return g # TODO: TEMP: uncomment to use with the plotting utilities in Util
end

end # module
