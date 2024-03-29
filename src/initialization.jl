"""
$(TYPEDSIGNATURES)

Assign each factor to a cluster that covers its variables.
"""
function assign_factors!(g, factors, root)

  # Construct an abstract tree using AbstractTrees.jl
  convertToAbstractTree!(g, root)

  # # DEBUG:
  # @show root.id
  # println("\nCluster tree:")
  # print_tree(root)

  # Initialize an empty vector of factors for each bag
  map(bag_id -> set_prop!(g, bag_id, :factors, Factor[]), vertices(g))

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

  # # DEBUG:
  # map(vertex -> println("Bag $vertex: ", get_prop(g, vertex, :vars)), vertices(g)) # vars on which each bag depends on
  # map(vertex -> println("Bag $vertex: ", get_prop(g, vertex, :factors)), vertices(g)) # factors assigned to each bag

end

"""
$(TYPEDSIGNATURES)

Compile each bag's potential into a Julia expression.
"""
function compile_bag_potentials(g, factor_eltype)

  pots = quote end |> rmlines

  # For each bag
  for bag in vertices(g)
    # Are there any factors assigned to the current bag?
    bag_factors = get_prop(g, bag, :factors)
    if isempty(bag_factors)
      # No, then assign a "unit" potential to this bag
      pot = Factor{factor_eltype,0}((), Array{factor_eltype,0}(undef))
    else
      # Yes, then compute the product of all potentials assigned to the current bag
      pot = prod(bag_factors...)
    end
    pot_var_name = Symbol("pot_", bag)
    push!(pots.args, :($pot_var_name = $pot))
  end

  # # DEBUG:
  # map(vertex -> (vertex, get_prop(g, vertex, :pot)), vertices(g)) # potential of each bag

  # DEBUG:
  # println(pots)
  # eval(pots)

  return pots
end

"""
$(TYPEDSIGNATURES)

Initialize the td graph by assigning the different factors to one bag
that covers its scope.
"""
function initialize_td_graph!(g, factors, smart_root_selection)

  root = select_rootnode(g, smart_root_selection = smart_root_selection)
  set_prop!(g, root.id, :isroot, true)
  assign_factors!(g, factors, root)
  pots = compile_bag_potentials(g, eltype(factors[1]))

  return root, pots

end

"""
$(TYPEDSIGNATURES)

Select and return the root node of `g`.
"""
function select_rootnode(g; smart_root_selection = true)

  if smart_root_selection
    root = map(v -> get_prop(g, v, :vars) |> length, vertices(g)) |> findmax |> x -> Node(x[2])
  else
    root = Node(1) # arbitrarily chose Node 1 as root
  end

  return root

end
