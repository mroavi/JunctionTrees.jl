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
  convertToAbstractTree!(g, root)

  # # DEBUG:
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

  # # DEBUG:
  # map(vertex -> println("Bag $vertex: ", get_prop(g, vertex, :vars)), vertices(g)) # vars on which each bag depends on
  # map(vertex -> println("Bag $vertex: ", get_prop(g, vertex, :factors)), vertices(g)) # factors assigned to each bag

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

  # # DEBUG:
  # map(vertex -> (vertex, get_prop(g, vertex, :pot)), vertices(g)) # potential of each bag

  # DEBUG:
  # println(pots)
  # eval(pots)

  return pots
end

"""
    initialize_td_graph!(g, uai_filepath, smart_root_selection)

Initialize the td graph by assigning the different factors to one bag
that covers its scope.

"""
function initialize_td_graph!(g, uai_filepath::String, smart_root_selection)

  nvars, cards, ntables, factors = read_uai_file(uai_filepath)
  root = assign_factors!(g, factors, smart_root_selection)
  pots = compile_bag_potentials(g)

  return root, pots, nvars

end

