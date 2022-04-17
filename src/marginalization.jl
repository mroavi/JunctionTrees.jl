"""
$(TYPEDSIGNATURES)

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
        edge_marginal = :($edge_mar_var_name = prod($(up_msg.args[1]), $(down_msg.args[1])))

        # 1.1 Has this edge marginal expr already been added to the algo?
        if !has_prop(g, edge, :sum)
          # No, then add it
          push!(edge_marginals.args, edge_marginal)
          # Mark as added
          set_prop!(g, edge, :sum, edge_marginal)
        end

        # 1.2 Marginalize the other vars (if any) and store the resulting expr in the algo
        mar_vars = setdiff(sepset, var)
        if isempty(mar_vars) 
          push!(unnormalized_marginals.args, :($unnorm_mar_var_name = $edge_mar_var_name))
        else
          push!(unnormalized_marginals.args, :($unnorm_mar_var_name = sum($(edge_marginal.args[1]), $(mar_vars...))))
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
          x -> :($bag_mar_var_name = prod($(x...)))

        # 2.1 Has this bag marginal expr already been added to the algo?
        if !has_prop(g, bag, :sum)
          push!(bag_marginals.args, bag_marginal) # no, then add it
          set_prop!(g, bag, :sum, bag_marginal) # mark as added
        end

        # 2.2 Marginalize the other vars (if any) and store the resulting expr in the algo
        mar_vars = setdiff(bag_vars, var)
        if isempty(mar_vars) 
          push!(unnormalized_marginals.args, :($unnorm_mar_var_name = $bag_mar_var_name))
        else
          push!(unnormalized_marginals.args, :($unnorm_mar_var_name = sum($(bag_marginal.args[1]), $(mar_vars...))))
        end

        @goto continue_with_next_var
      end
    end

    @label continue_with_next_var
  end

  # # DEBUG:
  # @show edge_marginals
  # @show bag_marginals
  # @show unnormalized_marginals

  return edge_marginals, bag_marginals, unnormalized_marginals

end
