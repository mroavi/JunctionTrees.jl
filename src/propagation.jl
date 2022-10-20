"""
$(TYPEDSIGNATURES)

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
        in_msgs_and_potential -> :(prod($(in_msgs_and_potential...))) #splatting interpol
    end

    # Get the variables that need to be marginalized
    bag_vars = get_prop(g, bag.id, :vars)
    out_edge_sepset = get_prop(g, bag.id, parent_bag.id, :sepset)
    mar_vars = setdiff(bag_vars, out_edge_sepset)

    # Marginalize vars to compute the upstream message
    msg_var_name = Symbol("msg_", bag.id, "_", parent_bag.id)
    up_msg = :($msg_var_name = sum($joint, $(mar_vars...)))

    # Set the resulting factor, the upstream message, as an edge property
    set_prop!(g, bag.id, parent_bag.id, :up_msg, up_msg)

    # Store message in parent's bag incoming messages array
    push!(get_prop(g, parent_bag.id, :in_msgs), up_msg)

    # Push the current up message expression to the algo expression
    push!(forward_pass.args, up_msg)

    # # DEBUG:
    # println(up_msg)

  end

  # # DEBUG:
  # map(bag -> bag.id , PostOrderDFS(root)) |> 
  #   x -> println("\nForward pass visiting order: \n", x)

  # @show forward_pass
  # @time eval(forward_pass) 
  # @btime eval(forward_pass) 

  return forward_pass

end

"""
$(TYPEDSIGNATURES)

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
        joint = :(prod($parent_msg, $pot_var_name))
      elseif isnothing(parent_node) && !isempty(siblings)
        # current bag has no parent, current child has sibling(s)
        joint =
          map(sibling -> get_prop(g, sibling.id, bag.id, :up_msg), siblings) |> # get sibling msgs
          sibling_msgs -> map(sibling_msg -> sibling_msg.args[1], sibling_msgs) |> # get var names from Exprs
          sibling_msgs_var_names -> vcat(sibling_msgs_var_names, pot_var_name) |> # concat sibling msgs and potential
          sibling_msgs_and_potential -> :(prod($(sibling_msgs_and_potential...)))
      else
        # current bag has parent, current child has sibling(s)
        parent_msg = get_prop(g, parent_node.id, bag.id, :down_msg)
        joint =
          map(sibling -> get_prop(g, sibling.id, bag.id, :up_msg), siblings) |> # get sibling msgs
          sibling_msgs -> map(sibling_msg -> sibling_msg.args[1], sibling_msgs) |> # get var names
          sibling_msgs_var_names -> vcat(sibling_msgs_var_names, parent_msg.args[1], pot_var_name) |> # concat all factors
          all_factors -> :(prod($(all_factors...)))
      end

      # Get the variables that need to be marginalized
      bag_vars = get_prop(g, vertices(g)[bag.id], :vars)
      out_edge_sepset = get_prop(g, bag.id, child.id, :sepset)
      mar_vars = setdiff(bag_vars, out_edge_sepset)

      # Marginalize vars
      msg_var_name = Symbol("msg_", bag.id, "_", child.id)
      down_msg = :($msg_var_name = sum($joint, $(mar_vars...)))

      # Set the resulting factor, the downstream message, as an edge property
      set_prop!(g, bag.id, child.id, :down_msg, down_msg)

      # Store the message in childs's incoming messages array
      push!(get_prop(g, child.id, :in_msgs), down_msg)

      # Push the up message expression to the backward pass expression
      push!(backward_pass.args, down_msg)

      # # DEBUG:
      # println(down_msg)

    end

  end

  # # DEBUG:
  # map(bag -> bag.id , PreOrderDFS(root)) |> 
  #   x -> println("Backward pass visiting order: \n", x)

  # @show backward_pass

  return backward_pass

end

"""
$(TYPEDSIGNATURES)

Compile the forward and backward passes of messages.
"""
function compile_message_propagation!(g, root)

  # Initialize an empty vector of expressions of incoming messages for each bag
  map(bag_id -> set_prop!(g, bag_id, :in_msgs, Union{Expr, Factor}[]), vertices(g))

  forward_pass = compile_forward_pass!(g, root)
  backward_pass = compile_backward_pass!(g, root)

  return forward_pass, backward_pass

end

"""
$(TYPEDSIGNATURES)

Normalize messages in the propagation phase that cause an overflow when running
the expressions inside `operations`. Operations can be either sum-prod messages
from the propagation phase or edge/bag marginals computations.
"""
function normalize_messages(
                            obsvals,
                            pots,
                            before_pass_forward_msgs,
                            before_pass_backward_msgs,
                            operations,
                           )

  # Evaluate the observed values and the potentials in order to later evaluate operations that use them
  eval(:(obsvals = $obsvals))
  eval(pots)

  # Initialize the expression to be returend with a copy of the original messages
  after_pass_all_msgs  = Expr(:block, vcat(
                                           before_pass_forward_msgs.args,
                                           before_pass_backward_msgs.args,
                                          )...)

  for operation in operations.args

    # Eval the operation to check whether it causes an overflow
    eval(operation)

    # Parse the operation AND check whether it causes an overflow
    if @capture(operation, ivar_ = sum(prod(pargs1__), sargs__)) ||
      @capture(operation, ivar_ = prod(pargs1__))

      # If none of the values of the evaled msg is `Inf` then continue with the next msg
      !any(isinf, eval(ivar).vals) && continue

      @debug "Overflow occured in:\n\t$operation"

      # Yes, then find each msg in `pargs1` in `after_pass_all_msgs` and replace it in line
      for parg in pargs1
        # Iterate the `after_pass_all_msgs` to search and replace `parg` with its normalized version
        for i in eachindex(after_pass_all_msgs.args)
          # Only consider sum-prod messages
          if @capture(after_pass_all_msgs.args[i], imsg_ = sum(prod(pargs2__), sargs2__))
            if imsg == parg
              # Replace the msg with its normalized version in line
              after_pass_all_msgs.args[i] = :($imsg = sum(prod($(pargs2...)), $(sargs2...)) |> norm)
              # Re-eval the msg now that it is normalized version
              eval(after_pass_all_msgs.args[i]) 
              # Break from iterating through `after_pass_all_msgs`
              break
            end
          end
        end
      end

      # Re-eval the current operation now that its argument msgs have been normalized
      eval(operation)

      # Check if the overflow was fixed
      any(isinf, eval(ivar).vals) && @error "Overflow not corrected in:\n\t$operation"

    end

  end

  n_msgs_per_pass = length(before_pass_forward_msgs.args)
  idxs_forward_msgs = 1:n_msgs_per_pass
  idxs_backward_msgs = (n_msgs_per_pass + 1):(2 * n_msgs_per_pass)
  after_pass_forward_msgs = Expr(:block, after_pass_all_msgs.args[idxs_forward_msgs]...)
  after_pass_backward_msgs = Expr(:block, after_pass_all_msgs.args[idxs_backward_msgs]...)

  # DEBUG:

  # @show before_pass_forward_msgs
  # @show after_pass_forward_msgs

  # @show before_pass_backward_msgs
  # @show after_pass_backward_msgs

  return after_pass_forward_msgs, after_pass_backward_msgs

end
