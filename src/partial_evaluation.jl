@enum PartialEval EvalMsg DoNotEvalMsg EvalProductOnly

# TODO: If an observed var is present in all clusters then it will not get
# marginalized in any message computation and hence a reduction statement will
# never be introduced for it.
"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

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
        if @capture(fargs[1], prod(pargs__))
          # Yes, then evaluate it, wrap it in the marg operation and add the msg to the new expr
          prod_evaled = :(prod($(pargs...))) |> eval # eval the product 
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
      # DEBUG:
      # println("Before PE: ", before_pass_msg, "\n", "After  PE: ", after_pass_msg, "\n")
    end
  end
  return after_pass_msgs
end

"""
$(TYPEDSIGNATURES)

Mark which nodes of `g` have at least one observed variable.
"""
function mark_obsbags!(g, obsvars)

  # If any, store the bag's observed vars in a property
  for bag in vertices(g)
    bag_vars = get_prop(g, bag, :vars)
    bag_obsvars = intersect(bag_vars, obsvars)
    !isempty(bag_obsvars) && set_prop!(g, bag, :obsvars, bag_obsvars)
  end

  # DEBUG:
  # @show filter_vertices(g, :obsvars) |> collect # bags that contain at least one observed var

end

"""
$(TYPEDSIGNATURES)

Partially evaluates messages of the propagation stage that do not depend on
online observations.
"""
function partial_evaluation(td, pots, forward_pass, backward_pass)

  partial_eval_analysis!(td)

  # # DEBUG: Print `pre_eval_msg_` for each edge in forward pass order
  # for bag in PostOrderDFS(root)
  #   parent_bag = Base.parent(root, bag)
  #   isnothing(parent_bag) && break
  #   prop_name = Symbol("pre_eval_msg_", bag.id, "_", parent_bag.id)
  #   println(prop_name, ": ", get_prop(td, bag.id, parent_bag.id, prop_name))
  # end

  # # DEBUG: Print `pre_eval_msg_` for each edge in backward pass order
  # for bag in PreOrderDFS(root)
  #   for child in bag.children
  #     prop_name = Symbol("pre_eval_msg_", bag.id, "_", child.id)
  #     println(prop_name, ": ", get_prop(td, bag.id, child.id, prop_name))
  #   end
  # end

  # # DEBUG: print the isconsistent flag for each bag and edge
  # map(vertex -> ("Bag $vertex", get_prop(td, vertex, :isconsistent)), vertices(td)) |>
  #   x -> show(IOContext(stdout, :limit=>false), MIME"text/plain"(), x)
  # map(edge -> (edge, get_prop(td, edge, :isconsistent)), edges(td)) |>
  #   x -> show(IOContext(stdout, :limit=>false), MIME"text/plain"(), x)

  eval(pots)

  # @show forward_pass
  forward_pass = partially_evaluate(td, forward_pass)
  # @show forward_pass

  # @show backward_pass
  backward_pass = partially_evaluate(td, backward_pass)
  # @show backward_pass

  return forward_pass, backward_pass

end
