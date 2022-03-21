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
function computeMarginalsExpr(td_filepath::String,
                              uai_filepath::String,
                              uai_evid_filepath::String = ""; 
                              partial_evaluation::Bool = false,
                              last_stage::LastStage = Marginals,
                              smart_root_selection::Bool = true,
                             )

  g = construct_td_graph(td_filepath)
  obsvars, obsvals = read_uai_evid_file(uai_evid_filepath)
  mark_obsbags!(g, obsvars)
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

    # @show forward_pass
    forward_pass = partially_evaluate(g, forward_pass)
    # @show forward_pass

    # @show backward_pass
    backward_pass = partially_evaluate(g, backward_pass)
    # @show backward_pass

  end

  # ==============================================================================
  ## Inject reduce statements
  # ==============================================================================

  if partial_evaluation

    # @show pots
    pots = inject_redus_in_pots(g, pots, obsvars, obsvals)
    # @show pots

    # @show forward_pass
    forward_pass = inject_redus_in_msgs(g, forward_pass, obsvars, obsvals)
    # @show forward_pass

    # @show backward_pass
    backward_pass = inject_redus_in_msgs(g, backward_pass, obsvars, obsvals)
    # @show backward_pass

  else

    # @show pots
    pots = inject_redus(g, pots, obsvars, obsvals)
    # @show pots

  end

  edge_marginals, bag_marginals, unnormalized_marginals = compile_unnormalized_marginals(g, nvars, partial_evaluation)
  normalize_marginals_expr = compile_normalized_marginals(unnormalized_marginals)

  # ==============================================================================
  ## Finalize algorithm
  # ==============================================================================

  # Concatenate the different expressions corresponding to the Junction Tree algo steps
  if last_stage == ForwardPass
    algo = Expr(:block, vcat(pots,
                             forward_pass,
                            )...)
  elseif last_stage == BackwardPass
    algo = Expr(:block, vcat(pots,
                             forward_pass,
                             backward_pass,
                            )...)
  elseif last_stage == JointMarginals
    algo = Expr(:block, vcat(pots,
                             forward_pass,
                             backward_pass,
                             edge_marginals,
                             bag_marginals,
                            )...)
  elseif last_stage == UnnormalizedMarginals
    algo = Expr(:block, vcat(pots,
                             forward_pass,
                             backward_pass,
                             edge_marginals,
                             bag_marginals,
                             unnormalized_marginals,
                            )...)
  elseif last_stage == Marginals
    algo = Expr(:block, vcat(pots,
                             forward_pass,
                             backward_pass,
                             edge_marginals,
                             bag_marginals,
                             unnormalized_marginals,
                             Expr(:block, normalize_marginals_expr),
                            )...)
  end

  # # DEBUG: save the message order in edge property
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

  # # DEBUG:
  # println(algo)

  # # DEBUG:
  # @btime eval(algo)

  function_name = :compute_marginals
  sig = (Tuple, Tuple)
  variables = [:obsvars, :obsvals]
  body = algo

  return generate_function_expression(function_name, sig, variables, body)

end

