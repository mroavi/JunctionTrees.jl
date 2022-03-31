"""
Enumerated type used to select up to which stage an expression of the junction
tree algorithm should be returned.
"""
@enum LastStage begin
  ForwardPass
  BackwardPass
  JointMarginals
  UnnormalizedMarginals
  Marginals
end
@doc "Return an expression up to and including the forward pass." ForwardPass
@doc "Return an expression up to and including the backward pass." BackwardPass
@doc "Return an expression that computes the cluster joint marginals." JointMarginals
@doc "Return an expression that computes the joint marginals." UnnormalizedMarginals
@doc "Return an expression that computes the posterior marginals." Marginals

"""
$(TYPEDSIGNATURES)

Return an expression containing the computations to compute the marginals of
all the variables in the model using the junction tree algorithm.

"""
function compile_algo(uai_filepath::String;
                      uai_evid_filepath::String = "",
                      td_filepath::String = "",
                      apply_partial_evaluation::Bool = false,
                      last_stage::LastStage = Marginals,
                      smart_root_selection::Bool = true,
                     )

  # Read PGM
  nvars, cards, _, factors = read_uai_file(uai_filepath)
  obsvars, obsvals = read_uai_evid_file(uai_evid_filepath)

  # Graphical transformation
  if isempty(td_filepath)
    mrf = construct_mrf_graph(nvars, factors)
    td = construct_td_graph(mrf, cards)
  else
    td = construct_td_graph(td_filepath)
  end

  # Observation entry
  mark_obsbags!(td, obsvars)

  # Initialization
  root, pots = initialize_td_graph!(td, factors, smart_root_selection)

  # Propagation
  forward_pass, backward_pass = compile_message_propagation!(td, root)

  # Partial evaluation
  if apply_partial_evaluation
    forward_pass, backward_pass = partial_evaluation(td, pots, forward_pass, backward_pass)
  end

  # Observation entry
  if apply_partial_evaluation
    pots, forward_pass, backward_pass = inject_redus(td, pots, forward_pass, backward_pass, obsvars, obsvals)
  else
    pots = inject_redus(td, pots, obsvars, obsvals)
  end

  # Marginalization
  edge_marginals, bag_marginals, unnormalized_marginals = compile_unnormalized_marginals(td, nvars, apply_partial_evaluation)

  # Normalization
  normalize_marginals_expr = compile_normalized_marginals(unnormalized_marginals)

  # ==============================================================================
  ## Concatenate the steps of the Junction tree algorithm into one expression
  # ==============================================================================

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
  #   set_prop!(td, Edge(bag.id, parent_bag.id), :up_msg_order, i)
  # end
  # for (i, bag) in enumerate(PreOrderDFS(root))
  #   for (j, child) in enumerate(bag.children)
  #     set_prop!(td, Edge(bag.id, child.id), :down_msg_order, i+j-1)
  #   end
  # end

  # # DEBUG:
  # println(algo)

  # # DEBUG:
  # @btime eval(algo)

  function_name = :run_algo
  sig = (Vector{Int64}, Vector{Int64})
  variables = [:obsvars, :obsvals]
  body = algo

  return generate_function_expression(function_name, sig, variables, body)

end
