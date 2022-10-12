"""
Enumerated type used to select up to which stage an expression of the junction
tree algorithm should be returned after calling [`compile_algo`](@ref).
"""
@enum LastStage begin
  ForwardPass
  BackwardPass
  JointMarginals
  UnnormalizedMarginals
  Marginals
end
@doc "When assigned to the keyword argument `last_stage` of [`compile_algo`](@ref), an expression up to and including the forward pass is returned." ForwardPass
@doc "When assigned to the keyword argument `last_stage` of [`compile_algo`](@ref), an expression up to and including the backward pass is returned." BackwardPass
@doc "When assigned to the keyword argument `last_stage` of [`compile_algo`](@ref), an expression that computes the cluster joint marginals is returned." JointMarginals
@doc "When assigned to the keyword argument `last_stage` of [`compile_algo`](@ref), an expression that computes the joint marginals is returned." UnnormalizedMarginals
@doc "When assigned to the keyword argument `last_stage` of [`compile_algo`](@ref), an expression that computes the posterior marginals is returned (default)." Marginals

"""
$(TYPEDSIGNATURES)

Return an expression of the junction tree algorithm that extracts the marginals
of all the variables in the model.

# Arguments
- `uai_filepath::AbstractString`: path to the model file defined in the [UAI model file format](@ref).
- `uai_evid_filepath::AbstractString = ""`: path to the evidence file defined in the [UAI evidence file format](@ref).
- `td_filepath::AbstractString = ""`: path to a pre-constructed junction tree defined in the [PACE graph format](@ref).
- `apply_partial_evaluation::Bool = false`: optimize the algorithm using partial evaluation.
- `last_stage::LastStage = Marginals`: return an expression up to the given stage. The options are `ForwardPass`, `BackwardPass`, `JointMarginals`, `UnnormalizedMarginals` and `Marginals`.
- `smart_root_selection::Bool = true`: select as root the cluster with the largest state space.
- `factor_eltype::DataType = Float64`: type used to represent the factor values. 
- `use_omeinsum::Bool = false`: use the OMEinsum tensor network contraction package as backend for the factor operations.

# Examples
```
package_root_dir = pathof(JunctionTrees) |> dirname |> dirname
uai_filepath = joinpath(package_root_dir, "docs", "src", "problems", "paskin", "paskin.uai")
algo = compile_algo(uai_filepath)
eval(algo)
obsvars, obsvals = Int64[], Int64[]
marginals = run_algo(obsvars, obsvals)

# output

6-element Vector{Factor{Float64, 1}}:
 Factor{Float64, 1}((1,), [0.33480077287635474, 0.33039845424729053, 0.33480077287635474])
 Factor{Float64, 1}((2,), [0.378700415763991, 0.621299584236009])
 Factor{Float64, 1}((3,), [0.3632859624875086, 0.6367140375124913])
 Factor{Float64, 1}((4,), [0.6200692707149191, 0.37993072928508087])
 Factor{Float64, 1}((5,), [0.649200314859223, 0.350799685140777])
 Factor{Float64, 1}((6,), [0.5968155611613972, 0.4031844388386027])
```

```
package_root_dir = pathof(JunctionTrees) |> dirname |> dirname
uai_filepath = joinpath(package_root_dir, "docs", "src", "problems", "paskin", "paskin.uai")
uai_evid_filepath = joinpath(package_root_dir, "docs", "src", "problems", "paskin", "paskin.uai.evid")
algo = compile_algo(
         uai_filepath,
         uai_evid_filepath = uai_evid_filepath)
eval(algo)
obsvars, obsvals = JunctionTrees.read_uai_evid_file(uai_evid_filepath)
marginals = run_algo(obsvars, obsvals)

# output

6-element Vector{Factor{Float64, 1}}:
 Factor{Float64, 1}((1,), [1.0, 0.0, 0.0])
 Factor{Float64, 1}((2,), [0.0959432982733719, 0.9040567017266281])
 Factor{Float64, 1}((3,), [0.07863089300137578, 0.9213691069986242])
 Factor{Float64, 1}((4,), [0.8440129077674895, 0.15598709223251056])
 Factor{Float64, 1}((5,), [0.9015456486772953, 0.09845435132270475])
 Factor{Float64, 1}((6,), [0.6118571666785584, 0.3881428333214415])
```
"""
function compile_algo(uai_filepath::AbstractString;
                      uai_evid_filepath::AbstractString = "",
                      td_filepath::AbstractString = "",
                      apply_partial_evaluation::Bool = false,
                      last_stage::LastStage = Marginals,
                      smart_root_selection::Bool = true,
                      factor_eltype::DataType = Float64,
                      use_omeinsum::Bool = false,
                     )

  # Read PGM
  nvars, cards, _, factors = read_uai_file(uai_filepath, factor_eltype = factor_eltype)
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

  # TODO: add normalization statements only when necessary
  if true
    # Normalize the messages that contain a product
    # (this is to avoid underflows in large problems)
    forward_pass = normalize_messages(forward_pass)
    backward_pass = normalize_messages(backward_pass)
  end

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

  ret = generate_function_expression(function_name, sig, variables, body)

  # OMEinsum
  if use_omeinsum
    ret = boost_algo(ret)
  end

  return ret

end
