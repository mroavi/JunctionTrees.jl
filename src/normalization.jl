"""
    compile_normalized_marginals(unnormalized_marginals)

Compile the normalized marginal expressions for each variable in the model.

"""
function compile_normalized_marginals(unnormalized_marginals)

  # Normalize all marginals
  normalize_marginals_expr =
    map(x -> x.args[1], unnormalized_marginals.args) |> # get the variable name
    x -> :(norm.([$(x...)])) # create an expression of vector form than normalizes each mar

  return normalize_marginals_expr

end

