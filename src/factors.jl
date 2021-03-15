"""
A composite type that implements the factor datatype.
"""

mutable struct Factor{T, N}
  vars::NTuple{N,Int64}
  vals::Array{T,N}
end

import Base: eltype
eltype(::Factor{T,N}) where {T,N} = T 

"""
    prepare_factor_prod(in_factors::Factor...)

Prepares input factors for a subsequent factor product by
assigning 1s to dimensions that correspond to variables
that are present in other factors. This converts the factors
to a common size which allows a subsequent broadcast
multiplication.

# Examples
```julia
A = Factor{Float64,2}((1, 2), [0.5 0.1 0.03; 0.8 0.0 0.9])
B = Factor{Float64,2}((1, 3), [0.5 0.1; 0.7 0.2])
in_factors_new = prepare_factor_prod(A, B)
```
"""
function prepare_factor_prod(in_factors::Factor{T}...) where T
  in_factors_card = map(x -> collect(size(x.vals)), in_factors)
  out_factor_vars = map(x -> x.vars, in_factors) |> x -> union(x...) |> sort |> Tuple
  out_factors_card = map(x -> ones(Int64, length(out_factor_vars)), in_factors)
  for (i, out_factor_var) in enumerate(out_factor_vars)
    for (j, in_factor_vars) in enumerate(map(x -> x.vars, in_factors))
      out_factor_var in in_factor_vars && (out_factors_card[j][i] = popfirst!(in_factors_card[j]))
    end
  end
  in_factors_vals = map(in_factor -> in_factor.vals, in_factors)
  out_factors_vals = map((x, y) -> reshape(x, Tuple(y)), in_factors_vals, out_factors_card)
  out_factors = map(x -> Factor{T, length(out_factor_vars)}(out_factor_vars, x), out_factors_vals)
  return out_factors
end

prepare_factor_prod(in_factor::Factor) = (in_factor,)
prepare_factor_prod() = ()

"""
    factor_prod(in_factors::Factor...)

Compute a factor product of all input factor arguments.

# Examples
```julia
A = Factor{Float64,2}((1, 3), [0.5 0.1 0.03; 0.8 0.0 0.9])
B = Factor{Float64,2}((1, 2), [0.5 0.1; 0.7 0.2])
in_factors_new = prepare_factor_prod(A, B)
C = factor_prod(in_factors_new...)
```
"""
function factor_prod(in_factors::Factor{T}...) where T
  out_factor_vals = 
    map(in_factor -> in_factor.vals, in_factors) |>
    in_factor_vals -> reduce((x,y) -> x .* y, in_factor_vals)
  Factor{T, length(in_factors[1].vars)}(in_factors[1].vars, out_factor_vals)
end

# TODO: do not hardcode the Float64s
factor_prod() = Factor{Float64, 0}((), Array{Float64,0}(undef))

"""
  prepare_factor_marg(A::FactorQ, vars::Tuple)

Returns the dimensions that correspond to the summed out variables in `vars`,
the variables (and cardinality) of the resulting factor, and the cardinality of
the intermediary array needed to perform the sum with the builtin `sum!`
function.

# Examples
```julia
A = Factor{Float64,3}((3, 2, 1), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
dims, B_vars, r_card = prepare_factor_marg(A, (1,2))
```
"""
function prepare_factor_marg(A::Factor, vars::Tuple)
  dims = indexin(vars, collect(A.vars)) |> Tuple # map vars to dims
  r_card = ntuple(d -> d in dims ? 1 : size(A.vals, d), length(A.vars)) # assign 1 to summed out dims
  B_vars = filter(v -> v ∉ vars, A.vars)
  return dims, B_vars, r_card
end

"""
  compute_factor_form_after_marg(A::Factor, vars::Tuple)

Returns the resulting factor form after marginalizing `vars` from factor `A`.
"""
function compute_factor_form_after_marg(A::Factor, vars::Tuple)
  B_vars = filter(v -> v ∉ vars, A.vars)
  mapB = indexin(B_vars, collect(A.vars))
  B_card = size(A.vals)[mapB]
  Factor{eltype(A),length(B_vars)}(B_vars, zeros(eltype(A), B_card...))
end

"""
    factor_marg(A::Factor, dims, B_vars, r_card)

Sum values over each of the dimenstions in `dims`.

# Examples
```julia
A = Factor{Float64,3}((3, 2, 1), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
dims, B_vars, r_card = prepare_factor_marg(A, (1,2))
factor_marg(A, dims, B_vars, r_card) 
```
"""
function factor_marg(A::Factor, dims, B_vars, r_card)
  B_vals = sum!(similar(A.vals, r_card), A.vals) |> x -> dropdims(x, dims=dims)
  Factor{eltype(A),length(B_vars)}(B_vars, B_vals)
end

"""
    redu(A::Factor, var::Int64, val::Int64)

Reduce/invalidate all entries that are not consitent with the evidence.

# Examples
```julia
A = Factor{Float64,3}((1, 2, 3), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
var = 3
val = 1
B = redu(A, var, val)
```
"""
function redu(A::Factor{T}, var::Int64, val::Int64) where T
  B_vars = setdiff(A.vars, var)
  mapB = indexin(B_vars, collect(A.vars))
  B_card = size(A.vals)[mapB]
  R_vars = intersect(A.vars, var)
  mapR = indexin(R_vars, collect(A.vars))
  indxA = ntuple(i -> (i == mapR[1]) ? val : :, length(A.vars))
  indxA = CartesianIndices(A.vals)[indxA...] # here occurs the actual reduction
  B_vals = A.vals[indxA]
  return Factor{T,length(B_vars)}(Tuple(B_vars), A.vals[indxA])
end

"""
    norm(A::Factor)

Normalize the values in Factor A such that all probabilities lie between 0
and 1.

# Examples
```julia
A = Factor([2, 1], [1.0 2.0 3.0; 4.0 5.0 6.0])
B = norm(A)
```
"""
function norm(A::Factor)
  return Factor(A.vars, A.vals ./ sum(A.vals))
end

