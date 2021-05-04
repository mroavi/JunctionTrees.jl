"""
A composite type that implements the factor datatype.
"""

mutable struct Factor{T,N}
  vars::NTuple{N,Int64}
  vals::Array{T,N}
end

import Base: eltype
eltype(::Factor{T,N}) where {T,N} = T 

"""
    product(in_factors::Factor{T}...) where T

Compute a factor product of all tables contained in `in_factors`.

# Examples
```julia
A = Factor{Float64,2}((2, 1), [0.5 0.1 0.03; 0.8 0.0 0.9])
B = Factor{Float64,2}((3, 2), [0.5 0.1; 0.7 0.2])
C = product(A, B)
```
"""

function product(in_factors::Factor{T}...) where T
  in_factors_card = map(x -> collect(size(x.vals)), in_factors)
  out_factor_vars = map(x -> x.vars, in_factors) |> x -> union(x...) |> sort |> Tuple
  in_factors_card_new = map(x -> ones(Int64, length(out_factor_vars)), in_factors)
  for (i, out_factor_var) in enumerate(out_factor_vars)
    for (j, in_factor_vars) in enumerate(map(x -> x.vars, in_factors))
      out_factor_var in in_factor_vars && (in_factors_card_new[j][i] = popfirst!(in_factors_card[j]))
    end
  end
  in_factors_vals = map(in_factor -> in_factor.vals, in_factors)
  in_factors_vals_new = map((x, y) -> reshape(x, Tuple(y)), in_factors_vals, in_factors_card_new)
  out_factor_card = hcat(in_factors_card_new...) |> x -> maximum(x, dims=2)
  out_factor_new = Factor{T, length(out_factor_vars)}(out_factor_vars, zeros(out_factor_card...))
  _product!(out_factor_new, in_factors_vals_new)
  return out_factor_new
end

function _product!(out_factor, in_factors_vals)
  out_factor.vals .= .*(in_factors_vals...)
end

"""
  init_prod(in_factors::Factor{T}...) where T

Compute the scope and card of the resulting factor after performing a factor 
product between the factors contained in `in_factors`. Allocate memory for the 
values of the resulting factor.

# Example
```julia
A = Factor{Float64,2}((2, 3), [0.5 0.7; 0.1 0.2])
B = Factor{Float64,2}((1, 2), [0.5 0.8; 0.1 0.0; 0.3 0.9])
C = Factor{Float64,3}((1, 2, 3), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
in_factors = (A,B,C)
D = init_prod(in_factors...)
```
"""
function init_prod(in_factors::Factor{T}...) where T
  in_factors_card = map(x -> collect(size(x.vals)), in_factors)
  out_factor_vars = map(x -> x.vars, in_factors) |> x -> union(x...) |> sort |> Tuple
  in_factors_card_new = map(x -> ones(Int64, length(out_factor_vars)), in_factors)
  for (i, out_factor_var) in enumerate(out_factor_vars)
    for (j, in_factor_vars) in enumerate(map(x -> x.vars, in_factors))
      out_factor_var in in_factor_vars && (in_factors_card_new[j][i] = popfirst!(in_factors_card[j]))
    end
  end
  in_factors_vals = map(in_factor -> in_factor.vals, in_factors)
  in_factors_vals_new = map((x, y) -> reshape(x, Tuple(y)), in_factors_vals, in_factors_card_new)
  out_factor_card = hcat(in_factors_card_new...) |> x -> maximum(x, dims=2)
  Factor{T, length(out_factor_vars)}(out_factor_vars, zeros(out_factor_card...)), Tuple.(in_factors_card_new)
end

"""
TODO:
"""
function find_common_shape(in_factors::Factor...)
  in_factors_card = map(x -> collect(size(x.vals)), in_factors)
  out_factor_vars = map(x -> x.vars, in_factors) |> x -> union(x...) |> sort |> Tuple
  in_factors_card_new = map(x -> ones(Int64, length(out_factor_vars)), in_factors)
  for (i, out_factor_var) in enumerate(out_factor_vars)
    for (j, in_factor_vars) in enumerate(map(x -> x.vars, in_factors))
      out_factor_var in in_factor_vars && (in_factors_card_new[j][i] = popfirst!(in_factors_card[j]))
    end
  end
  return Tuple.(in_factors_card_new)
end

function product!(out_factor, in_factors_card_new, in_factors::Tuple)
  in_factors_vals = map(in_factor -> in_factor.vals, in_factors)
  in_factors_vals_new = map((x, y) -> reshape(x, y), in_factors_vals, in_factors_card_new)
  out_factor.vals .= .*(in_factors_vals_new...)
end

"""
    marg(A::Factor, V::Ntuple{N,Int64})

Sum out the variables inside `V` from factor A.

# Example
```julia
A = Factor{Float64,3}((1, 2, 3), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
V = [1]
B = marg(A, V)
```
"""
function marg(A::Factor{T,ND}, V::NTuple{N,Int64} where N) where {T,ND}
  dims = my_indexin(V, A.vars) # map vars to dims
  r_size = ntuple(d->d in dims ? 1 : size(A.vals,d), ND) # assign 1 to summed out dims
  ret_vars = filter(!in(V), A.vars)
  r_vals = similar(A.vals, r_size)
  _marg(r_vals, ret_vars, A.vals,dims)
end
marg(A::Factor, V::Int...) = marg(A, V)

function my_indexin(x,y)
  indxs = Vector{eltype(x)}(undef, length(x))
  curr = 1
  for xval in x 
    for (j, yval) in pairs(y)
      if yval == xval
        indxs[curr] = j
        curr += 1
        break
      end
    end
  end
  indxs
end

function _marg(r_vals, ret_vars, Avals, dims)
  ret_vals = sum!(r_vals, Avals) |> x -> dropdims(x, dims=Tuple(dims))
  Factor{eltype(Avals),length(ret_vars)}(ret_vars, ret_vals)
end

"""
  init_marg(A::Factor{T,ND}, V::NTuple{N,Int64} where N) where {T,ND}

Compute the scope and card of the resulting factor after marginalizing the
variables in `V` from the factor `A`. Allocate memory for the values
of the resulting factor.

# Examples
```julia
A = Factor{Float64,3}((2, 4, 7), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
V = (4,)
(B, r_vals, dims_to_drop) = init_marg(A, V)
```
"""
function init_marg(A::Factor{T,ND}, V::NTuple{N,Int64} where N) where {T,ND}
  dims_to_drop = my_indexin(V, A.vars) # map vars to dims
  r_size = ntuple(d -> d in dims_to_drop ? 1 : size(A.vals,d), ND) # assign 1 to summed out dims
  ret_vars = filter(!in(V), A.vars)
  ret_card = filter(!isone, r_size)
  r_vals = similar(A.vals, r_size)
  Factor{T,length(ret_vars)}(ret_vars, zeros(ret_card...)), r_vals, Tuple(dims_to_drop)
end
init_marg(A::Factor, V::Int...) = init_marg(A, V)

"""
    marg!(in::Factor, r_vals, drop_dims, out::Factor)

Sum out the variables inside `drop_dims` from factor `in` and store the
result in factor `out`.

# Example
```julia
A = Factor{Float64,3}((1, 2, 3), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
V = [1]
B = marg(A, V)
```
"""
function marg!(out::Factor, in::Factor, r_vals, dims)
  out.vals = sum!(r_vals, in.vals) |> x -> dropdims(x, dims=dims)
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
A = Factor([1, 2], [1.0 2.0 3.0; 4.0 5.0 6.0])
B = norm(A)
```
"""
function norm(A::Factor)
  return Factor(A.vars, A.vals ./ sum(A.vals))
end

