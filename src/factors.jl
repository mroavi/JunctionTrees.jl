"""
$(TYPEDEF)

# Fields
- `vars`
- `vals`

Encodes a discrete function over the set of variables `vars` that maps each
instantiation of `vars` into a nonnegative number in `vals`.
"""
struct Factor{T,N}
  vars::NTuple{N,Int64}
  vals::Array{T,N}
end

import Base: eltype
eltype(::Factor{T,N}) where {T,N} = T 

"""
$(TYPEDSIGNATURES)

Compute a factor product of tables `A` and `B`.

# Examples
```jldoctest
A = Factor{Float64,2}((2, 3), [0.5 0.7; 0.1 0.2])
B = Factor{Float64,2}((1, 2), [0.5 0.8; 0.1 0.0; 0.3 0.9])
prod(A, B)

# output

Factor{Float64, 3}((1, 2, 3), [0.25 0.08000000000000002; 0.05 0.0; 0.15 0.09000000000000001;;; 0.35 0.16000000000000003; 0.06999999999999999 0.0; 0.21 0.18000000000000002])
```
"""
function prod(A::Factor{T}, B::Factor{T}) where T
  isempty(A.vars) && return Factor(B.vars, B.vals)
  isempty(B.vars) && return Factor(A.vars, A.vals)
  A_card = collect(size(A.vals))
  B_card = collect(size(B.vals))
  C_vars = union(A.vars, B.vars) |> sort
  A_card_new = ones(Int64, length(C_vars))
  B_card_new = ones(Int64, length(C_vars))
  for (i,C_var) in enumerate(C_vars)
    C_var in A.vars && (A_card_new[i] = popfirst!(A_card))
    C_var in B.vars && (B_card_new[i] = popfirst!(B_card))
  end
  A_vals_new = reshape(A.vals, Tuple(A_card_new))
  B_vals_new = reshape(B.vals, Tuple(B_card_new))
  _product(A_vals_new, B_vals_new, Tuple(C_vars))
end

"""
$(TYPEDSIGNATURES)

Performance critical code of the factor product operation.
"""
function _product(A_vals_new, B_vals_new, C_vars)
  Factor{eltype(A_vals_new), length(C_vars)}(C_vars, A_vals_new .* B_vals_new)
end

"""
$(TYPEDSIGNATURES)

Compute a factor product of an arbitrary number of factors.
"""
function prod(F::Vararg{Factor{T}}) where T
  reduce(prod, F; init = Factor{T,0}((), Array{T,0}(undef)))
end

"""
$(TYPEDSIGNATURES)

Sum out the variables in `V` from factor A.

# Examples
```jldoctest
A = Factor{Float64,2}((1, 2), [0.59 0.41; 0.22 0.78])
sum(A, (2,))

# output

Factor{Float64, 1}((1,), [1.0, 1.0])
```
"""
function sum(A::Factor{T,ND}, V::NTuple{N,Int64} where N) where {T,ND}
  ND == 0 && return Factor{eltype(A),0}((), Array{eltype(A),0}(undef))
  dims = my_indexin(V, A.vars) # map vars to dims
  r_size = ntuple(d->d in dims ? 1 : size(A.vals,d), ND) # assign 1 to summed out dims
  ret_vars = filter(!in(V), A.vars)
  r_vals = similar(A.vals, r_size)
  _marg(r_vals, ret_vars, A.vals,dims)
end

"""
$(TYPEDSIGNATURES)

Sum out an arbitrary number of variables from factor A.

# Examples
```jldoctest
A = Factor{Float64,3}((1, 2, 3), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
sum(A, 1, 2)

# output

Factor{Float64, 1}((3,), [0.6199999999999999, 0.97])
```
"""
sum(A::Factor, V::Int...) = sum(A, V)

"""
$(TYPEDSIGNATURES)

Optimized version of the function `indexin` defined in `Base`.
"""
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

"""
$(TYPEDSIGNATURES)

Performance critical code of the factor marginalization operation.
"""
function _marg(r_vals,ret_vars,Avals,dims)
  ret_vals = sum!(r_vals, Avals) |> x -> dropdims(x, dims=Tuple(dims))
  Factor{eltype(Avals),length(ret_vars)}(ret_vars, ret_vals)
end

"""
$(TYPEDSIGNATURES)

Reduce/invalidate all entries in `A` that are not consitent with the evidence
passed in `vars` and `vals`, where each variable in `vars` is assigned the
corresponding value in `vals`.

# Examples
```jldoctest
A = Factor{Float64,3}((1, 2, 3), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
obs_vars = (3,)
obs_vals = (1,)
redu(A, obs_vars, obs_vals)

# output

Factor{Float64, 3}((1, 2, 3), [0.25 0.08; 0.05 0.0; 0.15 0.09;;; 0.0 0.0; 0.0 0.0; 0.0 0.0])
```
"""
function redu(A::Factor{T}, vars::Tuple, vals::Tuple) where T
  mapVars = indexin(vars, collect(A.vars))
  indxValidTuple = ntuple(i -> (i in mapVars) ? vals[indexin(i, mapVars)...] : Colon(), length(A.vars))
  indxValid = CartesianIndices(A.vals)[indxValidTuple...]
  B_vals = zeros(size(A.vals))
  B_vals[indxValid] = A.vals[indxValid]
  return Factor{T,length(A.vars)}(A.vars, B_vals)
end

"""
$(TYPEDSIGNATURES)

Normalize the values in Factor A such they sum up to 1.

# Examples
```jldoctest
A = Factor{Float64,2}((1, 2), [0.2 0.4; 0.6 0.8])
norm(A)

# output

Factor{Float64, 2}((1, 2), [0.1 0.2; 0.3 0.4])
```
"""
function norm(A::Factor)
  return Factor(A.vars, A.vals ./ sum(A.vals))
end
