"""
A composite type that implements the factor datatype.
"""

struct Factor{T,N}
  vars::NTuple{N,Int64}
  vals::Array{T,N}
end

import Base: eltype
eltype(::Factor{T,N}) where {T,N} = T 

Base.show(io::IO, A::Factor) = print(io, "Factor", A.vars, "[...]")

"""
    product(A::Factor, B::Factor)

Compute a factor product of tables `A` and `B`.

# Examples
```julia
A = Factor{Float64,2}((2, 1), [0.5 0.1 0.03; 0.8 0.0 0.9])
B = Factor{Float64,2}((3, 2), [0.5 0.1; 0.7 0.2])
C = product(A, B)
```
"""
function product(A::Factor{T}, B::Factor{T}) where T
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

function _product(A_vals_new, B_vals_new, C_vars)
  Factor{Float64, length(C_vars)}(C_vars, A_vals_new .* B_vals_new)
end

function product(F::Vararg{Factor{T}}) where T
  reduce(product, F; init = Factor{T,0}((), Array{T,0}(undef)))
end

"""
    marg(A::Factor, V::Ntuple{N,Int64})

Sum out the variables inside `V` from factor A.

# Examples
```julia
A = Factor{Float64,3}((3, 2, 1), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
V = [1]
B = marg(A, V)

A = Factor{Float64,3}((2, 7, 1), cat([1 2; 3 4;  5  6],
                                     [7 8; 9 10; 11 12], dims=3))
V = [7]
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

function _marg(r_vals,ret_vars,Avals,dims)
  ret_vals = sum!(r_vals, Avals) |> x -> dropdims(x, dims=Tuple(dims))
  Factor{eltype(Avals),length(ret_vars)}(ret_vars, ret_vals)
end

"""
    redu(A::Factor{T}, vars::Tuple, vals::Tuple) where T

Reduce/invalidate all entries that are not consitent with the evidence.

# Examples
```julia
A = Factor{Float64,3}((1, 2, 3), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
vars = (1,3)
vals = (1,1)
B = redu(A, vars, vals)
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

