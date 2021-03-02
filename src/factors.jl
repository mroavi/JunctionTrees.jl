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

  C_vars = union(A.vars, B.vars)
  mapA = indexin(A.vars, C_vars)
  mapB = indexin(B.vars, C_vars)
  C_card = zeros(Int64, length(C_vars))
  C_card[mapA] = size(A.vals) |> collect
  C_card[mapB] = size(B.vals) |> collect
  C_vals = zeros(eltype(A), Tuple(C_card))
  assignments = CartesianIndices(C_vals)
  indxA = [CartesianIndex(i.I[mapA]) for i in assignments] # extract `mapA` columns
  indxB = [CartesianIndex(i.I[mapB]) for i in assignments] # extract `mapB` columns
  C_vals = A.vals[indxA] .* B.vals[indxB]
  return Factor{T,length(C_vars)}(Tuple(C_vars), C_vals)
end

function product(F::AbstractArray{<:Factor{T,N} where N, 1}) where T
  reduce(product, F; init = Factor{T,0}((), Array{T,0}(undef)))
end

product(F::Factor{T}...) where {T} = product(Factor{T}[F...])

"""
    marg(A::Factor, V::Vector{Int64})

Sum out the variables inside `V` from factor A.
Based on an assignment of the coursera course Probabilistic Graphical Models.
https://www.coursera.org/learn/probabilistic-graphical-models/home/week/1

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
function marg(A::Factor{T}, V::Vector{Int64}) where T
  dims = indexin(V, collect(A.vars)) # map vars to dims
  r_size = ntuple(d->d in dims ? 1 : size(A.vals,d), length(A.vars)) # assign 1 to summed out dims
  ret_size = filter(s -> s != 1, r_size)
  ret_vars = filter(v -> v ∉ V, A.vars)
  r_vals = similar(A.vals, r_size)
  ret_vals = sum!(r_vals, A.vals) |> x -> dropdims(x, dims=Tuple(dims))
  Factor{eltype(A.vals),length(ret_vars)}(ret_vars, ret_vals)
end
marg(A::Factor, V::Int) = marg(A, [V])

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

