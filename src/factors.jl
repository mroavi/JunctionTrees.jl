"""
A composite type that implements the factor datatype.
"""
struct Factor
  vars::Array{Int64}
  vals::Array{Float64}
end

"""
    product(A::Factor, B::Factor)

Compute a factor product of tables `A` and `B`.

# Examples
```julia
A = Factor([2; 1], [0.5 0.1 0.03; 0.8 0.0 0.9])
B = Factor([3; 2], [0.5 0.1; 0.7 0.2])
C = product(A, B)
```
"""
function product(A::Factor, B::Factor)

  isempty(A.vars) && return Factor(B.vars, B.vals)
  isempty(B.vars) && return Factor(A.vars, A.vals)

  C_vars = union(A.vars, B.vars)
  mapA = indexin(A.vars, C_vars)
  mapB = indexin(B.vars, C_vars)
  C_card = zeros(Int64, length(C_vars))
  C_card[mapA] = size(A.vals) |> collect
  C_card[mapB] = size(B.vals) |> collect
  C_vals = zeros(eltype(A.vals), Tuple(C_card))
  assignments = CartesianIndices(C_vals)
  indxA = [CartesianIndex(i.I[mapA]) for i in assignments] # extract `mapA` columns
  indxB = [CartesianIndex(i.I[mapB]) for i in assignments] # extract `mapB` columns
  C_vals = A.vals[indxA] .* B.vals[indxB]
  return Factor(C_vars, C_vals)
end

product(F::Array{Factor}) = reduce(product, F; init = Factor(Int64[], Float64[]))
product(F::Factor...) = product(Factor[F...])

"""
    marg(A::Factor, V::Array{Int64})

Sum out the variables inside `V` from factor A.
Based on an assignment of the coursera course Probabilistic Graphical Models.
https://www.coursera.org/learn/probabilistic-graphical-models/home/week/1

# Examples
```julia
A = Factor([3; 2; 1], cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                          [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
V = [1]
B = marg(A, V)

A = Factor([2; 7; 1], cat([1 2; 3 4;  5  6],
                          [7 8; 9 10; 11 12], dims=3))
V = [7]
B = marg(A, V)
```
"""
function marg(A::Factor, V::Array{Int64})
  B_vars = setdiff(A.vars, V)
  mapB = indexin(B_vars, A.vars)
  B_card = size(A.vals)[mapB]
  S_vars = intersect(A.vars, V)
  mapS = indexin(S_vars, A.vars)
  B_vals = sum(A.vals, dims=mapS) |> x -> dropdims(x, dims=Tuple(mapS))
  return Factor(B_vars, B_vals)
end

"""
    redu(A::Factor, var::Int64, val::Int64)

Reduce/invalidate all entries that are not consitent with the evidence.

# Examples
```julia
A = Factor([1; 2; 3], cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                          [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
var = 3
val = 1
B = redu(A, var, val)
```
"""
function redu(A::Factor, var::Int64, val::Int64)
  B_vars = setdiff(A.vars, var)
  mapB = indexin(B_vars, A.vars)
  B_card = size(A.vals)[mapB]
  R_vars = intersect(A.vars, var)
  mapR = indexin(R_vars, A.vars)
  indxA = ntuple(i -> (i == mapR[1]) ? val : :, length(A.vars))
  indxA = CartesianIndices(A.vals)[indxA...] # here occurs the actual reduction
  B_vals = A.vals[indxA]
  return Factor(B_vars, A.vals[indxA])
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

