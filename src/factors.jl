"""
A composite type that implements the factor datatype.
"""

struct Factor{V,C,T}
  vals::T
end

getvars(::Factor{V}) where {V} = V
getcard(::Factor{V,C}) where {V,C} = C
import Base: eltype, ndims, size
ndims(::Factor{V}) where {V} = length(V)
eltype(::Factor{V,C,T}) where {V,C,T} = eltype(T)

"""
    product(A::Factor, B::Factor)

Compute a factor product of tables `A` and `B`.

# Examples
```julia
A = Factor{(1,2), (2,3), Array{Float64,2}}([0.5 0.1 0.3; 0.8 0.0 0.9])
B = Factor{(1,3), (2,2), Array{Float64,2}}([0.5 0.1; 0.7 0.2])
C = product(A, B)
```
"""
function product(A::Factor, B::Factor)
  A_vars = getvars(A)
  A_card = getcard(A) |> collect
  B_vars = getvars(B)
  B_card = getcard(B) |> collect
  isempty(A_vars) && return B
  isempty(B_vars) && return A
  C_vars = union(A_vars, B_vars) |> sort |> Tuple
  A_card_new = ones(Int64, length(C_vars))
  B_card_new = ones(Int64, length(C_vars))
  for (i,C_var) in enumerate(C_vars)
    C_var in A_vars && (A_card_new[i] = popfirst!(A_card))
    C_var in B_vars && (B_card_new[i] = popfirst!(B_card))
  end
  C_card = hcat(A_card_new, B_card_new) |> x -> maximum(x, dims=2) |> x -> dropdims(x, dims=2) |> Tuple
  A_vals_new = reshape(A.vals, Tuple(A_card_new))
  B_vals_new = reshape(B.vals, Tuple(B_card_new))
  Factor{C_vars, C_card, Array{eltype(A), length(C_vars)}}(A_vals_new .* B_vals_new)
end

function product(F::AbstractArray{<:Factor, 1})
  reduce(product, F; init = Factor{(), (), Array{Float64,0}}(Array{Float64,0}(undef)))
end

product(F::Factor...) = product(Factor[F...])

"""
    marg(A::Factor, V::Vector{Int64})

Sum out the variables inside `V` from factor A.
Based on an assignment of the coursera course Probabilistic Graphical Models.
https://www.coursera.org/learn/probabilistic-graphical-models/home/week/1

# Examples
```julia
A = Factor{(1,2,3),(2,2,3), Array{Float64,3}}(cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                                  [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
V = [1]
B = marg(A, V)
```
"""
function marg(A::Factor, vars::Tuple)
  drop_dims = indexin(vars, collect(getvars(A))) |> Tuple # map V to dims
  r_card = ntuple(d -> d in drop_dims ? 1 : getcard(A)[d], ndims(A)) # assign 1 to summed out dims
  B_card = filter(s -> s != 1, r_card)
  B_vars = filter(v -> v ∉ vars, getvars(A))
  r_vals = similar(A.vals, r_card)
  B_vals = sum!(r_vals, A.vals) |> x -> dropdims(x, dims=drop_dims)
  return Factor{B_vars, B_card, Array{eltype(A),length(B_vars)}}(B_vals)
end
marg(A::Factor, V::Int) = marg(A, (V,))

"""
    redu(A::Factor, var::Int64, val::Int64)

Reduce/invalidate all entries that are not consitent with the evidence.

# Examples
```julia
A = Factor{(1, 2, 3),(3,2,2),Array{Float64,3}}(, cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
var = 3
val = 1
B = redu(A, var, val)
```
"""
function redu(A::Factor, var, val)
  A_vars = getvars(A)
  B_vars = setdiff(A_vars, var)
  mapB = indexin(B_vars, collect(A_vars))
  B_card = size(A.vals)[mapB]
  R_vars = intersect(A_vars, var)
  mapR = indexin(R_vars, collect(A_vars))
  indxA = ntuple(i -> (i == mapR[1]) ? val : :, length(A_vars))
  indxA = CartesianIndices(A.vals)[indxA...] # here occurs the actual reduction
  B_vals = A.vals[indxA]
  return Factor{Tuple(B_vars), Tuple(B_card), Array{eltype(A), length(B_vars)}}(A.vals[indxA])
end

"""
    norm(A::Factor)

Normalize the values in Factor A such that all probabilities lie between 0
and 1.

# Examples
```julia
A = Factor{(1,2),(2,3), Array{Float64,2}}([1.0 2.0 3.0; 4.0 5.0 6.0])
B = norm(A)
```
"""
function norm(A::Factor)
  return Factor{getvars(A), getcard(A), Array{eltype(A),length(getvars(A))}}(A.vals ./ sum(A.vals))
end

