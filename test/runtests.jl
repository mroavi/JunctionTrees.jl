using DiscreteBayes
using Test

@testset "factors.jl" begin

  # Factor product

  A = Factor{(1,),(2,), Array{Float64,1}}([0.11; 0.89])
  B = Factor{(1,2),(2,2),Array{Float64,2}}([0.59 0.41; 0.22 0.78])
  C = product(A, B)
  @test getvars(C) == (1, 2)
  @test C.vals ≈ [0.0649 0.0451; 0.1958 0.6942]

  A = Factor{(2,3), (2,2), Array{Float64,2}}([0.5 0.7; 0.1 0.2])
  B = Factor{(1,2), (3,2), Array{Float64,2}}([0.5 0.8; 0.1 0.0; 0.3 0.9])
  C = product(A, B)
  @test getvars(C) == (1, 2, 3)
  @test C.vals ≈ cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3)

  A = Factor{(1, 2), (3,2), Array{Float64,2}}( [0.5 0.8; 0.1 0.0; 0.3 0.9])
  B = Factor{(), (), Array{Float64,0}}(Array{Float64,0}(undef))
  C = product(A, B)
  @test getvars(C) == (1, 2)
  @test C.vals ≈ [0.5 0.8; 0.1 0.0; 0.3 0.9]

  A = Factor{(), (), Array{Float64,0}}(Array{Float64,0}(undef))
  B = Factor{(1,2), (3,2), Array{Float64,2}}([0.5 0.8; 0.1 0.0; 0.3 0.9])
  C = product(A, B)
  @test getvars(C) == (1, 2)
  @test C.vals ≈ [0.5 0.8; 0.1 0.0; 0.3 0.9]

  A = Factor{(), (), Array{Float64,0}}(Array{Float64,0}(undef))
  B = Factor{(), (), Array{Float64,0}}(Array{Float64,0}(undef))
  C = product(A, B)
  @test getvars(C) == ()
  @test ndims(C.vals) == 0

  A = Factor{(1,), (2,), Array{Float64,1}}([0.11; 0.89])
  B = Factor{(1, 2), (2,2), Array{Float64,2}}([0.59 0.41; 0.22 0.78])
  C = product([A, B])
  @test getvars(C) == (1, 2)
  @test C.vals ≈ [0.0649 0.0451; 0.1958 0.6942]

  A = Factor{(1,), (2,), Array{Float64,1}}([0.11; 0.89])
  B = Factor{(1, 2), (2,2), Array{Float64,2}}([0.59 0.41; 0.22 0.78])
  C = Factor{(1, 3), (2,2), Array{Float64,2}}([0.0649 0.0451; 0.1958 0.6942])
  D = product(A, B, C)
  @test getvars(D) == (1, 2, 3)
  @test D.vals ≈ cat([0.00421201 0.00292699; 0.03833764 0.13592436],
                     [0.00292699 0.00203401; 0.13592436 0.48191364], dims=3)

  # Factor marginalization

  A = Factor{(1,2), (2,2), Array{Float64,2}}([0.59 0.41; 0.22 0.78])
  V = (2,)
  B = marg(A, V)
  @test getvars(B) == (1,)
  @test B.vals ≈ [1.0, 1.0]

  A = Factor{(1,2,3), (3,2,2), Array{Float64,3}}(cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))

  V = (2,)
  B = marg(A, V)
  @test getvars(B) == (1, 3)
  @test vec(B.vals) ≈ [0.33, 0.05, 0.24, 0.51, 0.07, 0.39]

  # Factor reduction

  A = Factor{(1,2,3), (3,2,2), Array{Float64,3}}(cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
  var = 3
  val = 1
  B = redu(A, var, val)
  @test getvars(B) == (1, 2)
  @test B.vals ≈ [0.25 0.08; 0.05 0.0; 0.15 0.09]

  # Factor normalization

  A = Factor{(1,), (2,), Array{Float64,1}}([0.22; 0.89])
  C = norm(A)
  @test sum(C.vals) ≈ 1.0

  B = Factor{(1,2), (2,2), Array{Float64,2}}([0.59 0.41; 0.22 0.78])
  C = norm(A)
  @test sum(C.vals) ≈ 1.0

  A = Factor{(1,2,3), (3,2,2), Array{Float64,3}}(cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                                     [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
  C = norm(A)
  @test sum(C.vals) ≈ 1.0

end
