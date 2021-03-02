using DiscreteBayes
using Test

@testset "factors.jl" begin

  # Factor product

  A = Factor{Float64,1}((1,), [0.11; 0.89])
  B = Factor{Float64,2}((2, 1), [0.59 0.22; 0.41 0.78])
  C = product(A, B)
  @test C.vars == (1, 2)
  @test C.vals ≈ [0.0649 0.0451; 0.1958 0.6942]

  A = Factor{Float64,2}((3, 2), [0.5 0.1; 0.7 0.2])
  B = Factor{Float64,2}((2, 1), [0.5 0.1 0.3; 0.8 0.0 0.9])
  C = product(A, B)
  @test C.vars == (3, 2, 1)
  @test vec(C.vals) ≈ [0.2500, 0.3500, 0.0800, 0.1600, 0.0500, 0.0700,
                       0.0000, 0.0000, 0.1500, 0.2100, 0.0900, 0.1800]

  A = Factor{Float64,2}((2, 1), [0.5 0.1 0.3; 0.8 0.0 0.9])
  B = Factor{Float64,0}((), Array{Float64,0}(undef))
  C = product(A, B)
  @test C.vars == (2, 1)
  @test C.vals ≈ [0.5 0.1 0.3; 0.8 0.0 0.9]

  A = Factor{Float64,0}((), Array{Float64,0}(undef))
  B = Factor{Float64,2}((2, 1), [0.5 0.1 0.3; 0.8 0.0 0.9])
  C = product(A, B)
  @test C.vars == (2, 1)
  @test C.vals ≈ [0.5 0.1 0.3; 0.8 0.0 0.9]

  A = Factor{Float64,0}((), Array{Float64,0}(undef))
  B = Factor{Float64,0}((), Array{Float64,0}(undef))
  C = product(A, B)
  @test C.vars == ()
  @test ndims(C.vals) == 0

  A = Factor{Float64,1}((1,), [0.11; 0.89])
  B = Factor{Float64,2}((2, 1), [0.59 0.22; 0.41 0.78])
  C = product([A, B])
  @test C.vars == (1, 2)
  @test C.vals ≈ [0.0649 0.0451; 0.1958 0.6942]

  # Factor marginalization

  A = Factor{Float64,2}((2, 1), [0.59 0.41; 0.22 0.78])
  V = [1]
  B = marg(A, V)
  @test B.vars == (2,)
  @test B.vals ≈ [1.0, 1.0]

  A = Factor{Float64,3}((3, 2, 1), cat([0.25 0.08; 0.35 0.16],
                            [0.05 0.00; 0.07 0.00], 
                            [0.15 0.09; 0.21 0.18], dims=3))
  V = [2]
  B = marg(A, V)
  @test B.vars == (3, 1)
  @test vec(B.vals) ≈ [0.33, 0.51, 0.05, 0.07, 0.24, 0.39]

  # Factor reduction

  A = Factor{Float64,3}((3, 2, 1), cat([0.25 0.08; 0.35 0.16],
                            [0.05 0.00; 0.07 0.00], 
                            [0.15 0.09; 0.21 0.18], dims=3))
  var = 3
  val = 1
  B = redu(A, var, val)
  @test B.vars == (2, 1)
  @test vec(B.vals) ≈ [0.25, 0.08, 0.05, 0.00, 0.15, 0.09]

  # Factor normalization

  A = Factor{Float64,1}((1,), [0.22; 0.89])
  C = norm(A)
  @test sum(C.vals) ≈ 1.0

  A = Factor{Float64,2}((2, 1), [0.59 0.22; 0.41 0.78])
  C = norm(A)
  @test sum(C.vals) ≈ 1.0

  A = Factor{Float64,3}((3, 2, 1), cat([0.25 0.08; 0.35 0.16],
                            [0.05 0.00; 0.07 0.00], 
                            [0.15 0.09; 0.21 0.18], dims=3))
  C = norm(A)
  @test sum(C.vals) ≈ 1.0

end
