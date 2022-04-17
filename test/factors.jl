module TestFactors

using Test
using JunctionTrees

@testset "factors.jl" begin

  ## Factor product

  @testset "factor product" begin

    A = Factor{Float64,1}((1,), [0.11; 0.89])
    B = Factor{Float64,2}((1, 2), [0.59 0.41; 0.22 0.78])
    C = prod(A, B)
    @test C.vars == (1, 2)
    @test C.vals ≈ [0.0649 0.0451; 0.1958 0.6942]

    A = Factor{Float64,2}((2, 3), [0.5 0.7; 0.1 0.2])
    B = Factor{Float64,2}((1, 2), [0.5 0.8; 0.1 0.0; 0.3 0.9])
    C = prod(A, B)
    @test C.vars == (1, 2, 3)
    @test C.vals ≈ cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                       [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3)

    A = Factor{Float64,2}((1, 2), [0.5 0.8; 0.1 0.0; 0.3 0.9])
    B = Factor{Float64,0}((), Array{Float64,0}(undef))
    C = prod(A, B)
    @test C.vars == (1, 2)
    @test C.vals ≈ [0.5 0.8; 0.1 0.0; 0.3 0.9]

    A = Factor{Float64,0}((), Array{Float64,0}(undef))
    B = Factor{Float64,2}((1, 2), [0.5 0.8; 0.1 0.0; 0.3 0.9])
    C = prod(A, B)
    @test C.vars == (1, 2)
    @test C.vals ≈ [0.5 0.8; 0.1 0.0; 0.3 0.9]

    A = Factor{Float64,0}((), Array{Float64,0}(undef))
    B = Factor{Float64,0}((), Array{Float64,0}(undef))
    C = prod(A, B)
    @test C.vars == ()
    @test ndims(C.vals) == 0

  end

  ## Factor marginalization

  @testset "factor marginalization" begin

    A = Factor{Float64,2}((1, 2), [0.59 0.41; 0.22 0.78])
    marg_vars = (2,)
    B = sum(A, marg_vars)
    @test B.vars == (1,)
    @test B.vals ≈ [1.0, 1.0]

    A = Factor{Float64,3}((1, 2, 3), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                         [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))

    marg_vars = (2,)
    B = sum(A, marg_vars)
    @test B.vars == (1, 3)
    @test vec(B.vals) ≈ [0.33, 0.05, 0.24, 0.51, 0.07, 0.39]

    marg_var = 2
    B = sum(A, marg_var)
    @test B.vars == (1, 3)
    @test vec(B.vals) ≈ [0.33, 0.05, 0.24, 0.51, 0.07, 0.39]

    A = Factor{Float64,0}((), Array{Float64,0}(undef))
    marg_vars = ()
    B = sum(A, marg_vars)
    @test B.vars == A.vars

  end

  ## Factor reduction

  @testset "factor reduction" begin

    A = Factor{Float64,3}((1, 2, 3), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                         [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))

    obs_vars = (3,)
    obs_vals = (1,)
    B = redu(A, obs_vars, obs_vals)
    @test B.vars == A.vars
    @test B.vals ≈ cat([0.25 0.08; 0.05 0.00; 0.15 0.09],
                       [0.00 0.00; 0.00 0.00; 0.00 0.00], dims=3)

    obs_vars = (2, 3)
    obs_val = (1, 1)
    B = redu(A, obs_vars, obs_val)
    @test B.vars == A.vars
    @test B.vals ≈ cat([0.25 0.00; 0.05 0.00; 0.15 0.00],
                       [0.00 0.00; 0.00 0.00; 0.00 0.00], dims=3)

    A = Factor{Float64,1}((1,), [0.11; 0.89])
    obs_vars = (1,)
    obs_val = (1,)
    B = redu(A, obs_vars, obs_val)
    @test B.vars == A.vars
    @test B.vals ≈ [0.11, 0.00]

  end

  ## Factor normalization

  @testset "factor normalization" begin

    A = Factor{Float64,1}((1,), [0.22; 0.89])
    B = norm(A)
    @test sum(B.vals) ≈ 1.0

    A = Factor{Float64,2}((1, 2), [0.59 0.41; 0.22 0.78])
    B = norm(A)
    @test sum(B.vals) ≈ 1.0

    A = Factor{Float64,3}((1, 2, 3), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                         [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))
    B = norm(A)
    @test sum(B.vals) ≈ 1.0

  end

  ## Factor identities

  @testset "factor identities" begin

    A = Factor{Float64,3}((1, 2, 3), cat([0.25 0.08; 0.05 0.0; 0.15 0.09],
                                         [0.35 0.16; 0.07 0.0; 0.21 0.18], dims=3))

    B = Factor{Float64,2}((1, 2), [0.5 0.8; 0.1 0.0; 0.3 0.9])

    obs_vars = (3,)
    obs_vals = (1,)

    # Reduction followed by product is equivalent to product followed by reduction

    C1 = redu(A, obs_vars, obs_vals) |> x -> prod(x, B)
    C2 = prod(A, B) |> x -> redu(x, obs_vars, obs_vals)

    @test  C1.vals ≈ C2.vals

    # Reduction followed by sum of unobserved var is equivalent to sum followed by reduction

    marg_var = 1

    C1 = redu(A, obs_vars, obs_vals) |> x -> sum(x, marg_var)
    C2 = sum(A, marg_var) |> x -> redu(x, obs_vars, obs_vals)

    @test C1.vals ≈ C2.vals

    # Reduction followed by sum of observed var is not equivalent to sum followed by reduction

    marg_var = 3

    C1 = redu(A, obs_vars, obs_vals) |> x -> sum(x, marg_var)
    C2 = sum(A, marg_var) |> x -> redu(x, obs_vars, obs_vals)

    @test C1.vals ≉ C2.vals

    # 

    m1 = prod(A, B) |> x -> sum(x, (1,3))
    m2 = redu(A, obs_vars, obs_vals) |> x -> prod(x, B) |> x -> sum(x, (1,3))

    @test m1.vals ≉ m2.vals

  end

  @testset "Partial evaluation" begin

    # No partial evaluation pass

    obsvars, obsvals = ((8,), (2,))

    pot_4 = Factor{Float64, 4}((4, 5, 6, 8), [0.0009279862866476761 0.00584815014154717; 0.00037925022896286893 0.014309821640592284;;; 0.0048944901009020545 0.0011087984696530988; 0.0020002843987374005 0.0027131157634574456;;;; 0.010256074824636292 0.0005291501111688599; 0.004191461427253161 0.0012947758741916851;;; 0.054093748394313974 0.00010032588413087112; 0.022107079338546574 0.00024548711525858384]) |> (x->begin redu(x, (8,), (obsvals[1],)) end)
    pot_5 = Factor{Float64, 3}((6, 8, 9), [0.006166980122542561 0.01952542918228385; 0.002944368873451448 0.009322239537460154;;; 0.005749141918775925 0.022858174721902973; 0.012041576805968082 0.047876443206091024]) |> (x->begin redu(x, (8,), (obsvals[1],)) end)
    pot_6 = Factor{Float64, 3}((4, 7, 8), [0.020498711397885946 0.010623701923114055; 0.029130886644614055 0.007475646119385946;;; 0.0020937321096140553 0.10401149158438594; 0.002975419847885945 0.07319041037311406]) |> (x->begin redu(x, (8,), (obsvals[1],)) end)

    msg_5_4 = sum(pot_5, 9)
    msg_6_4 = sum(pot_6, 7)
    msg_4_3 = sum(prod(msg_5_4, msg_6_4, pot_4), 8)

    # After partial evaluation pass

    msg_5_4 = Factor{Float64, 2}((6, 8), [0.011916122041318486 0.04238360390418683; 0.01498594567941953 0.05719868274355118])
    msg_6_4 = Factor{Float64, 2}((4, 8), [0.031122413321 0.106105223694; 0.036606532764 0.07616583022100001])
    msg_4_3_pe = sum(redu(Factor{Float64, 4}((4, 5, 6, 8), [3.441515794150131e-7 2.168836045131908e-6; 1.6543195047310308e-7 6.242057417866914e-6;;; 2.282784287358366e-6 5.171422706329727e-7; 1.0973230394844235e-6 1.4883705726592891e-6;;;; 4.612281740155957e-5 2.379652486234837e-6; 1.3530801922221528e-5 4.179772661975819e-6;;; 0.00032829923389915167 6.088857192947307e-7; 9.631137374874008e-5 1.0694855229901174e-6]), (8,), (obsvals[1],)), 8)

    @test msg_4_3.vals ≈ msg_4_3_pe.vals

  end

end

end # module
