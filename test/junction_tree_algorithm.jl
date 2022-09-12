module TestJunctionTreeAlgorithm

using Test, Artifacts
using JunctionTrees

  @testset "compile algorithm" begin

    problem = "Promedus_34"
    uai_filepath = joinpath(artifact"uai2014", problem * ".uai")
    uai_evid_filepath = joinpath(artifact"uai2014", problem * ".uai.evid")
    uai_mar_filepath = joinpath(artifact"uai2014", problem * ".uai.MAR")
    td_filepath = joinpath(artifact"uai2014", problem * ".tamaki.td")

    reference_marginals = JunctionTrees.read_uai_mar_file(uai_mar_filepath)
    obsvars, obsvals = JunctionTrees.read_uai_evid_file(uai_evid_filepath)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence
    # ------------------------------------------------------------------------------

    algo = compile_algo(
             uai_filepath,
             uai_evid_filepath = uai_evid_filepath,
           )
    eval(algo)
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)

    @test isapprox(marginals, reference_marginals, atol=0.03)

    # boost the algorithm using OMEinsum
    eval(boost_algo(algo))
    marginals_boosted = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
    @test isapprox(marginals_boosted, reference_marginals, atol=0.03)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence with Float32 factor values
    # ------------------------------------------------------------------------------

    algo = compile_algo(
             uai_filepath,
             uai_evid_filepath = uai_evid_filepath,
             factor_eltype = Float32,
           )
    eval(algo)
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)

    @test isapprox(marginals, reference_marginals)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence and existing junction tree
    # ------------------------------------------------------------------------------

    algo = compile_algo(
             uai_filepath,
             uai_evid_filepath = uai_evid_filepath,
             td_filepath = td_filepath,
           )
    eval(algo)
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)

    @test isapprox(marginals, reference_marginals, atol=0.03)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence and with partial evaluation
    # ------------------------------------------------------------------------------

    algo = compile_algo(
             uai_filepath,
             uai_evid_filepath = uai_evid_filepath,
             apply_partial_evaluation = true,
           )
    eval(algo)
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)

    # Filter the observed variables from the reference solution
    reference_marginals_filtered = reference_marginals[setdiff(begin:end, obsvars)]

    @test isapprox(marginals, reference_marginals_filtered, atol=0.03)

  end

end # module
