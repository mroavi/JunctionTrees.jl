module TestJunctionTreeAlgorithm

using Test
using JunctionTrees

  @testset "compile algorithm" begin

    problem_number = "34"
    problem_filename = joinpath("Promedus_" * problem_number)
    problem_dir = joinpath(@__DIR__, "../examples/problems/Promedus/", problem_number)
    uai_filepath = joinpath(problem_dir, problem_filename * ".uai")
    uai_evid_filepath = joinpath(problem_dir, problem_filename * ".uai.evid")
    uai_mar_filepath = joinpath(problem_dir, problem_filename * ".uai.MAR")
    td_filepath = joinpath(problem_dir, problem_filename * ".tamaki.td")

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
