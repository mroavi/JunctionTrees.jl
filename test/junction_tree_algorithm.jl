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
    td_filepath = joinpath(problem_dir, problem_filename * ".td")

    reference_marginals = JunctionTrees.read_uai_mar_file(uai_mar_filepath)
    obsvars, obsvals = JunctionTrees.read_uai_evid_file(uai_evid_filepath)

    # Posterior marginals given evidence

    algo = compile_algo(
             uai_filepath,
             uai_evid_filepath = uai_evid_filepath,
           )
    eval(algo)
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)

    @test isapprox(marginals, reference_marginals, atol=0.03)

    # Posterior marginals given evidence and existing junction tree

    algo = compile_algo(
             uai_filepath,
             uai_evid_filepath = uai_evid_filepath,
             td_filepath = td_filepath,
           )
    eval(algo)
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)

    @test isapprox(marginals, reference_marginals, atol=0.03)

  end

end # module
