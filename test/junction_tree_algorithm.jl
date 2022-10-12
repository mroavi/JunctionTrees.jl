module TestJunctionTreeAlgorithm

using Test, Artifacts
using JunctionTrees

  @testset "compile algorithm" begin

    problem = "Promedus_12"
    uai_filepath = joinpath(artifact"uai2014", problem * ".uai")
    uai_evid_filepath = joinpath(artifact"uai2014", problem * ".uai.evid")
    uai_mar_filepath = joinpath(artifact"uai2014", problem * ".uai.MAR")
    td_filepath = joinpath(artifact"uai2014", problem * ".tamaki.td")

    reference_marginals = JunctionTrees.read_uai_mar_file(uai_mar_filepath)
    obsvars, obsvals = JunctionTrees.read_uai_evid_file(uai_evid_filepath)

    println("  Problem: $(problem)")

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence
    # ------------------------------------------------------------------------------

    println("    Test: Default")
    println("      Compiling algo...")
    algo = compile_algo(
             uai_filepath;
             uai_evid_filepath = uai_evid_filepath,
           )
    eval(algo)
    println("      Running algo...")
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
    @test isapprox(marginals, reference_marginals, atol=0.03)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence with Float32 factor values
    # ------------------------------------------------------------------------------

    println("    Test: Float32 factor values")
    println("      Compiling algo...")
    algo = compile_algo(
             uai_filepath;
             uai_evid_filepath = uai_evid_filepath,
             factor_eltype = Float32,
           )
    eval(algo)
    println("      Running algo...")
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
    @test isapprox(marginals, reference_marginals)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence and existing junction tree
    # ------------------------------------------------------------------------------

    println("    Test: use an existing junction tree")
    println("      Compiling algo...")
    algo = compile_algo(
             uai_filepath;
             uai_evid_filepath = uai_evid_filepath,
             td_filepath = td_filepath,
           )
    eval(algo)
    println("      Running algo...")
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
    @test isapprox(marginals, reference_marginals, atol=0.03)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence and with partial evaluation
    # ------------------------------------------------------------------------------

    println("    Test: partial evaluation")
    println("      Compiling algo...")
    algo = compile_algo(
             uai_filepath;
             uai_evid_filepath = uai_evid_filepath,
             apply_partial_evaluation = true,
           )
    eval(algo)
    println("      Running algo...")
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
    # Filter the observed variables from the reference solution
    reference_marginals_filtered = reference_marginals[setdiff(begin:end, obsvars)]
    @test isapprox(marginals, reference_marginals_filtered, atol=0.03)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence using OMEinsum as backend
    # ------------------------------------------------------------------------------

    println("    Test: partial evaluation")
    println("      Compiling algo...")
    algo = compile_algo(
             uai_filepath;
             uai_evid_filepath = uai_evid_filepath,
             use_omeinsum = true,
           )
    eval(algo)
    println("      Running algo...")
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
    @test isapprox(marginals, reference_marginals, atol=0.03)

  end

end # module
