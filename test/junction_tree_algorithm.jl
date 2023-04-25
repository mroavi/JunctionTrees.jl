module TestJunctionTreeAlgorithm

using Test, Artifacts
using JunctionTrees

  @testset "compile algorithm" begin

    problem = "Promedus_26"
    uai_filepath = joinpath(artifact"uai2014", problem * ".uai")
    uai_evid_filepath = joinpath(artifact"uai2014", problem * ".uai.evid")
    uai_mar_filepath = joinpath(artifact"uai2014", problem * ".uai.MAR")
    td_filepath = joinpath(artifact"uai2014", problem * ".tamaki.td")

    reference_marginals = JunctionTrees.read_uai_mar_file(uai_mar_filepath)
    obsvars, obsvals = JunctionTrees.read_uai_evid_file(uai_evid_filepath)

    @debug "  Problem: $(problem)"

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence
    # ------------------------------------------------------------------------------

    @debug "    Test: Default (Min-fill heuristic)"
    @debug "      Compiling algo..."
    algo = compile_algo(
             uai_filepath;
             uai_evid_filepath = uai_evid_filepath,
           )
    eval(algo)
    @debug "      Running algo..."
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
    @test isapprox(marginals, reference_marginals, atol=0.03)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence and existing junction tree
    # ------------------------------------------------------------------------------

    @debug "    Test: Using an existing junction tree"
    @debug "      Compiling algo..."
    algo = compile_algo(
             uai_filepath;
             uai_evid_filepath = uai_evid_filepath,
             td_filepath = td_filepath,
             correct_fp_overflows = true,
           )
    eval(algo)
    @debug "      Running algo..."
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
    @test isapprox(marginals, reference_marginals, atol=0.03)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence with Float32 factor values
    # ------------------------------------------------------------------------------

    @debug "    Test: Float32 factor values"
    @debug "      Compiling algo..."
    algo = compile_algo(
             uai_filepath;
             uai_evid_filepath = uai_evid_filepath,
             td_filepath = td_filepath,
             factor_eltype = Float32,
           )
    eval(algo)
    @debug "      Running algo..."
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
    @test isapprox(marginals, reference_marginals)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence and with partial evaluation
    # ------------------------------------------------------------------------------

    @debug "    Test: Partial evaluation"
    @debug "      Compiling algo..."
    algo = compile_algo(
             uai_filepath;
             uai_evid_filepath = uai_evid_filepath,
             td_filepath = td_filepath,
             apply_partial_evaluation = true,
             correct_fp_overflows = true,
           )
    eval(algo)
    @debug "      Running algo..."
    marginal_factors = run_algo(obsvars, obsvals)
    # Filter the observed variables from the obtained solution
    marginal_factors_filtered = filter(x -> !(x.vars[1] in obsvars) , marginal_factors)
    marginals = map(y -> y.vals, marginal_factors_filtered)
    # Filter the observed variables from the reference solution
    reference_marginals_filtered = reference_marginals[setdiff(begin:end, obsvars)]
    @test isapprox(marginals, reference_marginals_filtered, atol=0.03)

    # ------------------------------------------------------------------------------
    # Posterior marginals given evidence using OMEinsum as backend
    # ------------------------------------------------------------------------------

    @debug "    Test: OMEinsum"
    @debug "      Compiling algo..."
    algo = compile_algo(
             uai_filepath;
             uai_evid_filepath = uai_evid_filepath,
             td_filepath = td_filepath,
             use_omeinsum = true,
           )
    eval(algo)
    @debug "      Running algo..."
    marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
    @test isapprox(marginals, reference_marginals, atol=0.03)

  end

end # module
