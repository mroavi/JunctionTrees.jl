module TestUai2014PartialEvaluation

using Test, Artifacts
using JunctionTrees
using JunctionTrees: read_uai_file, read_uai_evid_file, read_uai_mar_file

@debug "Partial evaluation tests starting..."

benchmarks = [
              "Pedigree",
              "Promedus",
             ]

for benchmark in benchmarks

  @testset "$(benchmark) benchmark" begin

    @debug " Benchmark: $(benchmark)"

    rexp = Regex("($(benchmark)_\\d*)(\\.uai)\$") 
    problems = readdir(artifact"uai2014"; sort=false) |> 
      x -> map(y -> match(rexp, y), x) |>
      x -> filter(!isnothing, x) |>
      x -> map(first, x)

    for problem in problems

      @testset "$(problem)" begin

        @debug "  Problem: $(problem)"

        uai_filepath = joinpath(artifact"uai2014", problem * ".uai")
        uai_evid_filepath = joinpath(artifact"uai2014", problem * ".uai.evid")
        uai_mar_filepath = joinpath(artifact"uai2014", problem * ".uai.MAR")
        td_filepath = joinpath(artifact"uai2014", problem * ".tamaki.td")

        nvars, cards, nclique, factors = read_uai_file(uai_filepath; factor_eltype = Float64)
        obsvars, obsvals = read_uai_evid_file(uai_evid_filepath)
        reference_marginals = read_uai_mar_file(uai_mar_filepath)

        @debug "      Compiling algo..."
        algo = compile_algo(
                 uai_filepath;
                 uai_evid_filepath = uai_evid_filepath,
                 td_filepath = td_filepath,
                 apply_partial_evaluation = true,
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

      end
    end
  end
end

end # module
