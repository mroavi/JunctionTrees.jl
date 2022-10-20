module TestUai2014Omeinsum

using Test, Artifacts
using JunctionTrees
using JunctionTrees: read_uai_file, read_uai_evid_file, read_uai_mar_file

@debug "OMEinsum backend tests starting..."

benchmarks = [
              # "Alchemy", overflows
              "CSP",
              "DBN",
              # "Grids", # overflows
              # "linkage", # fails: OutOfMemoryError (problems 15,20,23)
              # "ObjectDetection", # fails from 36 onwards
              "Pedigree",
              "Promedus",
              # "relational", # fails: OutOfMemoryError (problem 1), Kill signal (problem 2)
              "Segmentation",
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

        @debug "     Compiling algo..."
        algo = compile_algo(
                            uai_filepath;
                            uai_evid_filepath = uai_evid_filepath, 
                            td_filepath = td_filepath,
                            factor_eltype = Float64,
                            use_omeinsum = true,
                          )
        eval(algo)
        @debug "     Running algo..."
        marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
        @test isapprox(marginals, reference_marginals, atol = 0.01)

      end
    end
  end
end

end # module
