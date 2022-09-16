import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.status()

using BenchmarkTools, Artifacts
using JunctionTrees
using JunctionTrees: read_uai_file, read_uai_evid_file, read_uai_mar_file

const SUITE = BenchmarkGroup()

SUITE["uai2014"] = BenchmarkGroup()


benchmarks = [
              # "Alchemy",
              "CSP",
              "DBN",
              # "Grids",
              # "linkage",
              "ObjectDetection",
              "Pedigree",
              "Promedus",
              # "relational",
              "Segmentation",
             ]

for benchmark in benchmarks

  SUITE["uai2014"][benchmark] = BenchmarkGroup()

  # Capture the problem names that belongs to the current benchmark
  rexp = Regex("($(benchmark)_\\d*)(\\.uai)\$") 
  problems = readdir(artifact"uai2014"; sort=false) |> 
    x -> map(y -> match(rexp, y), x) |> # apply regex
    x -> filter(!isnothing, x) |> # filter out `nothing` values
    x -> map(first, x) # get the first capture of each element

  for problem in problems

    SUITE["uai2014"][benchmark][problem] = BenchmarkGroup()

    uai_filepath = joinpath(artifact"uai2014", problem * ".uai")
    uai_evid_filepath = joinpath(artifact"uai2014", problem * ".uai.evid")
    uai_mar_filepath = joinpath(artifact"uai2014", problem * ".uai.MAR")
    td_filepath = joinpath(artifact"uai2014", problem * ".tamaki.td")

    nvars, cards, nclique, factors = read_uai_file(uai_filepath; factor_eltype=Float64)
    obsvars, obsvals = read_uai_evid_file(uai_evid_filepath)
    reference_marginals = read_uai_mar_file(uai_mar_filepath)

    # ------------------------------------------------------------------------------
    # OMEinsum
    # ------------------------------------------------------------------------------

    SUITE["uai2014"][benchmark][problem]["omeinsum"] = BenchmarkGroup()

    algo = compile_algo(
                        uai_filepath;
                        uai_evid_filepath = uai_evid_filepath, 
                        td_filepath = td_filepath,
                       )

    for use_omeinsum in [false, true]

      if use_omeinsum
        algo = boost_algo(algo)
      end

      eval(algo)

      obsvars, obsvals = JunctionTrees.read_uai_evid_file(uai_evid_filepath)

      SUITE["uai2014"][benchmark][problem]["omeinsum"][string(use_omeinsum)] =
      @benchmarkable run_algo($obsvars, $obsvals)

    end

  end
end
