using OMEinsum, OMEinsumContractionOrders
using JunctionTrees
nvars, cards, nclique, factors = JunctionTrees.read_uai_file("examples/problems/paskin/paskin.uai")

struct TensorNetworksSolver{ET,MT<:AbstractArray}
    code::ET
    tensors::Vector{MT}
    fixedvertices::Dict{Int,Int}
end

function TensorNetworksSolver(factors::Vector{<:Factor}; openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    tensors = getfield.(factors, :vals)
    rawcode = EinCode([[factor.vars...] for factor in factors], collect(Int, openvertices))  # labels for edge tensors
    code = optimize_code(rawcode, OMEinsum.get_size_dict(getixsv(rawcode), tensors), optimizer, simplifier)
    TensorNetworksSolver(code, tensors, fixedvertices)
end

([30, 17, 174], [2, 2, 2])
tn = TensorNetworksSolver(factors)
tn.code(tn.tensors...)

problem_number = "34"
problem_filename = joinpath("Promedus_" * problem_number)
problem_dir = joinpath(dirname(@__DIR__), "examples/problems/Promedus/", problem_number)
uai_filepath = joinpath(problem_dir, problem_filename * ".uai")
uai_evid_filepath = joinpath(problem_dir, problem_filename * ".uai.evid")
uai_mar_filepath = joinpath(problem_dir, problem_filename * ".uai.MAR")
td_filepath = joinpath(problem_dir, problem_filename * ".td")

reference_marginals = JunctionTrees.read_uai_mar_file(uai_mar_filepath)
obsvars, obsvals = JunctionTrees.read_uai_evid_file(uai_evid_filepath)
nvars, cards, nclique, factors = JunctionTrees.read_uai_file(uai_filepath)

function generate_tensors(gp::TensorNetworksSolver)
    fixedvertices = gp.fixedvertices
    isempty(fixedvertices) && return tensors
    ixs = getixsv(gp.code)
    map(gp.tensors, ixs) do t, ix
        dims = map(ixi->ixi ∉ keys(fixedvertices) ? Colon() : (fixedvertices[ixi]+1:fixedvertices[ixi]+1), ix)
        t[dims...]
    end
end

using LinearAlgebra
for openvertex in 1:10
    tn = TensorNetworksSolver(factors; fixedvertices=Dict(zip(obsvars, obsvals .- 1)), openvertices=(openvertex,))
    @info tn.code(generate_tensors(tn)...) |> normalize!
end

function slice_contract(tn.code, openvertex)(generate_tensors(tn)...)
end

for openvertex in 1:10
    tn = TensorNetworksSolver(factors; fixedvertices=Dict(zip(obsvars, obsvals .- 1)), openvertices=())
    @info slice_contract(tn.code, openvertex)(generate_tensors(tn)...) |> normalize!
end

algo = compile_algo(
            uai_filepath,
            uai_evid_filepath = uai_evid_filepath,
        )
eval(algo)
 