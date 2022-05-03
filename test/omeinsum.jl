using OMEinsum, OMEinsumContractionOrders
using JunctionTrees
using Graphs, Test
import LinearAlgebra
#nvars, cards, nclique, factors = JunctionTrees.read_uai_file("examples/problems/paskin/paskin.uai")

@testset "popindices!" begin
    code1 = EinCode([[e.src, e.dst] for e in edges(smallgraph(:petersen))], Int[2,3])
    opt1 = optimize_code(code1, uniformsize(code1, 2), GreedyMethod())

    # pop
    code2 = EinCode([[e.src, e.dst] for e in edges(smallgraph(:petersen))], Int[])
    opt2 = optimize_code(code2, uniformsize(code2, 2), TreeSA(; ntrials=1, nslices=2, fixed_slices=[8]))
    opt2 = popindices(opt2, (2,3))
    ts = [randn(2,2) for i=1:15]
    @test opt1(ts...) ≈ opt2(ts...)

    # slice over NE
    opt3 = optimize_code(code2, uniformsize(code2, 2), GreedyMethod())
    @test opt1(ts...) ≈ slice_contract(opt3, [2,3], ts...)

    # slice over SE
    opt4 = optimize_code(code2, uniformsize(code2, 2), TreeSA(; ntrials=1, nslices=2, fixed_slices=[8]))
    @test opt1(ts...) ≈ slice_contract(opt4, [2,3], ts...)
end

for i in 1:50
    @show i
    problem_number = "$i"
    problem_filename = joinpath("Promedus_" * problem_number)
    problem_dir = joinpath(dirname(@__DIR__), "examples/problems/Promedus/", problem_number)
    uai_filepath = joinpath(problem_dir, problem_filename * ".uai")
    uai_evid_filepath = joinpath(problem_dir, problem_filename * ".uai.evid")
    uai_mar_filepath = joinpath(problem_dir, problem_filename * ".uai.MAR")
    td_filepath = joinpath(problem_dir, problem_filename * ".td")

    try
        reference_marginals = JunctionTrees.read_uai_mar_file(uai_mar_filepath)
    catch
        continue
    end

    obsvars, obsvals = JunctionTrees.read_uai_evid_file(uai_evid_filepath)
    nvars, cards, nclique, factors = JunctionTrees.read_uai_file(uai_filepath)

    # does not optimize over open vertices
    tn = TensorNetworksSolver(factors; fixedvertices=Dict(zip(obsvars, obsvals .- 1)), openvertices=Int[], optimizer=TreeSA(ntrials=1))
    @info timespace_complexity(tn.code, OMEinsum.get_size_dict(getixsv(tn.code), tn.tensors))
end

@testset "tensor network solvers" begin
    ################# Load problem ####################
    problem_number = "14"
    problem_filename = joinpath("Promedus_" * problem_number)
    problem_dir = joinpath(dirname(@__DIR__), "examples/problems/Promedus/", problem_number)
    uai_filepath = joinpath(problem_dir, problem_filename * ".uai")
    uai_evid_filepath = joinpath(problem_dir, problem_filename * ".uai.evid")
    uai_mar_filepath = joinpath(problem_dir, problem_filename * ".uai.MAR")
    td_filepath = joinpath(problem_dir, problem_filename * ".td")

    reference_marginals = JunctionTrees.read_uai_mar_file(uai_mar_filepath)
    obsvars, obsvals = JunctionTrees.read_uai_evid_file(uai_evid_filepath)
    nvars, cards, nclique, factors = JunctionTrees.read_uai_file(uai_filepath)

    # does not optimize over open vertices
    tn = TensorNetworksSolver(factors; fixedvertices=Dict(zip(obsvars, obsvals .- 1)), openvertices=Int[], optimizer=TreeSA(ntrials=1))
    @info timespace_complexity(tn.code, OMEinsum.get_size_dict(getixsv(tn.code), tn.tensors))
    tensors = generate_tensors(tn)
    @time marginals = map(1:nvars) do openvertex
        code = popindices(tn.code, [openvertex])
        code(tensors...) |> LinearAlgebra.normalize!
    end;
    # for dangling vertices, the output size is 1.
    npass = 0
    for i=1:nvars
        npass += (length(marginals[i]) == 1 && reference_marginals[i] == [0.0, 1]) || isapprox(marginals[i], reference_marginals[i]; rtol=1e-1, atol=1e-4)
    end
    @test npass >= nvars / 2

    @time marginals = map(1:nvars) do openvertex
        slice_contract(tn.code, Int[openvertex], tensors...) |> LinearAlgebra.normalize!
    end;
    npass = 0
    for i=1:nvars
        npass += (length(marginals[i]) == 1 && reference_marginals[i] == [0.0, 1]) || isapprox(marginals[i], reference_marginals[i]; rtol=1e-1, atol=1e-4)
    end
    @test npass >= nvars / 2
    algo = compile_algo(
                uai_filepath,
                uai_evid_filepath = uai_evid_filepath,
            )
    eval(algo)
    @time marginals = run_algo(obsvars, obsvals) |> x -> map(y -> y.vals, x)
end 