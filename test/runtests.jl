using Test

@testset "factors" begin
  include("factors.jl")
end

@testset "tree" begin
  include("tree.jl")
end

@testset "utils" begin
  include("utils.jl")
end

@testset "junction_tree_algorithm" begin
  include("junction_tree_algorithm.jl")
end

@testset "uai2014" begin
  include("uai2014.jl")
end

@testset "doctests" begin
  include("doctests.jl")
end
