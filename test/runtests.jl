using Test

@testset "factors" begin
  include("factors.jl")
end

@testset "junction_tree_algorithm" begin
  include("junction_tree_algorithm.jl")
end

@testset "tree" begin
  include("tree.jl")
end
