using Test
using Pkg

Pkg.ensure_artifact_installed("uai2014", "Artifacts.toml")

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

@testset "doctests" begin
  include("doctests.jl")
end
