module TestUtils

using Test
using JunctionTrees: get_tree_width
using Artifacts

  @testset "Tree decompositions" begin

    td_filepath = joinpath(artifact"uai2014", "Promedus_34.tamaki.td")
    @test get_tree_width(td_filepath; td="merlin") == 14

  end

end # module
