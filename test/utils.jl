module TestUtils

using Test
using JunctionTrees: get_td_soln
using Artifacts

  @testset "Tree decompositions" begin

    td_filepath = joinpath(artifact"uai2014", "Promedus_34.tamaki.td")
    @test get_td_soln(td_filepath) == [353, 15, 415]

  end

end # module
