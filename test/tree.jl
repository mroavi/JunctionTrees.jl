module TestTree

using Test
using JunctionTrees: Node, addchildren!, addchild!, Leaves, convertToAbstractTree!
using AbstractTrees: repr_tree
using Graphs: binary_tree

  @testset "Custom tree type" begin

    root = Node(1)
    @test repr_tree(root) == """
    1
    """

    children = addchildren!([2,3,4], root)
    @test repr_tree(root) == """
    1
    ├─ 2
    ├─ 3
    └─ 4
    """

    addchild!(5, children[1])
    @test repr_tree(root) == """
    1
    ├─ 2
    │  └─ 5
    ├─ 3
    └─ 4
    """

    @test [n.id for n in Leaves(root)] == [5, 3, 4]
    @test parent(root, root) == nothing
    @test parent(root, children[1]) === root

    g = binary_tree(3)
    root = Node(1)
    convertToAbstractTree!(g, root)
    @test repr_tree(root) == """
    1
    ├─ 2
    │  ├─ 4
    │  └─ 5
    └─ 3
       ├─ 6
       └─ 7
    """

  end

end # module
