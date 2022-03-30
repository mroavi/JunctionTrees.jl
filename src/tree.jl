# Based on: https://github.com/JuliaCollections/AbstractTrees.jl/tree/master/examples

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Tree node definition intended to be used with AbstractTrees.jl.
"""
mutable struct Node{T}
  id::T
  children::Vector{Node{T}}
  parent::Node{T}

  # Root constructor
  Node{T}(id) where T = new{T}(id, Node{T}[])

  # Child node constructor
  Node{T}(id, parent::Node{T}) where T = new{T}(id, Node{T}[], parent)
end

# Outer constructor that extracts the type from the argument
Node(id) = Node{typeof(id)}(id)

"""
$(TYPEDSIGNATURES)

Add a child node with id `id` to `parent`.
"""
function addchild!(id, parent::Node)
  node = typeof(parent)(id, parent)
  push!(parent.children, node)
  return node
end

"""
$(TYPEDSIGNATURES)

Add several children with with ids `ids` to `parent`.
"""
function addchildren!(ids, parent::Node)
  nodes = map(id -> typeof(parent)(id, parent), ids)
  append!(parent.children, nodes)
  return nodes
end

# ------------------------------------------------------------------------------
# AbstractTrees interface implementation
# ------------------------------------------------------------------------------

# Things we need to define
AbstractTrees.children(node::Node) = node.children

# Things that make printing prettier
AbstractTrees.printnode(io::IO, node::Node) = print(io, node.id)

# Optional enhancements
# These next two definitions allow inference of the item type in iteration.
# (They are not sufficient to solve all internal inference issues, however.)
Base.eltype(::Type{<:TreeIterator{Node{T}}}) where T = Node{T}
Base.IteratorEltype(::Type{<:TreeIterator{Node{T}}}) where T = Base.HasEltype()

Base.parent(root::Node, node::Node) = isdefined(node, :parent) ? node.parent : nothing

"""
$(TYPEDSIGNATURES)

Construct a tree decomposition abstract tree based on the graph `g` using
`root` as the root node.

# Example
```
using LightGraphs

g = double_binary_tree(3)
root = Node(1)
convertToAbstractTree!(g, root)
print_tree(root)
```
"""
function convertToAbstractTree!(g::MetaGraph, root::Node, parent::Node=root)

  # Is the parent node the root?
  if root === parent
    # Yes, then include all parent's neighbors in the children set
    children_ids = neighbors(g, parent.id)
  else
    # No, then include all parent's neighbors except its parent in the children set
    grandparent = Base.parent(root, parent)
    children_ids = neighbors(g, parent.id) |> x -> setdiff(x, grandparent.id)
  end

  # Base case
  length(children_ids) == 0 && return

  children = addchildren!(children_ids, parent)
  map(child -> convertToAbstractTree!(g, root, child), children)

  return

end
