# Based on: https://github.com/JuliaCollections/AbstractTrees.jl/tree/master/examples

mutable struct Node{T}
  id::T
  children::Vector{Node{T}}
  parent::Node{T}

  # Root constructor
  Node{T}(id) where T = new{T}(id, Node{T}[])

  # Child node constructor
  Node{T}(id, parent::Node{T}) where T = new{T}(id, Node{T}[], parent)
end
Node(id) = Node{typeof(id)}(id)

function addchild!(id, parent::Node)
  node = typeof(parent)(id, parent)
  push!(parent.children, node)
  return node
end

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

