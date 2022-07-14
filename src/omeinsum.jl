using OMEinsum

export popindices, TensorNetworksSolver, generate_tensors, slice_contract

# code: the tensor network contraction pattern.
# tensors: the tensors fed into the tensor network.
# fixedvertices: the degree of freedoms fixed to a value.
struct TensorNetworksSolver{ET,MT<:AbstractArray}
    code::ET
    tensors::Vector{MT}
    fixedvertices::Dict{Int,Int}
end

function TensorNetworksSolver(factors::Vector{<:Factor}; openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    tensors = getfield.(factors, :vals)
    # The 1st argument of `EinCode` is a vector of vector of labels for specifying the input tensors, 
    # The 2nd argument of `EinCode` is a vector of labels for specifying the output tensor,
    # e.g.
    # `EinCode([[1, 2], [2, 3]], [1, 3])` is the EinCode for matrix multiplication.
    rawcode = EinCode([[factor.vars...] for factor in factors], collect(Int, openvertices))  # labels for edge tensors
    TensorNetworksSolver(rawcode, tensors; fixedvertices, optimizer, simplifier)
end
function TensorNetworksSolver(rawcode::EinCode, tensors::Vector{<:AbstractArray}; fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    # `optimize_code` optimizes the contraction order of a raw tensor network without a contraction order specified.
    # The 1st argument is the contraction pattern to be optimized (without contraction order).
    # The 2nd arugment is the size dictionary, which is a label-integer dictionary.
    # The 3rd and 4th arguments are the optimizer and simplifier that configures which algorithm to use and simplify.
    code = optimize_code(rawcode, OMEinsum.get_size_dict(getixsv(rawcode), tensors), optimizer, simplifier)
    TensorNetworksSolver(code, tensors, fixedvertices)
end

# generate tensors based on which vertices are fixed.
function generate_tensors(gp::TensorNetworksSolver)
    fixedvertices = gp.fixedvertices
    isempty(fixedvertices) && return tensors
    ixs = getixsv(gp.code)
    # `ix` is the vector of labels (or a degree of freedoms) for a tensor,
    # if a label in `ix` is fixed to a value, do the slicing to the tensor it associates to.
    map(gp.tensors, ixs) do t, ix
        dims = map(ixi->ixi ∉ keys(fixedvertices) ? Colon() : (fixedvertices[ixi]+1:fixedvertices[ixi]+1), ix)
        t[dims...]
    end
end

########################### The following algorithms are not relevant anymore #######################
# METHOD1: slice indices
function slice_contract(se::AbstractEinsum, slices, @nospecialize(xs::AbstractArray{T}...); size_info = nothing, kwargs...) where T
    length(slices) == 0 && return se(xs...; size_info=size_info, kwargs...)
    size_dict = size_info===nothing ? Dict{OMEinsum.labeltype(se),Int}() : copy(size_info)
    OMEinsum.get_size_dict!(se, xs, size_dict)

    it = OMEinsumContractionOrders.SliceIterator(getixsv(se), getiyv(se), slices, size_dict)
    # slices are the last several dimensions
    res = zeros(T, getindex.(Ref(size_dict), (it.iyv..., slices...)))
    eins_sliced = _drop_slicedim(se, OMEinsumContractionOrders.Slicing(slices))
    for (k, slicemap) in enumerate(it)
        @debug "computing slice $k/$(length(it))"
        xsi = ntuple(i->OMEinsum.take_slice(xs[i], it.ixsv[i], slicemap), length(xs))
        resi = eins_sliced(xsi...; size_info=it.size_dict_sliced, kwargs...)
        OMEinsum.fill_slice!(res, [it.iyv..., slices...], resi, slicemap)
    end
    return res
end
_drop_slicedim(se::NestedEinsum, slices) = OMEinsum.drop_slicedim(se, slices)
_drop_slicedim(se::SlicedEinsum, slices) = SlicedEinsum(se.slicing, OMEinsum.drop_slicedim(se.eins, slices))

# METHOD2: pop indices
function popindices!(code::DynamicNestedEinsum{LT}, indices, out) where LT
    OMEinsum.isleaf(code) && return
    for (arg,ix) in zip(code.args, code.eins.ixs)
        popindices!(arg, indices, ix)
        # loop over `indices` rather than `iy` because we want to keep the added indices ordered!
        for l in indices
            if l ∈ ix
                l ∉ code.eins.iy && push!(code.eins.iy, l)
                l ∉ out && push!(out, l)
            end
        end
    end
end
popindices!(se::SlicedEinsum, indices, out) = popindices!(se.eins, indices, out)
popindices(code, indices) = (code = deepcopy(code); popindices!(code, indices, []); code)

########################### The above algorithms are not relevant anymore #######################

# METHOD3: back propagation
# `CacheTree` stores intermediate `NestedEinsum` contraction results.
# It is a tree structure that isomorphic to the contraction tree,
# `siblings` are the siblings of current node.
# `content` is the cached intermediate contraction result.
struct CacheTree{T}
    content::AbstractArray{T}
    siblings::Vector{CacheTree{T}}
end
function cached_einsum(se::SlicedEinsum, @nospecialize(xs), size_dict)
    # slicing is not supported yet.
    if length(se.slicing) != 0
        @warn "Slicing is not supported for caching, got nslices = $(length(se.slicing))! Fallback to `NestedEinsum`."
    end
    return cached_einsum(se.eins, xs, size_dict)
end

# recursively contract and cache a tensor network
function cached_einsum(code::NestedEinsum, @nospecialize(xs), size_dict)
    if OMEinsum.isleaf(code)
        # For a leaf node, cache the input tensor
        y = xs[code.tensorindex]
        return CacheTree(y, CacheTree{eltype(y)}[])
    else
        # For a non-leaf node, compute the einsum and cache the contraction result
        caches = [cached_einsum(arg, xs, size_dict) for arg in code.args]
        # `einsum` evaluates the einsum contraction,
        # Its 1st argument is the contraction pattern,
        # Its 2nd one is a tuple of input tensors,
        # Its 3rd argument is the size dictionary (label as the key, size as the value).
        y = einsum(code.eins, ntuple(i->caches[i].content, length(caches)), size_dict)
        return CacheTree(y, caches)
    end
end

# computed gradient tree by back propagation
function generate_gradient_tree(se::SlicedEinsum, cache::CacheTree{T}, dy::AbstractArray{T}, size_dict::Dict) where T
    if length(se.slicing) != 0
        @warn "Slicing is not supported for generating masked tree! Fallback to `NestedEinsum`."
    end
    return generate_gradient_tree(se.eins, cache, dy, size_dict)
end

# recursively compute the gradients and store it into a tree.
# also known as the back-propagation algorithm.
function generate_gradient_tree(code::NestedEinsum, cache::CacheTree{T}, dy::AbstractArray{T}, size_dict::Dict) where T
    if OMEinsum.isleaf(code)
        return CacheTree(dy, CacheTree{T}[])
    else
        xs = (getfield.(cache.siblings, :content)...,)
        # `einsum_grad` is the back-propagation rule for einsum function.
        # If the forward pass is `y = einsum(EinCode(inputs_labels, output_labels), (A, B, ...), size_dict)`
        # Then the back-propagation pass is
        # ```
        # A̅ = einsum_grad(inputs_labels, (A, B, ...), output_labels, size_dict, y̅, 1)
        # B̅ = einsum_grad(inputs_labels, (A, B, ...), output_labels, size_dict, y̅, 2)
        # ...
        # ```
        # Let `L` be the loss, we will have `y̅ := ∂L/∂y`, `A̅ := ∂L/∂A`...
        dxs = ntuple(i -> OMEinsum.einsum_grad(OMEinsum.getixs(code.eins), xs, OMEinsum.getiy(code.eins), size_dict, conj(dy), i), length(xs))
        return CacheTree(dy, generate_gradient_tree.(code.args, cache.siblings, dxs, Ref(size_dict)))
    end
end

# the main function for generating the gradient tree.
function gradient_tree(code, xs)
    # infer size from the contraction code and the input tensors `xs`, returns a label-size dictionary.
    size_dict = OMEinsum.get_size_dict!(getixsv(code), xs, Dict{Int,Int}())
    # forward compute and cache intermediate results.
    cache = cached_einsum(code, xs, size_dict)
    # initialize `y̅` as `1`. Note we always start from `L̅ := 1`.
    dy = fill!(similar(cache.content), one(eltype(cache.content)))
    # back-propagate
    return generate_gradient_tree(code, cache, dy, size_dict)
end

function gradient(code, xs)
    tree = gradient_tree(code, xs)
    # extract the gradients on leaves (i.e. the input tensors).
    return extract_leaves(code, tree)
end

# since slicing is not supported, we forward it to NestedEinsum.
extract_leaves(code::SlicedEinsum, cache::CacheTree) = extract_leaves(code.eins, cache)

function extract_leaves(code::NestedEinsum, cache::CacheTree)
    res = Vector{Any}(undef, length(getixsv(code)))
    return extract_leaves!(code, cache, res)
end

function extract_leaves!(code, cache, res)
    if OMEinsum.isleaf(code)
        # extract
        res[code.tensorindex] = cache.content
    else
        # resurse deeper
        extract_leaves!.(code.args, cache.siblings, Ref(res))
    end
    return res
end