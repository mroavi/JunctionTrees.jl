using OMEinsumContractionOrders, OMEinsum

export popindices, TensorNetworksSolver, generate_tensors, slice_contract

struct TensorNetworksSolver{ET,MT<:AbstractArray}
    code::ET
    tensors::Vector{MT}
    fixedvertices::Dict{Int,Int}
end

function TensorNetworksSolver(factors::Vector{<:Factor}; openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    tensors = getfield.(factors, :vals)
    rawcode = EinCode([[factor.vars...] for factor in factors], collect(Int, openvertices))  # labels for edge tensors
    TensorNetworksSolver(rawcode, tensors; fixedvertices, optimizer, simplifier)
end
function TensorNetworksSolver(rawcode::EinCode, tensors::Vector{<:AbstractArray}; fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    code = optimize_code(rawcode, OMEinsum.get_size_dict(getixsv(rawcode), tensors), optimizer, simplifier)
    TensorNetworksSolver(code, tensors, fixedvertices)
end

function generate_tensors(gp::TensorNetworksSolver)
    fixedvertices = gp.fixedvertices
    isempty(fixedvertices) && return tensors
    ixs = getixsv(gp.code)
    map(gp.tensors, ixs) do t, ix
        dims = map(ixi->ixi ∉ keys(fixedvertices) ? Colon() : (fixedvertices[ixi]+1:fixedvertices[ixi]+1), ix)
        t[dims...]
    end
end

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
        xsi = ntuple(i->OMEinsumContractionOrders.take_slice(xs[i], it.ixsv[i], slicemap), length(xs))
        resi = eins_sliced(xsi...; size_info=it.size_dict_sliced, kwargs...)
        OMEinsumContractionOrders.fill_slice!(res, [it.iyv..., slices...], resi, slicemap)
    end
    return res
end
_drop_slicedim(se::NestedEinsum, slices) = OMEinsumContractionOrders.drop_slicedim(se, slices)
_drop_slicedim(se::SlicedEinsum, slices) = SlicedEinsum(se.slicing, OMEinsumContractionOrders.drop_slicedim(se.eins, slices))

# METHOD2: pop indices
function popindices!(code::NestedEinsum{DynamicEinCode{LT}}, indices, out) where LT
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

# METHOD3: back propagation
# the data structure storing intermediate `NestedEinsum` contraction results.
struct CacheTree{T}
    content::AbstractArray{T}
    siblings::Vector{CacheTree{T}}
end
function cached_einsum(se::SlicedEinsum, @nospecialize(xs), size_dict)
    if length(se.slicing) != 0
        @warn "Slicing is not supported for caching, got nslices = $(length(se.slicing))! Fallback to `NestedEinsum`."
    end
    return cached_einsum(se.eins, xs, size_dict)
end
function cached_einsum(code::NestedEinsum, @nospecialize(xs), size_dict)
    if OMEinsum.isleaf(code)
        y = xs[code.tensorindex]
        return CacheTree(y, CacheTree{eltype(y)}[])
    else
        caches = [cached_einsum(arg, xs, size_dict) for arg in code.args]
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
function generate_gradient_tree(code::NestedEinsum, cache::CacheTree{T}, dy::AbstractArray{T}, size_dict::Dict) where T
    if OMEinsum.isleaf(code)
        return CacheTree(dy, CacheTree{T}[])
    else
        xs = (getfield.(cache.siblings, :content)...,)
        dxs = ntuple(i -> OMEinsum.einsum_grad(OMEinsum.getixs(code.eins), xs, OMEinsum.getiy(code.eins), size_dict, conj(dy), i), length(xs))
        return CacheTree(dy, generate_gradient_tree.(code.args, cache.siblings, dxs, Ref(size_dict)))
    end
end

function gradient_tree(code, xs)
    size_dict = OMEinsum.get_size_dict!(getixsv(code), xs, Dict{Int,Int}())
    cache = cached_einsum(code, xs, size_dict)
    dy = fill!(similar(cache.content), one(eltype(cache.content)))
    return generate_gradient_tree(code, cache, dy, size_dict)
end

function gradient(code, xs)
    tree = gradient_tree(code, xs)
    return extract_leaves(code, tree)
end

extract_leaves(code::SlicedEinsum, cache::CacheTree) = extract_leaves(code.eins, cache)
function extract_leaves(code::NestedEinsum, cache::CacheTree)
    res = Vector{Any}(undef, length(getixsv(code)))
    return extract_leaves!(code, cache, res)
end

function extract_leaves!(code, cache, res)
    if OMEinsum.isleaf(code)
        res[code.tensorindex] = cache.content
    else
        extract_leaves!.(code.args, cache.siblings, Ref(res))
    end
    return res
end