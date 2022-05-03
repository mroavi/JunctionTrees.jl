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

