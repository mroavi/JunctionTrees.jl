"""
$(TYPEDSIGNATURES)

Speed up the sum-product in the algorithm with einsum contraction routines in `OMEinsum`.

# Examples

```jldoctest
algo = compile_algo("../examples/problems/paskin/paskin.uai");
algo2 = boost_algo(algo)
eval(algo2)

# output
```
"""
function boost_algo(algo::Expr; optimizer=GreedyMethod())
    info = Dict{Symbol, Vector{Int}}()
    size_dict = Dict{Int,Int}()
    return Expr(algo.head, algo.args[1], boost_ex(algo.args[2], info, size_dict, optimizer))
end

# boost a single statement
function boost_ex(ex::Expr, info, size_dict, optimizer)
    MLStyle.@match ex begin
        # multiple
        :(begin $(body...) end) => Expr(:block, boost_ex.(body, Ref(info), Ref(size_dict), Ref(optimizer))...)
        # sum-prod
        :($var = sum(prod($(tensors...)), $(labels...))) => begin
            ixs = map(t->info[t], tensors)
            # NOTE: sort because the prod sort the indices automatically
            iy = sort(setdiff(∪(ixs...), collect(Int, labels)))
            code = EinCode(ixs, iy)
            optcode = optimize_code(code, size_dict, optimizer)
            info[var] = iy
            :($var = $einsum($optcode, ($(tensors...),), $size_dict))
        end
        # sum (not optimized)
        :($var = sum($tensor, $(labels...))) => begin
            info[var] = setdiff(info[tensor], labels)
            ex
        end
        # prod (not optimized)
        :($var = prod($(tensors...))) => begin
            # NOTE: sort because the prod sort the indices automatically
            info[var] = sort(∪(getindex.(Ref(info), tensors)...))
            ex
        end
        :($var = redu($tensor, $vars, $vals)) => begin
            # reduce size
            for v in vars
                size_dict[v] = 1
            end
            info[var] = Int[tensor.vars...]
            ex
        end
        :($var = $target) => begin
            # assignment
            if target isa Factor
                info[var] = Int[target.vars...]
                for (var, sz) in zip(target.vars, size(target.vals))
                    size_dict[var] = sz
                end
                ex
            else
                # identity (not optimized)
                @assert target isa Symbol
                ex
            end
        end
        # norm (not optimized)
        :(norm.($args)) => ex
        _=>error("$ex is not handled.")
    end
end

for CT in [:DynamicEinCode, :StaticEinCode, :NestedEinsum, :SlicedEinsum]
    @eval function OMEinsum.einsum(neinsum::$CT, @nospecialize(xs::NTuple{N,Factor} where N), size_dict::Dict)
        # TODO: remove this patch after fixing rank-0 factor.
        tensors = map(x-> x isa (Factor{T,0} where T) ? fill(one(eltype(x.vals))) : x.vals, xs)
        return Factor((OMEinsum.getiyv(neinsum)...,), einsum(neinsum, tensors, size_dict))
    end
end

