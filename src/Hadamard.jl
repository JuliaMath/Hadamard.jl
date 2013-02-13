# Compute fast Walsh-Hadamard transforms (in natural, dyadic, or sequency order)
# using FFTW, by interpreting them as 2x2x2...x2x2 DFTs.  We follow Matlab's
# convention in which ifwht is the unnormalized transform and fwht has a 1/N
# normalization (as opposed to using a unitary normalization).

module Hadamard
export fwht, ifwht, fwht_natural, ifwht_natural, fwht_dyadic, ifwht_dyadic, hadamard

importall Base.FFTW
import FFTW.set_timelimit
import FFTW.dims_howmany
import FFTW.Plan
import FFTW.execute
import FFTW.complexfloat
import FFTW.normalization

power_of_two(n::Integer) = n > 0 && (n & (n - 1)) == 0

# A power-of-two dimension to be transformed is interpreted as a
# 2x2x2x....x2x2  multidimensional DFT.  This function transforms
# each individual power-of-two dimension n into the corresponding log2(n)
# DFT dimensions.  If bitreverse is true, then the output is in
# bit-reversed (transposed) order (which may not work in-place).
function hadamardize(dims::Array{Int,2}, bitreverse::Bool)
    ntot = 0
    for i = 1:size(dims,2)
        n = dims[1,i]
        if !power_of_two(n)
            throw(ArgumentError("non power-of-two Hadamard-transform length"))
        end
        ntot += trailing_zeros(n)
    end
    hdims = Array(Int,3,ntot)
    j = 0
    for i = 1:size(dims,2)
        n = dims[1,i]
        is = dims[2,i]
        os = dims[3,i]
        log2n = trailing_zeros(n)
        krange = j+1:j+log2n
        for k in krange
            hdims[1,k] = 2
            hdims[2,k] = is
            hdims[3,k] = os
            is *= 2
            os *= 2
        end
        if bitreverse
            hdims[3,krange] = fliplr(hdims[3,krange])
        end
        j += log2n
    end
    return hdims
end

for (Tr,Tc,fftw,lib) in ((:Float64,:Complex128,"fftw",FFTW.libfftw),
                         (:Float32,:Complex64,"fftwf",FFTW.libfftwf))
    @eval function Plan_Hadamard(X::StridedArray{$Tc}, Y::StridedArray{$Tc},
                                 region, flags::Unsigned, timelimit::Real, 
                                 bitreverse::Bool)
        set_timelimit($Tr, timelimit)
        dims, howmany = dims_howmany(X, Y, [size(X)...], region)
        dims = hadamardize(dims, bitreverse)
        plan = ccall(($(string(fftw,"_plan_guru64_dft")),$lib),
                     Ptr{Void},
                     (Int32, Ptr{Int}, Int32, Ptr{Int},
                      Ptr{$Tc}, Ptr{$Tc}, Int32, Uint32),
                     size(dims,2), dims, size(howmany,2), howmany,
                     X, Y, FFTW.FORWARD, flags)
        set_timelimit($Tr, NO_TIMELIMIT)
        if plan == C_NULL
            error("FFTW could not create plan") # shouldn't normally happen
        end
        return Plan(plan, X)
    end

    @eval function Plan_Hadamard(X::StridedArray{$Tr}, Y::StridedArray{$Tr},
                                 region, flags::Unsigned, timelimit::Real,
                                 bitreverse::Bool)
        set_timelimit($Tr, timelimit)
        dims, howmany = dims_howmany(X, Y, [size(X)...], region)
        dims = hadamardize(dims, bitreverse)
        kind = Array(Int32, size(dims,2))
        kind[:] = R2HC
        plan = ccall(($(string(fftw,"_plan_guru64_r2r")),$lib),
                     Ptr{Void},
                     (Int32, Ptr{Int}, Int32, Ptr{Int},
                      Ptr{$Tr}, Ptr{$Tr}, Ptr{Int32}, Uint32),
                     size(dims,2), dims, size(howmany,2), howmany,
                     X, Y, kind, flags)
        set_timelimit($Tr, NO_TIMELIMIT)
        if plan == C_NULL
            error("FFTW could not create plan") # shouldn't normally happen
        end
        return Plan(plan, X)
    end
end

############################################################################
# Define ifwht (inverse/unnormalized) transforms for various orderings

# Natural (Hadamard) ordering:

function ifwht_natural{T<:fftwNumber}(X::StridedArray{T}, region)
    Y = similar(X)
    p = Plan_Hadamard(X, Y, region, ESTIMATE, NO_TIMELIMIT, false)
    execute(T, p.plan)
    return Y
end

function ifwht_natural{T<:Number}(X::StridedArray{T}, region)
    Y = T<:Complex ? complexfloat(X) : float(X)
    p = Plan_Hadamard(Y, Y, region, ESTIMATE, NO_TIMELIMIT, false)
    execute(p.plan, Y, Y)
    return Y
end

# Dyadic (Paley, bit-reversed) ordering:

function ifwht_dyadic{T<:fftwNumber}(X::StridedArray{T}, region)
    Y = similar(X)
    p = Plan_Hadamard(X, Y, region, ESTIMATE, NO_TIMELIMIT, true)
    execute(T, p.plan)
    return Y
end

function ifwht_dyadic{T<:Number}(X::StridedArray{T}, region)
    return ifwht_dyadic(T<:Complex ? complexfloat(X) : float(X), region)
end

############################################################################
# Sequency (Walsh) ordering:

# ifwht along a single dimension d of X
function ifwht{T<:fftwNumber}(X::Array{T}, region)
    Y = ifwht_dyadic(X, region)

    # Perform Gray-code permutation of Y (TODO: in-place?)
    if isempty(region)
        return Y
    elseif ndims(Y) == 1
        return [ Y[1 + ((i >> 1) $ i)] for i = 0:length(Y)-1 ]
    else
        sz = [size(Y)...]
        tmp = Array(T, max(sz[region])) # storage for out-of-place permutations
        for d in region
            # somewhat ugly loops to do 1d permutations along dimension d
            na = prod(sz[d+1:end])
            n = sz[d]
            nb = prod(sz[1:d-1])
            sa = nb * n
            for ia = 0:na-1
                for ib = 1:nb
                    i0 = ib + sa * ia
                    for i = 0:n-1
                        tmp[i+1] = Y[i0 + nb * ((i >> 1) $ i)]
                    end
                    for i = 0:n-1
                        Y[i0 + nb * i] = tmp[i+1]
                    end
                end
            end
        end
        return Y
    end
end

# handle 1d case of strided arrays (loops in multidim case are too annoying)
function ifwht{T<:fftwNumber}(X::StridedVector{T}, region)
    Y = ifwht_dyadic(X, region)

    # Perform Gray-code permutation of Y (TODO: in-place?)
    if isempty(region)
        return Y
    else
        return [ Y[1 + ((i >> 1) $ i)] for i = 0:length(Y)-1 ]
    end
end

# fallback for subarrays
function ifwht{T<:fftwNumber}(X::StridedArray{T}, region)
    return ifwht(copy(X), region)
end

# fallback for other types
function ifwht{T<:Number}(X::StridedArray{T}, region)
    return ifwht(T<:Complex ? complexfloat(X) : float(X), region)
end

############################################################################
# Forward transforms (normalized by 1/N as in Matlab) and transforms
# without the region argument:

for f in (:ifwht_natural, :ifwht_dyadic, :ifwht)
    g = symbol(string(f)[2:end])
    @eval begin
        $f(X) = $f(X, 1:ndims(X))
        $g(X) = scale!($f(X), normalization(X))
        $g(X,r) = scale!($f(X,r), normalization(X,r))
    end
end

############################################################################
# create (floating-point) Hadamard matrix, similar to Matlab function,
# in natural order.

function hadamard(n::Integer)
    if power_of_two(n)
        return ifwht_natural(eye(n), 1)
    else
        # Punt.  Matlab supports power-of-two multiples of 12 or 20,
        # but lots of other sizes are known as well (see e.g. Sloane's
        # web page http://neilsloane.com/hadamard/) and there is no
        # particularly efficient way to construct them other than
        # a look-up-table of known factors.  [Given Hadamard matrices Hm
        # and Hn of sizes m and n, a Hadamard matrix of size mn is kron(Hm,Hn).]
        # It is not really clear what factors are important to support,
        # nor which of the (generally nonunique) Hadamard matrices to pick.
        throw(ArgumentError("non power-of-two Hadamard matrices aren't supported"))
    end
end

############################################################################

end # module
