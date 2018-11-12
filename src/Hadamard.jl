# Compute fast Walsh-Hadamard transforms (in natural, dyadic, or sequency order)
# using FFTW, by interpreting them as 2x2x2...x2x2 DFTs.  We follow Matlab's
# convention in which ifwht is the unnormalized transform and fwht has a 1/N
# normalization (as opposed to using a unitary normalization).
VERSION < v"0.7.0-beta2.199" && __precompile__()

"""
The `Hadamard` module provides functions to compute fast Walsh-Hadamard transforms in Julia,
for arbitrary dimensions and arbitrary power-of-two transform sizes, with the three standard
orderings: natural (Hadamard), dyadic (Paley), and sequency (Walsh) ordering.
See the `fwht`, `fwht_natural`, and `fwht_dyadic` functions, along with their inverses
`ifwht` etc.

There is also a function `hadamard(n)` that returns Hadamard matrices for known sizes
(including non powers of two).
"""
module Hadamard
export fwht, ifwht, fwht_natural, ifwht_natural, fwht_natural!, ifwht_natural!, fwht_dyadic, ifwht_dyadic, hadamard

using FFTW
import FFTW: set_timelimit, dims_howmany, unsafe_execute!, cFFTWPlan, r2rFFTWPlan, PlanPtr, fftwNumber, ESTIMATE, NO_TIMELIMIT, R2HC
import AbstractFFTs: normalization, complexfloat

using Compat

# A power-of-two dimension to be transformed is interpreted as a
# 2x2x2x....x2x2  multidimensional DFT.  This function transforms
# each individual power-of-two dimension n into the corresponding log2(n)
# DFT dimensions.  If bitreverse is true, then the output is in
# bit-reversed (transposed) order (which may not work in-place).
function hadamardize(dims::Array{Int,2}, bitreverse::Bool)
    ntot = 0
    for i = 1:size(dims,2)
        n = dims[1,i]
        if !ispow2(n)
            throw(ArgumentError("non power-of-two Hadamard-transform length"))
        end
        ntot += trailing_zeros(n)
    end
    hdims = Array{Int}(undef, 3,ntot)
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
            hdims[3,krange] = reverse(hdims[3,krange])
        end
        j += log2n
    end
    return hdims
end

const libfftw = isdefined(FFTW, :libfftw) ? FFTW.libfftw : FFTW.libfftw3
const libfftwf = isdefined(FFTW, :libfftwf) ? FFTW.libfftwf : FFTW.libfftw3f

for (Tr,Tc,fftw,lib) in ((:Float64,:ComplexF64,"fftw",libfftw),
                         (:Float32,:ComplexF32,"fftwf",libfftwf))
    @eval function Plan_Hadamard(X::StridedArray{$Tc,N}, Y::StridedArray{$Tc,N},
                                 region, flags::Unsigned, timelimit::Real,
                                 bitreverse::Bool) where {N}
        set_timelimit($Tr, timelimit)
        dims, howmany = dims_howmany(X, Y, [size(X)...], region)
        dims = hadamardize(dims, bitreverse)
        plan = ccall(($(string(fftw,"_plan_guru64_dft")),$lib),
                     PlanPtr,
                     (Int32, Ptr{Int}, Int32, Ptr{Int},
                      Ptr{$Tc}, Ptr{$Tc}, Int32, UInt32),
                     size(dims,2), dims, size(howmany,2), howmany,
                     X, Y, FFTW.FORWARD, flags)
        set_timelimit($Tr, NO_TIMELIMIT)
        if plan == C_NULL
            if $(occursin("libmkl", lib))
                error("MKL is not supported — reconfigure FFTW.jl to use FFTW")
            else
                error("FFTW could not create plan") # shouldn't normally happen
            end
        end
        return cFFTWPlan{$Tc,FFTW.FORWARD,X===Y,N}(plan, flags, region, X, Y)
    end

    @eval function Plan_Hadamard(X::StridedArray{$Tr,N}, Y::StridedArray{$Tr,N},
                                 region, flags::Unsigned, timelimit::Real,
                                 bitreverse::Bool) where {N}
        set_timelimit($Tr, timelimit)
        dims, howmany = dims_howmany(X, Y, [size(X)...], region)
        dims = hadamardize(dims, bitreverse)
        kind = Array{Int32}(undef, size(dims,2))
        kind .= R2HC
        plan = ccall(($(string(fftw,"_plan_guru64_r2r")),$lib),
                     PlanPtr,
                     (Int32, Ptr{Int}, Int32, Ptr{Int},
                      Ptr{$Tr}, Ptr{$Tr}, Ptr{Int32}, UInt32),
                     size(dims,2), dims, size(howmany,2), howmany,
                     X, Y, kind, flags)
        set_timelimit($Tr, NO_TIMELIMIT)
        if plan == C_NULL
            if $(occursin("libmkl", lib))
                error("MKL is not supported — reconfigure FFTW.jl to use FFTW")
            else
                error("FFTW could not create plan") # shouldn't normally happen
            end
        end
        return r2rFFTWPlan{$Tr,(map(Int,kind)...,),X===Y,N}(plan, flags, region, X, Y)
    end
end

############################################################################
# Define ifwht (inverse/unnormalized) transforms for various orderings

# Natural (Hadamard) ordering:

function ifwht_natural(X::StridedArray{<:fftwNumber}, region)
    Y = similar(X)
    p = Plan_Hadamard(X, Y, region, ESTIMATE, NO_TIMELIMIT, false)
    unsafe_execute!(p)
    return Y
end

function ifwht_natural(X::StridedArray{<:Number}, region)
    Y = float(X)
    p = Plan_Hadamard(Y, Y, region, ESTIMATE, NO_TIMELIMIT, false)
    unsafe_execute!(p)
    return Y
end

# Dyadic (Paley, bit-reversed) ordering:

function ifwht_dyadic(X::StridedArray{<:fftwNumber}, region)
    Y = similar(X)
    p = Plan_Hadamard(X, Y, region, ESTIMATE, NO_TIMELIMIT, true)
    unsafe_execute!(p)
    return Y
end

function ifwht_dyadic(X::StridedArray{<:Number}, region)
    return ifwht_dyadic(float(X), region)
end

############################################################################
# Sequency (Walsh) ordering:

# ifwht along a single dimension d of X
function ifwht(X::Array{T}, region) where {T<:fftwNumber}
    Y = ifwht_dyadic(X, region)

    # Perform Gray-code permutation of Y (TODO: in-place?)
    if isempty(region)
        return Y
    elseif ndims(Y) == 1
        return [ Y[1 + ((i >> 1) ⊻ i)] for i = 0:length(Y)-1 ]
    else
        sz = [size(Y)...]
        tmp = Array{T}(undef, maximum(sz[region])) # storage for out-of-place perm.
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
                        tmp[i+1] = Y[i0 + nb * ((i >> 1) ⊻ i)]
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
function ifwht(X::StridedVector{<:fftwNumber}, region)
    Y = ifwht_dyadic(X, region)

    # Perform Gray-code permutation of Y (TODO: in-place?)
    if isempty(region)
        return Y
    else
        return [ Y[1 + ((i >> 1) ⊻ i)] for i = 0:length(Y)-1 ]
    end
end

# fallback for subarrays
function ifwht(X::StridedArray{<:fftwNumber}, region)
    return ifwht(copy(X), region)
end

# fallback for other types
function ifwht(X::StridedArray{<:Number}, region)
    return ifwht(float(X), region)
end

############################################################################
# in-place transforms, currently for natural ordering only.

"""
    ifwht_natural!(X, dims=1:ndims(X))

Similar to `ifwht_natural`, but works in-place on the input array `X`.
"""
function ifwht_natural!(X::StridedArray{<:fftwNumber}, region=1:ndims(X))
    p = Plan_Hadamard(X, X, region, ESTIMATE, NO_TIMELIMIT, false)
    unsafe_execute!(p)
    return X
end


"""
    fwht_natural!(X, dims=1:ndims(X))

Similar to `fwht_natural`, but works in-place on the input array `X`.
"""
fwht_natural!(X::StridedArray{<:fftwNumber}, region=1:ndims(X)) =
    Compat.rmul!(ifwht_natural!(X, region), normalization(X,region))

############################################################################
# Forward transforms (normalized by 1/N as in Matlab) and transforms
# without the region argument:

for f in (:ifwht_natural, :ifwht_dyadic, :ifwht)
    g = Symbol(string(f)[2:end])
    @eval begin
        $f(X) = $f(X, 1:ndims(X))
        $g(X) = Compat.rmul!($f(X), normalization(X,1:ndims(X)))
        $g(X,r) = Compat.rmul!($f(X,r), normalization(X,r))
    end
end

############################################################################
# Docstrings — since they are all very similar, we
# assign them in a loop.

for inverse in (true, false), ord in (:natural, :dyadic, :sequency)
    name = (inverse ? "i" : "") * "fwht"
    if ord != :sequency
        name = string(name, '_', ord)
    end
    ordstr = ord == :natural ? "natural (Hadamard)" :
             ord == :dyadic ? "dyadic (Paley, bit-reversed)" :
             "sequency"
    docstring = """    $name(X, dims=1:ndims(X))

Return the$(inverse ? " inverse" : "") fast Walsh-Hadamard transform (WHT) of
the array `X` along the dimensions `dims` (an integer,
tuple, or array of dimensions, defaulting to all dimensions).
Only power-of-two sizes (along the transformed dimensions)
are supported.  The result is returned in the $ordstr ordering.

Our WHT is normalized so that the forward transform has a `1/n`
coefficient (where `n` is the product of the transformed dimensions)
and the inverse WHT has no scaling factor.
"""
    @eval @doc $docstring $(Symbol(name))
end

############################################################################
# Utilities to work with a precomputed cache of known Hadamard matrices
# of various sizes, produced by util/fetchhadamard.jl from Sloane's web page
# and stored as BitMatrices

function readcache(cachefile::AbstractString)
    B = BitMatrix[]
    open(cachefile, "r") do io
        while !eof(io)
            k = convert(Int, ntoh(read(io, Int64)))
            b = BitArray(undef, k, k)
            b.chunks .= ntoh.(read!(io, b.chunks))
            push!(B, b)
        end
    end
    sort!(B, by = h -> (isodd(size(h,1)>>2), size(h,1)))
    return B
end
readcache() = readcache(joinpath(dirname(@__FILE__), "..", "src", "cache.dat"))

"""
    printsigns([io, ] A)

Print a table of `+` and `-` characters indicating the
signs of the entries of `A` (with `0` for zero entries).
The output stream `io` defaults to `stdout`.
"""
function printsigns(io::IO, A::AbstractMatrix{<:Real})
    m, n = size(A)
    println(io, m, "×", n, " sign matrix from ", typeof(A))
    for i = 1:m
        for j = 1:n
            print(io, A[i,j] > 0 ? '+' : A[i,j] < 0 ? '-' : '0')
        end
        println(io)
    end
end
printsigns(A) = printsigns(stdout, A)

function frombits(B::BitMatrix)
    A = Matrix{Int8}(B)
    for i = 1:length(A)
        A[i] = A[i] == 0 ? -1 : 1
    end
    return A
end

const hadamards = BitMatrix[] # read lazily below

############################################################################
# Create Int8 order-n Hadamard matrix (in natural order), by factorizing
# n into a product of known Hadamard sizes (if possible)

const H2 = Int8[1 1; 1 -1]

"""
    hadamard(n)

Return a Hadamard matrix of order `n`.  The known Hadamard
matrices up to size 256 are currently supported (via a lookup table),
along with any size that factorizes into products of these known
sizes and/or powers of two.

The return value is a matrix of `Int8` values. If you want to do
further matrix computations with this matrix, you may want to
convert to `Float64` first via `float(hadamard(n))`.

You can pretty-print a Hadamard matrix as a table of `+` and `-`
(indicating the signs of the entries) via `Hadamard.printsigns`,
e.g. `Hadamard.printsigns(hadamard(28))` for the 28×28 Hadamard matrix.
"""
function hadamard(n::Integer)
    n < 1 && throw(ArgumentError("size n=$n should be positive"))
    n0 = n
    H = reshape(Int8[1], 1,1)
    if !ispow2(n)
        isempty(hadamards) && append!(hadamards, readcache())
        for i = length(hadamards):-1:1
            k = size(hadamards[i], 1)
            if rem(n, k) == 0
                Hk = frombits(hadamards[i])
                while true
                    H = kron(H, Hk)
                    n = div(n, k)
                    rem(n, k) != 0 && break
                end
            end
        end
    end
    if !ispow2(n)
        n >>= trailing_zeros(n)
        throw(ArgumentError("unknown Hadamard factor $n of $n0"))
    end
    # Note: it would be faster to do a "power-by-squaring" like algorithm
    # here, where we repeatedly double the order via kron(H, H), but
    # it's not clear to me that we care much about performance here.
    while n > 1
        H = kron(H, H2)
        n >>= 1
    end
    return H
end

############################################################################

end # module
