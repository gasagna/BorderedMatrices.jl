# Banded LU factorisation: LAPACK's gbtrf, with solves by blocks for many right-hand sides.
#
# The factors are in LAPACK band storage, AB of size (2kl + ku + 1) × n, with kv = kl + ku:
#   A[i, j]  = AB[kv + 1 + i - j, j] on input, for j - ku ≤ i ≤ j + kl;
#   U[i, j]  = AB[kv + 1 + i - j, j] on output, for j - kv ≤ i ≤ j (the row interchanges
#              widen U by kl);
#   the multipliers of L in column j are AB[kv + 1 + k, j], k = 1, ..., min(kl, n - j), in
#   LINPACK form: not permuted by the row interchanges of the later columns.
#
# LAPACK's gbtrs solves one right-hand side at a time, with matrix-vector operations. For
# many right-hand sides the solves below work by blocks of NB columns of L and rows of U,
# with a triangular solve (trsm) and a matrix product (gemm) per block, and the same row
# interchanges as gbtrs: same pivoting, same result up to rounding.

const NB = 64       # block size: 32–128 perform alike
const KLMIN = 48    # below this lower bandwidth gbtrs is as fast or faster

"""
    BandedLU

LU factorisation with partial pivoting of a banded matrix, from `bandedlu`.
"""
struct BandedLU{T} <: Factorization{T}
    AB::Matrix{T}
    ipiv::Vector{LinearAlgebra.BlasInt}
    kl::Int
    ku::Int
end

"""
    bandedlu(A, kl, ku)

LU factorisation with partial pivoting (LAPACK `gbtrf`) of the square matrix `A`,
of any type, with `kl` sub- and `ku` super-diagonals. Entries outside the band are
ignored. The factors take `(2kl + ku + 1) × n` numbers. `ldiv!` and `\\` solve
with the matrix and with its transpose, `transpose(F)`.
"""
function bandedlu(A::AbstractMatrix{T}, kl::Int, ku::Int) where {T}
    n = size(A, 1)
    AB = zeros(T, 2kl + ku + 1, n)
    for j in 1:n, i in max(1, j - ku):min(n, j + kl)
        AB[kl + ku + 1 + i - j, j] = A[i, j]
    end
    return bandedlu!(AB, kl, ku)
end

"""
    bandedlu!(AB, kl, ku)

As `bandedlu`, for a matrix already in LAPACK band storage: `AB` is
`(2kl + ku + 1) × n`, with `A[i, j]` in `AB[kl + ku + 1 + i - j, j]` and the first
`kl` rows free. `AB` is overwritten with the factors. Assembling `A` in this form
avoids any conversion.
"""
bandedlu!(AB::Matrix{T}, kl::Int, ku::Int) where {T} =
    BandedLU{T}(LAPACK.gbtrf!(kl, ku, size(AB, 2), AB)..., kl, ku)

Base.size(F::BandedLU) = (size(F.AB, 2), size(F.AB, 2))

# X ← A⁻¹ X and X ← A⁻ᵀ X: gbtrs for vectors and narrow bands, by blocks otherwise
function LinearAlgebra.ldiv!(F::BandedLU{T}, X::StridedVecOrMat{T}) where {T}
    X isa AbstractVector || F.kl < KLMIN ?
        LAPACK.gbtrs!('N', F.kl, F.ku, size(F.AB, 2), F.AB, F.ipiv, X) : (_lsolve!(F, X); _usolve!(F, X))
    return X
end
# transpose(F) of a real F is an AdjointFactorization in LinearAlgebra
const TransposedBandedLU{T} = Union{LinearAlgebra.TransposeFactorization{T, BandedLU{T}},
                                    LinearAlgebra.AdjointFactorization{T, BandedLU{T}}}
function LinearAlgebra.ldiv!(Ft::TransposedBandedLU{T}, X::StridedVecOrMat{T}) where {T}
    F = parent(Ft)
    c = Ft isa LinearAlgebra.AdjointFactorization && T <: Complex ? 'C' : 'T'
    X isa AbstractVector || F.kl < KLMIN || c == 'C' ?
        LAPACK.gbtrs!(c, F.kl, F.ku, size(F.AB, 2), F.AB, F.ipiv, X) : (_utsolve!(F, X); _ltsolve!(F, X))
    return X
end
Base.:\(F::BandedLU, X::StridedVecOrMat) = ldiv!(F, copy(X))
Base.:\(Ft::TransposedBandedLU, X::StridedVecOrMat) = ldiv!(Ft, copy(X))

_swaprows!(X, i, k) = i == k || for c in axes(X, 2)
    X[i, c], X[k, c] = X[k, c], X[i, c]
end

# The block of L for columns j0:j1, as a dense (j1 - j0 + 1 + kl) × (j1 - j0 + 1) unit lower
# trapezoidal matrix, the multipliers of each column permuted by the row interchanges of
# the later columns of the block (LAPACK form within the block).
function _lblock!(L, F, j0, j1)
    AB, kv, n = F.AB, F.kl + F.ku, size(F.AB, 2)
    fill!(L, 0)
    for c in 1:(j1 - j0 + 1)
        j = j0 + c - 1
        p = F.ipiv[j] - j0 + 1
        p == c || for cc in 1:c-1
            L[c, cc], L[p, cc] = L[p, cc], L[c, cc]
        end
        L[c, c] = 1
        for k in 1:min(F.kl, n - j)
            L[c + k, c] = AB[kv + 1 + k, j]
        end
    end
    return L
end

# U[i0:i1, j0:j1] as a dense matrix (zero outside the band)
function _ublock!(U, F, i0, i1, j0, j1)
    AB, kv = F.AB, F.kl + F.ku
    fill!(U, 0)
    for j in j0:j1, i in max(i0, j - kv):min(i1, j)
        U[i - i0 + 1, j - j0 + 1] = AB[kv + 1 + i - j, j]
    end
    return U
end

# X ← L⁻¹ P X: for each block, apply its row interchanges, solve with the unit lower
# triangular block and update the kl rows below
function _lsolve!(F, X)
    kl, n = F.kl, size(X, 1)
    Lw = zeros(eltype(X), NB + kl, NB)
    for j0 in 1:NB:n-1
        j1 = min(j0 + NB - 1, n - 1); w = j1 - j0 + 1; e = min(j1 + kl, n)
        L = _lblock!(view(Lw, 1:e-j0+1, 1:w), F, j0, j1)
        for j in j0:j1
            _swaprows!(X, j, F.ipiv[j])
        end
        XJ = view(X, j0:j1, :)
        ldiv!(UnitLowerTriangular(view(L, 1:w, 1:w)), XJ)
        e > j1 && mul!(view(X, j1+1:e, :), view(L, w+1:e-j0+1, 1:w), XJ, -1, 1)
    end
    return X
end

# the transpose of the step above: the blocks in reverse order
function _ltsolve!(F, X)
    kl, n = F.kl, size(X, 1)
    Lw = zeros(eltype(X), NB + kl, NB)
    for j0 in reverse(1:NB:n-1)
        j1 = min(j0 + NB - 1, n - 1); w = j1 - j0 + 1; e = min(j1 + kl, n)
        L = _lblock!(view(Lw, 1:e-j0+1, 1:w), F, j0, j1)
        XJ = view(X, j0:j1, :)
        e > j1 && mul!(XJ, transpose(view(L, w+1:e-j0+1, 1:w)), view(X, j1+1:e, :), -1, 1)
        ldiv!(transpose(UnitLowerTriangular(view(L, 1:w, 1:w))), XJ)
        for j in j1:-1:j0
            _swaprows!(X, j, F.ipiv[j])
        end
    end
    return X
end

# X ← U⁻¹ X: blocks from the bottom; solve with the diagonal block, update the kv rows above
function _usolve!(F, X)
    kv, n = F.kl + F.ku, size(X, 1)
    Dw = zeros(eltype(X), NB, NB); Aw = zeros(eltype(X), kv, NB)
    for i1 in n:-NB:1
        i0 = max(1, i1 - NB + 1); w = i1 - i0 + 1; a = max(1, i0 - kv)
        XI = view(X, i0:i1, :)
        ldiv!(UpperTriangular(_ublock!(view(Dw, 1:w, 1:w), F, i0, i1, i0, i1)), XI)
        a < i0 && mul!(view(X, a:i0-1, :), _ublock!(view(Aw, 1:i0-a, 1:w), F, a, i0-1, i0, i1), XI, -1, 1)
    end
    return X
end

# X ← U⁻ᵀ X: blocks from the top; solve with the transposed diagonal block, update the kv rows below
function _utsolve!(F, X)
    kv, n = F.kl + F.ku, size(X, 1)
    Dw = zeros(eltype(X), NB, NB); Bw = zeros(eltype(X), NB, kv)
    for i0 in 1:NB:n
        i1 = min(i0 + NB - 1, n); w = i1 - i0 + 1; b = min(n, i1 + kv)
        XI = view(X, i0:i1, :)
        ldiv!(transpose(UpperTriangular(_ublock!(view(Dw, 1:w, 1:w), F, i0, i1, i0, i1))), XI)
        b > i1 && mul!(view(X, i1+1:b, :), transpose(_ublock!(view(Bw, 1:w, 1:b-i1), F, i0, i1, i1+1, b)), XI, -1, 1)
    end
    return X
end
