# Blocked solves with many right-hand sides for the banded LU of BandedMatrices.jl.
#
# `lu(::BandedMatrix)` calls LAPACK's gbtrf, and `ldiv!` calls gbtrs, which treats one
# right-hand side at a time (matrix-vector operations). The factorisation of a bordered
# matrix solves with p right-hand sides at once (W = A⁻ᵀC and V = A⁻¹B), so here the same
# factors are applied by blocks of nb columns: a small triangular solve (trsm) and a
# matrix-matrix product (gemm) per block, with the same row interchanges as gbtrs, hence
# the same partial pivoting and the same result up to rounding.
#
# The factors are in LAPACK band storage, AB of size (2kl + ku + 1) × n with kv = kl + ku:
#   U[i, j] = AB[kv + 1 + i - j, j]          for j - kv ≤ i ≤ j
#   L's multipliers of column j: AB[kv + 1 + k, j], k = 1, ..., min(kl, n - j), stored
#   in LINPACK form, i.e. not permuted by the later row interchanges.
module BorderedMatricesBandedMatricesExt

using BorderedMatrices, BandedMatrices, LinearAlgebra
import BorderedMatrices: _solve!, _tsolve!

const NB = 64       # block size: 32–128 perform alike, larger is slower

const KLMIN = 48    # below this lower bandwidth gbtrs is as fast or faster (benchmark/solves.jl)

_solve!(F::BandedMatrices.BandedLU, X::StridedMatrix)  = _kl(F) < KLMIN ? ldiv!(F, X) : blocked_solve!(F, X)
_tsolve!(F::BandedMatrices.BandedLU, X::StridedMatrix) = _kl(F) < KLMIN ? ldiv!(transpose(F), X) : blocked_tsolve!(F, X)

# X ← A⁻¹ X and X ← A⁻ᵀ X, by blocks
blocked_solve!(F, X)  = (_lsolve!(F, X); _usolve!(F, X); X)
blocked_tsolve!(F, X) = (_utsolve!(F, X); _ltsolve!(F, X); X)

_data(F) = BandedMatrices.bandeddata(F.factors)
_kl(F) = bandwidths(F.factors)[1]
_kv(F) = bandwidths(F.factors)[2]                # kl + ku

_swaprows!(X, i, k) = i == k || for c in axes(X, 2)
    X[i, c], X[k, c] = X[k, c], X[i, c]
end

# The block of L for columns j0:j1, as a dense (j1 - j0 + 1 + kl) × (j1 - j0 + 1) unit lower
# trapezoidal matrix, with the multipliers of each column permuted by the row interchanges
# of the later columns of the block (LAPACK form within the block).
function _lblock!(L, AB, ipiv, kv, kl, n, j0, j1)
    fill!(L, 0)
    for c in 1:(j1 - j0 + 1)
        j = j0 + c - 1
        p = ipiv[j] - j0 + 1
        p == c || for cc in 1:c-1
            L[c, cc], L[p, cc] = L[p, cc], L[c, cc]
        end
        L[c, c] = 1
        for k in 1:min(kl, n - j)
            L[c + k, c] = AB[kv + 1 + k, j]
        end
    end
    return L
end

# X ← L⁻¹ P X: for each block, apply its row interchanges, solve with the unit lower
# triangular block and update the kl rows below
function _lsolve!(F, X)
    AB, ipiv, kl, kv, n = _data(F), F.ipiv, _kl(F), _kv(F), size(X, 1)
    Lw = zeros(eltype(X), NB + kl, NB)
    for j0 in 1:NB:n-1
        j1 = min(j0 + NB - 1, n - 1); w = j1 - j0 + 1; e = min(j1 + kl, n)
        L = _lblock!(view(Lw, 1:e-j0+1, 1:w), AB, ipiv, kv, kl, n, j0, j1)
        for j in j0:j1
            _swaprows!(X, j, ipiv[j])
        end
        XJ = view(X, j0:j1, :)
        ldiv!(UnitLowerTriangular(view(L, 1:w, 1:w)), XJ)
        e > j1 && mul!(view(X, j1+1:e, :), view(L, w+1:e-j0+1, 1:w), XJ, -1, 1)
    end
    return X
end

# X ← Lᵀ⁻¹-part of A⁻ᵀ: the blocks in reverse order, each the transpose of the step above
function _ltsolve!(F, X)
    AB, ipiv, kl, kv, n = _data(F), F.ipiv, _kl(F), _kv(F), size(X, 1)
    Lw = zeros(eltype(X), NB + kl, NB)
    for j0 in reverse(1:NB:n-1)
        j1 = min(j0 + NB - 1, n - 1); w = j1 - j0 + 1; e = min(j1 + kl, n)
        L = _lblock!(view(Lw, 1:e-j0+1, 1:w), AB, ipiv, kv, kl, n, j0, j1)
        XJ = view(X, j0:j1, :)
        e > j1 && mul!(XJ, transpose(view(L, w+1:e-j0+1, 1:w)), view(X, j1+1:e, :), -1, 1)
        ldiv!(transpose(UnitLowerTriangular(view(L, 1:w, 1:w))), XJ)
        for j in j1:-1:j0
            _swaprows!(X, j, ipiv[j])
        end
    end
    return X
end

# U[i0:i1, j0:j1] as a dense matrix (zero outside the band)
function _ublock!(U, AB, kv, i0, i1, j0, j1)
    fill!(U, 0)
    for j in j0:j1, i in max(i0, j - kv):min(i1, j)
        U[i - i0 + 1, j - j0 + 1] = AB[kv + 1 + i - j, j]
    end
    return U
end

# X ← U⁻¹ X: blocks from the bottom; solve with the diagonal block, update the kv rows above
function _usolve!(F, X)
    AB, kv, n = _data(F), _kv(F), size(X, 1)
    Dw = zeros(eltype(X), NB, NB); Aw = zeros(eltype(X), kv, NB)
    for i1 in n:-NB:1
        i0 = max(1, i1 - NB + 1); w = i1 - i0 + 1; a = max(1, i0 - kv)
        XI = view(X, i0:i1, :)
        ldiv!(UpperTriangular(_ublock!(view(Dw, 1:w, 1:w), AB, kv, i0, i1, i0, i1)), XI)
        a < i0 && mul!(view(X, a:i0-1, :), _ublock!(view(Aw, 1:i0-a, 1:w), AB, kv, a, i0-1, i0, i1), XI, -1, 1)
    end
    return X
end

# X ← U⁻ᵀ X: blocks from the top; solve with the transposed diagonal block, update the kv rows below
function _utsolve!(F, X)
    AB, kv, n = _data(F), _kv(F), size(X, 1)
    Dw = zeros(eltype(X), NB, NB); Bw = zeros(eltype(X), NB, kv)
    for i0 in 1:NB:n
        i1 = min(i0 + NB - 1, n); w = i1 - i0 + 1; b = min(n, i1 + kv)
        XI = view(X, i0:i1, :)
        ldiv!(transpose(UpperTriangular(_ublock!(view(Dw, 1:w, 1:w), AB, kv, i0, i1, i0, i1))), XI)
        b > i1 && mul!(view(X, i1+1:b, :), transpose(_ublock!(view(Bw, 1:w, 1:b-i1), AB, kv, i0, i1, i1+1, b)), XI, -1, 1)
    end
    return X
end

end
