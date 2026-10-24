# BorderedVector and BorderedMatrix. All blocks have the same element type T.

"""
    BorderedVector(x, y)

The vector `[x; y]`, with `x` of length `n` and `y` of length `p`.
"""
struct BorderedVector{T} <: AbstractVector{T}
    x::Vector{T}
    y::Vector{T}
end

Base.size(v::BorderedVector) = (length(v.x) + length(v.y),)
Base.getindex(v::BorderedVector, i::Int) =
    i <= length(v.x) ? v.x[i] : v.y[i - length(v.x)]
Base.setindex!(v::BorderedVector, val, i::Int) =
    i <= length(v.x) ? (v.x[i] = val) : (v.y[i - length(v.x)] = val)
Base.copy(v::BorderedVector) = BorderedVector(copy(v.x), copy(v.y))
Base.similar(v::BorderedVector) = BorderedVector(similar(v.x), similar(v.y))

"""
    BorderedMatrix(A, B, C, D)

The matrix `[A B; Cᵀ D]`: `A` is `n × n` (any matrix type whose factorisation
supports `ldiv!` with the matrix and with its transpose), `B` and `C` are
`n × p` and `D` is `p × p`. Note that the bottom block is `Cᵀ`, the transpose
of `C`, also for complex matrices.
"""
struct BorderedMatrix{T, M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    A::M
    B::Matrix{T}
    C::Matrix{T}
    D::Matrix{T}
    function BorderedMatrix(A::M, B::Matrix{T}, C::Matrix{T}, D::Matrix{T}) where {T, M<:AbstractMatrix{T}}
        n, p = size(A, 1), size(D, 1)
        size(A) == (n, n) && size(B) == size(C) == (n, p) && size(D) == (p, p) ||
            throw(DimensionMismatch("blocks must be n × n, n × p, n × p and p × p"))
        return new{T, M}(A, B, C, D)
    end
end

Base.size(M::BorderedMatrix) = size(M.A) .+ size(M.D)

function Base.getindex(M::BorderedMatrix, i::Int, j::Int)
    n = size(M.A, 1)
    i <= n && j <= n && return M.A[i, j]
    i <= n && return M.B[i, j - n]
    j <= n && return M.C[j, i - n]
    return M.D[i - n, j - n]
end
