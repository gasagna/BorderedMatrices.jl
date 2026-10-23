# ------------------------------------------------------------------- #
# Copyright 2015-2026, Davide Lasagna                                 #
# ------------------------------------------------------------------- #
"""
    BorderedMatrices

Bordered linear systems

    / A   B \\ / x \\   / f \\
    |       | |   | = |   |
    \\ Cᵀ  D / \\ y /   \\ g /

with `A` an `n × n` matrix of any type that can be factorised, and borders
`B`, `C` of size `n × p` (or vectors, `p = 1`) and `D` of size `p × p` (or a
scalar). The factorisation of `A` is reused: the bordered system is solved by
block elimination, with the mixed method of Govaerts and Pryce (`:BEM`), stable
even when `A` is nearly singular, or with the plain Doolittle-type elimination
(`:BED`).

A `BandedLU` factorisation (LAPACK `gbtrf`) is provided for banded `A`, so that
a cyclic block-banded matrix can be solved as a banded matrix bordered by its
last unknowns, without fill outside the band.
"""
module BorderedMatrices

using LinearAlgebra, SparseArrays
using LinearAlgebra: BlasFloat, BlasInt

export BorderedVector, BorderedMatrix, BorderedMatrixLU,
       BandedLU, bandedlu, bandwidths, nborder

# ~~~ helpers: borders are stored as vectors/scalars when p = 1 ~~~
_asmat(b::AbstractVector) = reshape(b, :, 1)
_asmat(b::AbstractMatrix) = b
_asmat(d::Number) = fill(d, 1, 1)
_asvec(g::Number) = [g]
_asvec(g::AbstractVector) = g
_nb(b::AbstractVector) = 1
_nb(b::AbstractMatrix) = size(b, 2)
_nb(g::Number) = 1

# ~~~ Bordered Vector ~~~

"""
    BorderedVector(x, y)

A vector `x` (length `n`) followed by a border `y`: a scalar or a vector of
length `p`.
"""
mutable struct BorderedVector{T<:Number, V<:AbstractVector{T}, Y} <: AbstractVector{T}
    _1::V # main part
    _2::Y # border: a scalar of type T, or a vector with eltype T
    BorderedVector(v₁::V, v₂::Number) where {T, V<:AbstractVector{T}} =
        new{T, V, T}(v₁, convert(T, v₂))
    BorderedVector(v₁::V, v₂::AbstractVector) where {T, V<:AbstractVector{T}} =
        (w = convert(Vector{T}, v₂); new{T, V, Vector{T}}(v₁, w))
end

"""
    nborder(x)

Size `p` of the border of a `BorderedVector`, `BorderedMatrix` or
`BorderedMatrixLU`.
"""
nborder(v::BorderedVector) = v._2 isa Number ? 1 : length(v._2)

Base.size(v::BorderedVector) = (length(v._1) + nborder(v), )
Base.IndexStyle(::Type{<:BorderedVector}) = IndexLinear()

function Base.getindex(v::BorderedVector, i::Integer)
    n = length(v._1)
    1 <= i <= n && return v._1[i]
    n < i <= n + nborder(v) && return v._2[i - n]
    throw(BoundsError(v, i))
end

function Base.setindex!(v::BorderedVector, val, i::Integer)
    n = length(v._1)
    1 <= i <= n && (v._1[i] = val; return v)
    if n < i <= n + nborder(v)
        v._2 isa Number ? (v._2 = val) : (v._2[i - n] = val)
        return v
    end
    throw(BoundsError(v, i))
end

Base.copy(v::BorderedVector) = BorderedVector(copy(v._1), copy(v._2))
Base.similar(v::BorderedVector) =
    BorderedVector(similar(v._1), v._2 isa Number ? zero(v._2) : similar(v._2))

# ~~~ Bordered Matrix ~~~

"""
    BorderedMatrix(A, B, C, D)

The matrix `[A B; Cᵀ D]`. `A` is `n × n`; `B` and `C` are vectors of length `n`
(a single border) or `n × p` matrices; `D` is a scalar or a `p × p` matrix.
Note that `C` is stored as a column (block): the bottom row (block) is `Cᵀ`.

    BorderedMatrix(K, p)

Partition the square matrix `K` so that its last `p` rows and columns are the
border. For a sparse `K`, `A` stays sparse and the borders become dense.
"""
mutable struct BorderedMatrix{T<:Number, M<:AbstractMatrix{T}, V<:AbstractVecOrMat{T}, S} <: AbstractMatrix{T}
    _11::M # main top left - any matrix
    _12::V # right border: vector or n × p
    _21::V # bottom border, stored transposed: vector or n × p
    _22::S # bottom right: scalar or p × p
    function BorderedMatrix(M₁₁::M, M₁₂::V, M₂₁::V, M₂₂::Union{Number, AbstractMatrix}) where {T, M<:AbstractMatrix{T}, V<:AbstractVecOrMat{T}}
        n = size(M₁₁, 1)
        size(M₁₁, 2) == n || throw(DimensionMismatch("top left block must be square"))
        p = _nb(M₁₂)
        (size(M₁₂, 1) == n && size(M₂₁, 1) == n && _nb(M₂₁) == p) ||
            throw(DimensionMismatch("inconsistent border size"))
        if M₂₂ isa Number
            p == 1 || throw(DimensionMismatch("scalar corner needs a single border"))
            return new{T, M, V, T}(M₁₁, M₁₂, M₂₁, convert(T, M₂₂))
        else
            size(M₂₂) == (p, p) || throw(DimensionMismatch("corner block must be p × p"))
            D = convert(Matrix{T}, M₂₂)
            return new{T, M, V, Matrix{T}}(M₁₁, M₁₂, M₂₁, D)
        end
    end
end

function BorderedMatrix(K::AbstractMatrix{T}, p::Integer) where {T}
    N = size(K, 1)
    size(K, 2) == N || throw(DimensionMismatch("matrix must be square"))
    0 < p < N || throw(ArgumentError("need 0 < p < size(K, 1)"))
    n = N - p
    A = K[1:n, 1:n]
    # borders stay sparse if K is sparse (in a block-banded K they are mostly zero)
    B = K isa AbstractSparseMatrix ? K[1:n, n+1:N] : Matrix{T}(K[1:n, n+1:N])
    C = K isa AbstractSparseMatrix ? sparse(transpose(K[n+1:N, 1:n])) : Matrix{T}(transpose(K[n+1:N, 1:n]))
    D = Matrix{T}(K[n+1:N, n+1:N])
    return BorderedMatrix(A, B, C, D)
end

nborder(M::BorderedMatrix) = _nb(M._12)

Base.size(M::BorderedMatrix) = (size(M._11, 1) + nborder(M), size(M._11, 2) + nborder(M))
Base.IndexStyle(::Type{<:BorderedMatrix}) = IndexCartesian()

_bget(b::AbstractVector, i, k) = (k == 1 || throw(BoundsError(b, (i, k))); b[i])
_bget(b::AbstractMatrix, i, k) = b[i, k]
_dget(d::Number, i, j) = ((i, j) == (1, 1) || throw(BoundsError(d, (i, j))); d)
_dget(d::AbstractMatrix, i, j) = d[i, j]

function Base.getindex(M::BorderedMatrix, i::Integer, j::Integer)
    n = size(M._11, 1); N = size(M, 1)
    (1 <= i <= N && 1 <= j <= N) || throw(BoundsError(M, (i, j)))
    i <= n && j <= n && return M._11[i, j]
    i <= n && return _bget(M._12, i, j - n)
    j <= n && return _bget(M._21, j, i - n)
    return _dget(M._22, i - n, j - n)
end

function Base.setindex!(M::BorderedMatrix, val, i::Integer, j::Integer)
    n = size(M._11, 1); N = size(M, 1)
    (1 <= i <= N && 1 <= j <= N) || throw(BoundsError(M, (i, j)))
    if i <= n && j <= n
        M._11[i, j] = val
    elseif i <= n
        M._12 isa AbstractVector ? (M._12[i] = val) : (M._12[i, j - n] = val)
    elseif j <= n
        M._21 isa AbstractVector ? (M._21[j] = val) : (M._21[j, i - n] = val)
    else
        M._22 isa Number ? (M._22 = val) : (M._22[i - n, j - n] = val)
    end
    return val
end

Base.copy(M::BorderedMatrix) =
    BorderedMatrix(copy(M._11), copy(M._12), copy(M._21), copy(M._22))
Base.similar(M::BorderedMatrix) =
    BorderedMatrix(similar(M._11), similar(M._12), similar(M._21),
                   M._22 isa Number ? zero(M._22) : similar(M._22))

function Base.Matrix(M::BorderedMatrix{T}) where {T}
    n = size(M._11, 1)
    F = zeros(T, size(M))
    F[1:n, 1:n]         .= M._11
    F[1:n, n+1:end]     .= _asmat(M._12)
    F[n+1:end, 1:n]     .= transpose(_asmat(M._21))
    F[n+1:end, n+1:end] .= _asmat(M._22)
    return F
end

# matrix-vector product, mostly for checking residuals
function Base.:*(M::BorderedMatrix, x::BorderedVector)
    nborder(M) == nborder(x) || throw(DimensionMismatch("border sizes differ"))
    B, C, D = _asmat(M._12), _asmat(M._21), _asmat(M._22)
    y = _asvec(x._2)
    f = M._11 * x._1 .+ B * y
    g = transpose(C) * x._1 .+ D * y
    return BorderedVector(f, x._2 isa Number ? g[1] : g)
end

# ~~~ Banded LU (LAPACK gbtrf/gbtrs) ~~~

"""
    bandwidths(A) -> (kl, ku)

Number of sub- and super-diagonals of the sparse (or dense) matrix `A`.
"""
function bandwidths(A::SparseMatrixCSC)
    kl = ku = 0
    rows = rowvals(A)
    for j in 1:size(A, 2), k in nzrange(A, j)
        i = rows[k]
        kl = max(kl, i - j); ku = max(ku, j - i)
    end
    return kl, ku
end
bandwidths(A::AbstractMatrix) = bandwidths(sparse(A))

"""
    BandedLU

LU factorisation with partial pivoting of a banded matrix, in LAPACK band
storage. Build it with `bandedlu(A)`. Supports `ldiv!`, `\\` and solves with the
transpose, `transpose(F) \\ b`.
"""
struct BandedLU{T<:BlasFloat} <: Factorization{T}
    AB::Matrix{T}
    ipiv::Vector{BlasInt}
    kl::Int
    ku::Int
    n::Int
end

"""
    bandedlu(A; kl, ku)

Banded LU factorisation of the square matrix `A` (sparse or dense) with `kl`
sub- and `ku` super-diagonals, by default those of `A`. Storage is
`(2kl + ku + 1) × n`; there is no fill outside the band.
"""
function bandedlu(A::AbstractMatrix{T}; kl::Integer = -1, ku::Integer = -1) where {T<:BlasFloat}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("matrix must be square"))
    S = sparse(A)
    if kl < 0 || ku < 0
        kl_, ku_ = bandwidths(S)
        kl = kl < 0 ? kl_ : kl; ku = ku < 0 ? ku_ : ku
    end
    AB = zeros(T, 2kl + ku + 1, n)
    rows, vals = rowvals(S), nonzeros(S)
    for j in 1:n, k in nzrange(S, j)
        i = rows[k]
        (-ku <= i - j <= kl) || throw(ArgumentError("entry ($i, $j) outside the band"))
        AB[kl + ku + 1 + i - j, j] = vals[k]
    end
    AB, ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, n, AB)
    return BandedLU{T}(AB, ipiv, kl, ku, n)
end
bandedlu(A::AbstractMatrix{<:Integer}; kw...) = bandedlu(float(A); kw...)

Base.size(F::BandedLU) = (F.n, F.n)
Base.size(F::BandedLU, i::Integer) = i <= 2 ? F.n : 1

LinearAlgebra.ldiv!(F::BandedLU{T}, b::StridedVecOrMat{T}) where {T} =
    LinearAlgebra.LAPACK.gbtrs!('N', F.kl, F.ku, F.n, F.AB, F.ipiv, b)
Base.:\(F::BandedLU, b::AbstractVecOrMat) = ldiv!(F, copy_oftype(b, eltype(F)))

# solves with the transpose (and, for real matrices, the adjoint)
_tchar(::Type{<:Real}, adj) = 'T'
_tchar(::Type{<:Complex}, adj) = adj ? 'C' : 'T'
for (W, adj) in ((:TransposeFactorization, false), (:AdjointFactorization, true))
    @eval begin
        LinearAlgebra.ldiv!(Ft::LinearAlgebra.$W{T, <:BandedLU{T}}, b::StridedVecOrMat{T}) where {T} =
            (F = parent(Ft); LinearAlgebra.LAPACK.gbtrs!(_tchar(T, $adj), F.kl, F.ku, F.n, F.AB, F.ipiv, b))
        Base.:\(Ft::LinearAlgebra.$W{T, <:BandedLU{T}}, b::AbstractVecOrMat) where {T} =
            ldiv!(Ft, copy_oftype(b, T))
    end
end

copy_oftype(b::AbstractArray, ::Type{T}) where {T} = copyto!(similar(b, T), b)

# ~~~ Linear algebra for bordered systems ~~~

# solve in place with A and with Aᵀ, whatever the factorisation
_solve!(F::Union{LU, BandedLU}, X::StridedVecOrMat) = ldiv!(F, X)
_solve!(F, X::AbstractVecOrMat) = copyto!(X, F \ X)
_tsolve!(F::BandedLU{T}, X::StridedVecOrMat{T}) where {T} = ldiv!(transpose(F), X)
_tsolve!(F::LU, X::StridedVecOrMat) = ldiv!(transpose(F), X)
_tsolve!(F, X::AbstractVecOrMat) = copyto!(X, transpose(F) \ X)

"""
    BorderedMatrixLU

Factorisation of a `BorderedMatrix` `[A B; Cᵀ D]`: the factorisation of `A`,
the borders, and the precomputed `W = A⁻ᵀC`, `V = A⁻¹B` (`:BEM` only) and the
LU factorisations of the `p × p` Schur complements `D - WᵀB` and `D - CᵀV`.
Each solve then costs a single solve with `A`.
"""
struct BorderedMatrixLU{T<:Number, F, M<:AbstractMatrix{T}} <: Factorization{T}
    A::F            # factorisation of the top left block
    B::M            # right border, n × p (dense, or sparse if given sparse)
    C::M            # bottom border (transposed), n × p
    D::Matrix{T}    # corner, p × p
    W::Matrix{T}    # A⁻ᵀ C
    V::Matrix{T}    # A⁻¹ B (empty for :BED)
    Sw::LU{T, Matrix{T}, Vector{BlasInt}}   # D - Wᵀ B
    Sv::LU{T, Matrix{T}, Vector{BlasInt}}   # D - Cᵀ V (= Sw for :BED)
    alg::Symbol
    scalar::Bool    # the border was given as vectors/scalar
end

nborder(F::BorderedMatrixLU) = size(F.D, 1)
Base.size(F::BorderedMatrixLU) = (size(F.B, 1) + nborder(F), size(F.B, 1) + nborder(F))

function _factor(Af, M::BorderedMatrix{T}, alg::Symbol) where {T}
    alg in (:BEM, :BED) || throw(ArgumentError("alg must be :BEM or :BED"))
    B, C, D = _asmat(M._12), _asmat(M._21), Matrix{T}(_asmat(M._22))
    B isa AbstractSparseMatrix || (B = Matrix{T}(B); C = Matrix{T}(C))
    W  = _tsolve!(Af, Matrix{T}(C))
    Sw = lu(D - transpose(W) * B)
    if alg === :BEM
        V  = _solve!(Af, Matrix{T}(B))
        Sv = lu(D - transpose(C) * V)
    else
        V, Sv = zeros(T, 0, 0), Sw
    end
    return BorderedMatrixLU{T, typeof(Af), typeof(B)}(Af, B, C, D, W, V, Sw, Sv, alg, M._22 isa Number)
end

"""
    lu(M::BorderedMatrix; alg = :BEM, factorize = lu)

Factorise the bordered matrix `M`. `factorize` is applied to the top left
block: `lu` by default (dense or sparse), or e.g. `bandedlu`. `alg` selects the
block elimination used by the solves: `:BEM` (Govaerts--Pryce mixed block
elimination, stable for nearly singular `A`) or `:BED` (plain elimination,
cheaper in memory but unstable when `A` is ill-conditioned).
"""
LinearAlgebra.lu(M::BorderedMatrix; alg::Symbol = :BEM, factorize = lu) =
    _factor(factorize(M._11), M, alg)

"""
    lu!(M::BorderedMatrix; alg = :BEM)

As `lu`, but factorise the top left block in place (dense matrices only).
"""
LinearAlgebra.lu!(M::BorderedMatrix; alg::Symbol = :BEM) = _factor(lu!(M._11), M, alg)

# core: overwrite f (length n) and g (length p) with the solution
function _bsolve!(F::BorderedMatrixLU{T}, f::AbstractVector, g::AbstractVector) where {T}
    B, C, D, W = F.B, F.C, F.D, F.W
    if F.alg === :BED
        # y = (D - WᵀB)⁻¹ (g - Wᵀf);  x = A⁻¹ (f - B y)
        y = F.Sw \ (g .- transpose(W) * f)
        mul!(f, B, y, -one(T), one(T))
        _solve!(F.A, f)
        g .= y
    else
        # first pass as BED, then a correction with the other Schur complement
        y₁ = F.Sw \ (g .- transpose(W) * f)
        mul!(f, B, y₁, -one(T), one(T))          # f₁ = f - B y₁
        g₁ = g .- D * y₁                         # g₁ = g - D y₁
        _solve!(F.A, f)                          # ξ = A⁻¹ f₁
        y₂ = F.Sv \ (g₁ .- transpose(C) * f)     # y₂ = (D - CᵀV)⁻¹ (g₁ - Cᵀξ)
        mul!(f, F.V, y₂, -one(T), one(T))        # x = ξ - V y₂
        g .= y₁ .+ y₂
    end
    return f, g
end

function LinearAlgebra.ldiv!(F::BorderedMatrixLU, r::BorderedVector)
    nborder(F) == nborder(r) || throw(DimensionMismatch("border sizes differ"))
    g = _asvec(copy(r._2))
    _bsolve!(F, r._1, g)
    r._2 isa Number ? (r._2 = g[1]) : (r._2 .= g)
    return r
end

function LinearAlgebra.ldiv!(F::BorderedMatrixLU, r::AbstractVector)
    n = size(F.B, 1); p = nborder(F)
    length(r) == n + p || throw(DimensionMismatch("vector has the wrong length"))
    f = view(r, 1:n); g = r[n+1:n+p]
    _bsolve!(F, f, g)
    r[n+1:n+p] .= g
    return r
end

Base.:\(F::BorderedMatrixLU, r::BorderedVector) = ldiv!(F, copy(r))
Base.:\(F::BorderedMatrixLU, r::AbstractVector) = ldiv!(F, copy(r))

"""
    ldiv!(M::BorderedMatrix, r::BorderedVector, alg = :BEM, inplace = true)

Solve `M z = r`, overwriting `r`, factorising `M` first (in place if
`inplace`). Kept for compatibility with the original interface; to solve
several systems with the same matrix, factorise once with `lu`.
"""
LinearAlgebra.ldiv!(M::BorderedMatrix, r::BorderedVector, alg::Symbol = :BEM, inplace::Bool = true) =
    ldiv!(inplace ? lu!(M; alg = alg) : lu(M; alg = alg), r)

end
