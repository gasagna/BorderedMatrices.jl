# Factorisation of bordered matrices and solution of bordered systems
#
#     ⎡ A   B ⎤ ⎡ x ⎤   ⎡ f ⎤
#     ⎣ Cᵀ  D ⎦ ⎣ y ⎦ = ⎣ g ⎦
#
# by block elimination, reusing a factorisation of A:
#   1. the public interface, lu and \;
#   2. BED, plain block elimination: factorisation and solve;
#   3. BEM, mixed block elimination: factorisation and solve.

"""
    BorderedMatrixLU{T}

Factorisation of a `BorderedMatrix`, returned by `lu`: either a
`BEDFactorization` or a `BEMFactorization`. Everything that depends only on the
matrix is computed once, so that each solve costs one solve with `A`.
"""
abstract type BorderedMatrixLU{T} <: Factorization{T} end


# ~~~ 1. Public interface ~~~

"""
    lu(M::BorderedMatrix; alg = :BEM, factorize = lu)

Factorise `M`. `factorize` is applied to `A`: `lu` for a dense, sparse or
`BandedMatrix` `A`, or `lu!` to factorise `A` in place. `alg` is `:BEM`, the mixed block elimination of Govaerts and Pryce, stable
when `A` is nearly singular, or `:BED`, the plain block elimination, which is
inaccurate when `A` is ill-conditioned but does not store `A⁻¹B`.
"""
function LinearAlgebra.lu(M::BorderedMatrix; alg::Symbol = :BEM, factorize = lu)
    alg === :BEM && return BEMFactorization(factorize(M.A), M)
    alg === :BED && return BEDFactorization(factorize(M.A), M)
    throw(ArgumentError("alg must be :BEM or :BED"))
end

Base.:\(F::BorderedMatrixLU, r::BorderedVector) = ldiv!(F, copy(r))


# ~~~ 2. BED: plain block elimination ~~~
#
#   W = A⁻ᵀ C
#   y = (D - WᵀB)⁻¹ (g - Wᵀ f)
#   x = A⁻¹ (f - B y)

struct BEDFactorization{T, F, N<:AbstractMatrix{T}} <: BorderedMatrixLU{T}
    A::F                            # factorisation of A
    B::N
    W::Matrix{T}                    # A⁻ᵀ C
    S::LU{T, Matrix{T}, Vector{Int}}  # D - Wᵀ B
end

function BEDFactorization(Af, M::BorderedMatrix{T}) where {T}
    W = ldiv!(transpose(Af), Matrix(M.C))
    S = lu(M.D - transpose(W) * M.B)
    return BEDFactorization{T, typeof(Af), typeof(M.B)}(Af, M.B, W, S)
end

# no allocations: g is overwritten with y, then f with x
function LinearAlgebra.ldiv!(F::BEDFactorization{T}, r::BorderedVector{T}) where {T}
    f, g = r.x, r.y
    mul!(g, transpose(F.W), f, -one(T), one(T))   # g - Wᵀ f
    ldiv!(F.S, g)                                 # y
    mul!(f, F.B, g, -one(T), one(T))              # f - B y
    ldiv!(F.A, f)                                 # x
    return r
end


# ~~~ 3. BEM: mixed block elimination (Govaerts and Pryce) ~~~
#
#   W = A⁻ᵀ C,  V = A⁻¹ B
#   y₁ = (D - WᵀB)⁻¹ (g - Wᵀ f)          first pass, as BED
#   f₁ = f - B y₁,  g₁ = g - D y₁        residual of the first pass
#   ξ  = A⁻¹ f₁
#   y₂ = (D - CᵀV)⁻¹ (g₁ - Cᵀ ξ)         correction, with the other Schur complement
#   x  = ξ - V y₂,  y = y₁ + y₂

struct BEMFactorization{T, F, N<:AbstractMatrix{T}} <: BorderedMatrixLU{T}
    A::F                            # factorisation of A
    B::N
    C::N
    D::Matrix{T}
    W::Matrix{T}                    # A⁻ᵀ C
    V::Matrix{T}                    # A⁻¹ B
    Sw::LU{T, Matrix{T}, Vector{Int}} # D - Wᵀ B
    Sv::LU{T, Matrix{T}, Vector{Int}} # D - Cᵀ V
    y₁::Vector{T}                   # workspace, length p
end

function BEMFactorization(Af, M::BorderedMatrix{T}) where {T}
    W  = ldiv!(transpose(Af), Matrix(M.C))      # W and V are dense even if B and C are sparse
    V  = ldiv!(Af, Matrix(M.B))
    Sw = lu(M.D - transpose(W) * M.B)
    Sv = lu(M.D - transpose(M.C) * V)
    return BEMFactorization{T, typeof(Af), typeof(M.B)}(Af, M.B, M.C, M.D, W, V, Sw, Sv, zeros(T, size(M.D, 1)))
end

# no allocations: y₁ is kept in the workspace F.y₁; f is overwritten with f₁, ξ and
# then x; g with g₁, y₂ and then y. The workspace makes a factorisation unsafe to use
# from several threads at once.
function LinearAlgebra.ldiv!(F::BEMFactorization{T}, r::BorderedVector{T}) where {T}
    f, g, y₁ = r.x, r.y, F.y₁
    copyto!(y₁, g)
    mul!(y₁, transpose(F.W), f, -one(T), one(T))  # g - Wᵀ f
    ldiv!(F.Sw, y₁)                               # y₁
    mul!(f, F.B, y₁, -one(T), one(T))             # f₁ = f - B y₁
    mul!(g, F.D, y₁, -one(T), one(T))             # g₁ = g - D y₁
    ldiv!(F.A, f)                                 # ξ = A⁻¹ f₁
    mul!(g, transpose(F.C), f, -one(T), one(T))   # g₁ - Cᵀ ξ
    ldiv!(F.Sv, g)                                # y₂
    mul!(f, F.V, g, -one(T), one(T))              # x = ξ - V y₂
    g .+= y₁                                      # y = y₁ + y₂
    return r
end

