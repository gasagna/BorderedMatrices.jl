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

Factorise `M`. `factorize` is applied to `A` (`lu` for a dense, sparse or
`BandedMatrix` `A`; `lu!` to factorise it in place). `alg` is `:BEM`, the mixed
block elimination of Govaerts and Pryce, stable when `A` is nearly singular, or
`:BED`, the plain block elimination, which is inaccurate when `A` is
ill-conditioned but does not store `A⁻¹B`.
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

struct BEDFactorization{T, F} <: BorderedMatrixLU{T}
    A::F                            # factorisation of A
    B::Matrix{T}
    W::Matrix{T}                    # A⁻ᵀ C
    S::LU{T, Matrix{T}, Vector{Int}}  # D - Wᵀ B
end

function BEDFactorization(Af, M::BorderedMatrix{T}) where {T}
    W = ldiv!(transpose(Af), copy(M.C))
    S = lu(M.D - transpose(W) * M.B)
    return BEDFactorization{T, typeof(Af)}(Af, M.B, W, S)
end

function LinearAlgebra.ldiv!(F::BEDFactorization{T}, r::BorderedVector{T}) where {T}
    f, g = r.x, r.y
    y = F.S \ (g - transpose(F.W) * f)
    mul!(f, F.B, y, -1, 1)          # f - B y
    ldiv!(F.A, f)                   # x
    g .= y
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

struct BEMFactorization{T, F} <: BorderedMatrixLU{T}
    A::F                            # factorisation of A
    B::Matrix{T}
    C::Matrix{T}
    D::Matrix{T}
    W::Matrix{T}                    # A⁻ᵀ C
    V::Matrix{T}                    # A⁻¹ B
    Sw::LU{T, Matrix{T}, Vector{Int}} # D - Wᵀ B
    Sv::LU{T, Matrix{T}, Vector{Int}} # D - Cᵀ V
end

function BEMFactorization(Af, M::BorderedMatrix{T}) where {T}
    W  = ldiv!(transpose(Af), copy(M.C))
    V  = ldiv!(Af, copy(M.B))
    Sw = lu(M.D - transpose(W) * M.B)
    Sv = lu(M.D - transpose(M.C) * V)
    return BEMFactorization{T, typeof(Af)}(Af, M.B, M.C, M.D, W, V, Sw, Sv)
end

function LinearAlgebra.ldiv!(F::BEMFactorization{T}, r::BorderedVector{T}) where {T}
    f, g = r.x, r.y
    y₁ = F.Sw \ (g - transpose(F.W) * f)
    mul!(f, F.B, y₁, -1, 1)         # f₁
    g₁ = g - F.D * y₁
    ldiv!(F.A, f)                   # ξ
    y₂ = F.Sv \ (g₁ - transpose(F.C) * f)
    mul!(f, F.V, y₂, -1, 1)         # x
    g .= y₁ .+ y₂
    return r
end
