# ------------------------------------------------------------------- #
# Copyright 2015-2026, Davide Lasagna                                 #
# ------------------------------------------------------------------- #
"""
    BorderedMatrices

Direct solution of bordered linear systems

    ⎡ A   B ⎤ ⎡ x ⎤   ⎡ f ⎤
    ⎣ Cᵀ  D ⎦ ⎣ y ⎦ = ⎣ g ⎦

by block elimination, reusing a factorisation of `A`. The blocks are stored
arrays with a common element type, real or complex: `A` is any matrix that can
be factorised (dense, sparse, a `BandedMatrix`, ...), `B`, `C` and `D` are dense
matrices, `x`, `y`, `f` and `g` are vectors.
"""
module BorderedMatrices

using LinearAlgebra

export BorderedVector, BorderedMatrix, BorderedMatrixLU, BandedMatrix, BandedLU, blockcyclic

include("types.jl")
include("linalg.jl")
include("banded.jl")   # BandedMatrix and its LU, with blocked solves
include("cyclic.jl")   # cyclic block-banded matrices as bordered matrices

end
