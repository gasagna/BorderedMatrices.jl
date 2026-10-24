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
be factorised (dense, sparse, or banded with `bandedlu`, ...), `B`, `C` and `D` are dense
matrices, `x`, `y`, `f` and `g` are vectors.
"""
module BorderedMatrices

using LinearAlgebra

export BorderedVector, BorderedMatrix, BorderedMatrixLU, BandedLU, bandedlu, bandedlu!

include("types.jl")
include("linalg.jl")
include("banded.jl")   # banded LU with blocked solves

end
