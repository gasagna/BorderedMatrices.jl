# Sparsity patterns for the README figure: a small cyclic block-banded matrix K with one
# dense border row and column, the nonzeros of UMFPACK's factors of K, and the storage of
# the bordered factorisation (banded LU of A, dense W and V).
#
#     julia --project=benchmark benchmark/sparsity.jl     (writes benchmark/sparsity.txt)

using BorderedMatrices, LinearAlgebra, SparseArrays, Random

include(joinpath(@__DIR__, "matrices.jl"))

r, Ns, h = 4, 24, 2
f, b, c, d, K = cyclic_problem(r, Ns, h; rng = MersenneTwister(1))
N = size(K, 1); p = h*r + 1; n = N - p

F = lu(K)                                   # UMFPACK, with its own row and column ordering
LU = F.L + F.U                              # pattern of the factors, in UMFPACK's ordering
Fb = lu(blockcyclic(f, r, Ns, h, b, c, d))
kl, ku = Fb.A.kl, Fb.A.kl + Fb.A.ku          # bandwidths of the banded factors (ku grows by kl)

open(joinpath(@__DIR__, "sparsity.txt"), "w") do io
    println(io, "N n p kl ku nnzK nnzLU")
    println(io, join((N, n, p, kl, ku, nnz(K), nnz(LU)), " "))
    I, J, _ = findnz(K);  println(io, join(I, " ")); println(io, join(J, " "))
    I, J, _ = findnz(LU); println(io, join(I, " ")); println(io, join(J, " "))
end
println("N = $N, n = $n, p = $p, nnz(K) = $(nnz(K)), nnz(L + U) = $(nnz(LU)), banded factors: kl = $kl, ku = $ku")
