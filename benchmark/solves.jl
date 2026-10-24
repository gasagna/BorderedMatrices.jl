# Speed-up of the blocked banded solves of BandedLU over LAPACK's gbtrs, for the p = h r + 1 right-hand sides that the factorisation of a
# cyclic block-banded bordered matrix needs (W = A⁻ᵀC and V = A⁻¹B). N_s = 100 slices.
#
#     julia --project=benchmark benchmark/solves.jl      (writes benchmark/results_solves.csv)

using BorderedMatrices, LinearAlgebra, SparseArrays, Random, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "matrices.jl"))

const BM = BorderedMatrices
best(f, k) = minimum(@elapsed(f()) for _ in 1:k)
Ns = 100
out = open(joinpath(@__DIR__, "results_solves.csv"), "w")
println(out, "r,h,Ns,n,p,kl,ku,gbtrs,gbtrs_T,blocked,blocked_T")
for h in (1, 2, 4), r in (5, 10, 20, 50, 100, 200)
    f, b, c, d, _ = cyclic_problem(r, Ns, h; rng = MersenneTwister(0))
    M = blockcyclic(f, r, Ns, h, b, c, d)
    F = lu(M.A)
    n, p = size(M.A, 1), size(M.D, 1)
    kl, ku = F.kl, F.ku
    X = randn(n, p)
    k = r <= 50 ? 5 : 2
    t  = best(() -> LAPACK.gbtrs!('N', kl, ku, n, F.AB, F.ipiv, copy(X)), k)
    tt = best(() -> LAPACK.gbtrs!('T', kl, ku, n, F.AB, F.ipiv, copy(X)), k)
    b  = best(() -> (Y = copy(X); BM._lsolve!(F, Y); BM._usolve!(F, Y)), k)    # always blocked, whatever kl
    bt = best(() -> (Y = copy(X); BM._utsolve!(F, Y); BM._ltsolve!(F, Y)), k)
    println(out, join((r, h, Ns, n, p, kl, ku, t, tt, b, bt), ","))
    @printf "h = %d, r = %3d (n = %5d, p = %4d): gbtrs %.4f / %.4f s, blocked %.4f / %.4f s, speed-up %.1f / %.1f\n" h r n p t tt b bt t/b tt/bt
    flush(stdout); flush(out)
end
close(out)
