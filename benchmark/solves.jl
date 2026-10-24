# Speed-up of the blocked banded solves (BorderedMatrices' extension for BandedMatrices.jl)
# over LAPACK's gbtrs, for the p = h r + 1 right-hand sides that the factorisation of a
# cyclic block-banded bordered matrix needs (W = A⁻ᵀC and V = A⁻¹B). N_s = 100 slices.
#
#     julia --project=benchmark benchmark/solves.jl      (writes benchmark/results_solves.csv)

using BorderedMatrices, BandedMatrices, LinearAlgebra, SparseArrays, Random, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "matrices.jl"))

const ext = Base.get_extension(BorderedMatrices, :BorderedMatricesBandedMatricesExt)
best(f, k) = minimum(@elapsed(f()) for _ in 1:k)
Ns = 100
out = open(joinpath(@__DIR__, "results_solves.csv"), "w")
println(out, "r,h,Ns,n,p,kl,ku,gbtrs,gbtrs_T,blocked,blocked_T")
for h in (1, 2, 4), r in (5, 10, 20, 50, 100, 200)
    K = cyclic_matrix(r, Ns, h; rng = MersenneTwister(0))
    N = size(K, 1); p = h*r + 1; n = N - p
    F = lu(BandedMatrix(K[1:n, 1:n]))
    kl, ku = bandwidths(F.factors)
    X = randn(n, p)
    k = r <= 50 ? 5 : 2
    t  = best(() -> ldiv!(F, copy(X)), k)
    tt = best(() -> ldiv!(transpose(F), copy(X)), k)
    b  = best(() -> ext.blocked_solve!(F, copy(X)), k)          # always blocked, whatever kl
    bt = best(() -> ext.blocked_tsolve!(F, copy(X)), k)
    println(out, join((r, h, Ns, n, p, kl, ku, t, tt, b, bt), ","))
    @printf "h = %d, r = %3d (n = %5d, p = %4d): gbtrs %.4f / %.4f s, blocked %.4f / %.4f s, speed-up %.1f / %.1f\n" h r n p t tt b bt t/b tt/bt
    flush(stdout); flush(out)
end
close(out)
