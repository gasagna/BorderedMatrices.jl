# Benchmark: cyclic block-banded matrices with one dense border row and column, solved
#   (a) with UMFPACK's sparse LU of the whole matrix, and
#   (b) assembled by blockcyclic as a bordered matrix, the last h slices and the extra
#       unknown in the border and the other slices in a BandedMatrix, with BEM,
#   (c) the same with plain block elimination (BED).
# Ns slices of r unknowns; each slice couples to the h slices on each side (periodically).
#
#     julia --project=benchmark benchmark/cyclic.jl [h]
#
# Writes benchmark/results_h<h>.csv: factorisation time, time of one solve (s), memory of the
# factorisation (MB), relative residual. benchmark/plot.py draws them.

using BorderedMatrices, LinearAlgebra, SparseArrays, Random, Printf
include(joinpath(@__DIR__, "matrices.jl"))

# best of a few runs, so that compilation and noise do not count
best(f, k) = minimum(@elapsed(f()) for _ in 1:k)

h = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 4
println("h = $h, BLAS threads = 1, $(Sys.cpu_info()[1].model)")
# optional: only some cases, appended to the existing file, e.g. ONLY="100:500,200:20"
only = [Tuple(parse.(Int, split(c, ":"))) for c in split(get(ENV, "ONLY", ""), ",", keepempty = false)]
out = open(joinpath(@__DIR__, "results_h$h.csv"), isempty(only) ? "w" : "a")
isempty(only) && println(out, "r,Ns,N,factor_umfpack,factor_bem,factor_bed,solve_umfpack,solve_bem,solve_bed,mem_umfpack,mem_bem,mem_bed,res_umfpack,res_bem,res_bed,matrix_K,matrix_M")
rng = MersenneTwister(0)
for r in (5, 10, 20, 50, 100, 200), Ns in (20, 50, 100, 200, 500, 1000)
    r^2 * Ns > 6_000_000 && continue            # keep each case within a few GB
    isempty(only) || (r, Ns) in only || continue
    f, bb, cc, dd, K = cyclic_problem(r, Ns, h; rng)
    M = blockcyclic(f, r, Ns, h, bb, cc, dd)       # assembled once; lu(M) copies A
    N = size(K, 1); n = size(M.A, 1)
    b = randn(rng, N)
    rb = BorderedVector(b[1:n], b[n+1:N])
    k = r^3 * Ns < 50_000_000 ? 3 : 1             # one run for the largest cases
    # (a) UMFPACK
    tfu = @elapsed(Fu = lu(K))                    # the first run is kept; more runs only if small
    k > 1 && (tfu = min(tfu, best(() -> lu(K), k - 1)))
    tsu = best(() -> Fu \ b, 20)
    xu  = Fu \ b
    mu  = 16 * (nnz(Fu.L) + nnz(Fu.U)) / 1e6           # values + row indices
    Fu  = nothing; GC.gc()
    # (b) BEM
    tfb = @elapsed(Fb = lu(M))
    k > 1 && (tfb = min(tfb, best(() -> lu(M), k - 1)))
    tsb = best(() -> Fb \ rb, 20)
    xb  = collect(Fb \ rb)
    mb  = (Base.summarysize(Fb.A) + sizeof(Fb.W) + sizeof(Fb.V)) / 1e6       # B, C, D belong to M
    Fb = nothing; GC.gc()
    # (c) the same with plain block elimination (no V = A⁻¹B)
    tfd = @elapsed(Fd = lu(M; alg = :BED))
    k > 1 && (tfd = min(tfd, best(() -> lu(M; alg = :BED), k - 1)))
    tsd = best(() -> Fd \ rb, 20)
    xd  = collect(Fd \ rb)
    md  = (Base.summarysize(Fd.A) + sizeof(Fd.W)) / 1e6
    res(x) = norm(K * x - b) / norm(b)
    mK = 16 * nnz(K) / 1e6                          # the matrices themselves (MB)
    mM = (sizeof(M.A.AB) + sizeof(M.B) + sizeof(M.C) + sizeof(M.D)) / 1e6
    println(out, join((r, Ns, N, tfu, tfb, tfd, tsu, tsb, tsd, mu, mb, md, res(xu), res(xb), res(xd), mK, mM), ","))
    @printf "r = %3d, Ns = %3d: factor %.3f / %.3f / %.3f s, solve %.2f / %.2f / %.2f ms, memory %.1f / %.1f / %.1f MB (UMFPACK / BEM / BED)\n" r Ns tfu tfb tfd 1e3tsu 1e3tsb 1e3tsd mu mb md
    flush(stdout); flush(out)
end
close(out)
