# Benchmark: cyclic block-banded matrices with one dense border row and column, solved
#   (a) with UMFPACK's sparse LU of the whole matrix, and
#   (b) as a banded matrix bordered by its last h slices and the extra unknown, with the banded
#       LU of BandedMatrices.jl for the interior and BEM for the border,
#   (c) the same with plain block elimination (BED).
# Ns slices of r unknowns; each slice couples to the h slices on each side (periodically).
#
#     julia --project=benchmark benchmark/cyclic.jl [h]
#
# Writes benchmark/results_h<h>.csv: factorisation time, time of one solve (s), memory of the
# factorisation (MB), relative residual. benchmark/plot.py draws them.

using BorderedMatrices, BandedMatrices, LinearAlgebra, SparseArrays, Random, Printf

bandedlu(A) = lu(BandedMatrix(A))     # the conversion from sparse is included in the timings

# the bordered matrix of K: interior slices in A (banded), the last p unknowns as the border
function bordered(K, p)
    N = size(K, 1); n = N - p
    # B and C stay sparse: only the first and last h slices and the extra unknown are nonzero
    return BorderedMatrix(K[1:n, 1:n], K[1:n, n+1:N], sparse(transpose(K[n+1:N, 1:n])), Matrix(K[n+1:N, n+1:N]))
end

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "matrices.jl"))

# best of a few runs, so that compilation and noise do not count
best(f, k) = minimum(@elapsed(f()) for _ in 1:k)

h = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 4
println("h = $h, BLAS threads = 1, $(Sys.cpu_info()[1].model)")
out = open(joinpath(@__DIR__, "results_h$h.csv"), "w")
println(out, "r,Ns,N,factor_umfpack,factor_bem,factor_bed,solve_umfpack,solve_bem,solve_bed,mem_umfpack,mem_bem,mem_bed,res_umfpack,res_bem,res_bed")
rng = MersenneTwister(0)
for r in (5, 10, 50, 100), Ns in (20, 50, 100, 200)
    K = cyclic_matrix(r, Ns, h; rng)
    N = size(K, 1)
    b = randn(rng, N)
    k = N < 20_000 ? 3 : 2
    # (a) UMFPACK
    Fu = lu(K)
    tfu = best(() -> lu(K), k)
    tsu = best(() -> Fu \ b, 20)
    xu  = Fu \ b
    mu  = 16 * (nnz(Fu.L) + nnz(Fu.U)) / 1e6           # values + row indices
    # (b) banded interior + border of the last h slices and the extra unknown
    M   = bordered(K, h * r + 1)
    rb  = BorderedVector(b[1:N-h*r-1], b[N-h*r:N])
    Fb  = lu(M; factorize = bandedlu)
    tfb = best(() -> lu(bordered(K, h * r + 1); factorize = bandedlu), k)
    tsb = best(() -> Fb \ rb, 20)
    xb  = collect(Fb \ rb)
    mb  = (Base.summarysize(Fb.A) + sizeof(Fb.W) + sizeof(Fb.V)) / 1e6       # B, C, D belong to M
    # (c) the same with plain block elimination (no V = A⁻¹B)
    Fd  = lu(M; alg = :BED, factorize = bandedlu)
    tfd = best(() -> lu(bordered(K, h * r + 1); alg = :BED, factorize = bandedlu), k)
    tsd = best(() -> Fd \ rb, 20)
    xd  = collect(Fd \ rb)
    md  = (Base.summarysize(Fd.A) + sizeof(Fd.W)) / 1e6
    res(x) = norm(K * x - b) / norm(b)
    println(out, join((r, Ns, N, tfu, tfb, tfd, tsu, tsb, tsd, mu, mb, md, res(xu), res(xb), res(xd)), ","))
    @printf "r = %3d, Ns = %3d: factor %.3f / %.3f / %.3f s, solve %.2f / %.2f / %.2f ms, memory %.1f / %.1f / %.1f MB (UMFPACK / BEM / BED)\n" r Ns tfu tfb tfd 1e3tsu 1e3tsb 1e3tsd mu mb md
    flush(stdout); flush(out)
end
close(out)
