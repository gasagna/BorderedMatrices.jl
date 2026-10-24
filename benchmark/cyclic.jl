# Benchmark: cyclic block-banded matrices with one dense border row and column, solved
#   (a) with UMFPACK's sparse LU of the whole matrix, and
#   (b) as a banded matrix bordered by its last h slices and the extra unknown (bandedlu + BEM),
#   (c) the same with plain block elimination (bandedlu + BED).
# Ns slices of r unknowns; each slice couples to the h slices on each side (periodically).
#
#     julia --project benchmark/cyclic.jl [h]
#
# Prints a markdown table: factorisation time, time of one solve, memory of the factors,
# relative residual.

using BorderedMatrices, LinearAlgebra, SparseArrays, Random, Printf

BLAS.set_num_threads(1)

function cyclic_matrix(r, Ns, h; rng = Random.default_rng())
    n = r * Ns
    Is, Js, Vs = Int[], Int[], Float64[]
    for l in 1:Ns, j in -h:h
        m = mod(l - 1 + j, Ns) + 1
        blk = randn(rng, r, r) ./ sqrt(r) .+ (j == 0 ? (2h + 2) * Matrix(1.0I, r, r) : 0.0)
        for q in 1:r, k in 1:r
            push!(Is, (l - 1) * r + k); push!(Js, (m - 1) * r + q); push!(Vs, blk[k, q])
        end
    end
    for k in 1:n   # one dense border column and row (e.g. a period and a phase condition)
        push!(Is, k); push!(Js, n + 1); push!(Vs, randn(rng))
        push!(Is, n + 1); push!(Js, k); push!(Vs, randn(rng))
    end
    return sparse(Is, Js, Vs, n + 1, n + 1)
end

# best of a few runs, so that compilation and noise do not count
best(f, k) = minimum(@elapsed(f()) for _ in 1:k)

h = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 4
println("h = $h, BLAS threads = 1, $(Sys.cpu_info()[1].model)")
println()
println("| r | N_s | unknowns | factor (s): UMFPACK / BEM / BED | solve (ms): UMFPACK / BEM / BED | memory (MB): UMFPACK / BEM / BED | residual: UMFPACK / BEM / BED |")
println("|---:|---:|---:|:---:|:---:|:---:|:---:|")
rng = MersenneTwister(0)
for r in (5, 10, 50, 100), Ns in (20, 50, 100)
    K = cyclic_matrix(r, Ns, h; rng)
    N = size(K, 1)
    b = randn(rng, N)
    k = N < 20_000 ? 3 : 2
    # (a) UMFPACK
    Fu = lu(K)
    tfu = best(() -> lu(K), k)
    tsu = best(() -> Fu \ b, 5)
    xu  = Fu \ b
    mu  = 16 * (nnz(Fu.L) + nnz(Fu.U)) / 1e6           # values + row indices
    # (b) banded interior + border of the last h slices and the extra unknown
    M   = BorderedMatrix(K, h * r + 1)
    Fb  = lu(M; factorize = bandedlu)
    tfb = best(() -> lu(BorderedMatrix(K, h * r + 1); factorize = bandedlu), k)
    tsb = best(() -> Fb \ b, 5)
    xb  = Fb \ b
    mb  = (sizeof(Fb.A.AB) + sizeof(Fb.W) + sizeof(Fb.V) + Base.summarysize(Fb.B) + Base.summarysize(Fb.C)) / 1e6
    # (c) the same with plain block elimination (no V = A⁻¹B)
    Fd  = lu(M; alg = :BED, factorize = bandedlu)
    tfd = best(() -> lu(BorderedMatrix(K, h * r + 1); alg = :BED, factorize = bandedlu), k)
    tsd = best(() -> Fd \ b, 5)
    xd  = Fd \ b
    md  = (sizeof(Fd.A.AB) + sizeof(Fd.W) + Base.summarysize(Fd.B) + Base.summarysize(Fd.C)) / 1e6
    res(x) = norm(K * x - b) / norm(b)
    @printf "| %d | %d | %d | %.3f / %.3f / %.3f | %.2f / %.2f / %.2f | %.1f / %.1f / %.1f | %.0e / %.0e / %.0e |\n" r Ns N tfu tfb tfd 1e3tsu 1e3tsb 1e3tsd mu mb md res(xu) res(xb) res(xd)
    flush(stdout)
end
