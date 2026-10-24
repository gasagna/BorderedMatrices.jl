# Benchmark: cyclic block-banded matrices with one dense border row and column, solved
#   (a) with UMFPACK's sparse LU of the whole matrix, and
#   (b) as a banded matrix bordered by its last h slices and the extra unknown, with the banded
#       LU of BandedMatrices.jl for the interior and BEM for the border,
#   (c) the same with plain block elimination (BED).
# Ns slices of r unknowns; each slice couples to the h slices on each side (periodically).
#
#     julia --project=benchmark benchmark/cyclic.jl [h]
#
# Prints a markdown table: factorisation time, time of one solve, memory of the factors,
# relative residual.

using BorderedMatrices, BandedMatrices, LinearAlgebra, SparseArrays, Random, Printf

bandedlu(A) = lu(BandedMatrix(A))     # the conversion from sparse is included in the timings

# the bordered matrix of K: interior slices in A (banded), the last p unknowns as the border
function bordered(K, p)
    N = size(K, 1); n = N - p
    return BorderedMatrix(K[1:n, 1:n], Matrix(K[1:n, n+1:N]), Matrix(transpose(K[n+1:N, 1:n])), Matrix(K[n+1:N, n+1:N]))
end

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
    M   = bordered(K, h * r + 1)
    rb  = BorderedVector(b[1:N-h*r-1], b[N-h*r:N])
    Fb  = lu(M; factorize = bandedlu)
    tfb = best(() -> lu(bordered(K, h * r + 1); factorize = bandedlu), k)
    tsb = best(() -> Fb \ rb, 5)
    xb  = collect(Fb \ rb)
    mb  = (Base.summarysize(Fb.A) + sizeof(Fb.W) + sizeof(Fb.V)) / 1e6       # B, C, D belong to M
    # (c) the same with plain block elimination (no V = A⁻¹B)
    Fd  = lu(M; alg = :BED, factorize = bandedlu)
    tfd = best(() -> lu(bordered(K, h * r + 1); alg = :BED, factorize = bandedlu), k)
    tsd = best(() -> Fd \ rb, 5)
    xd  = collect(Fd \ rb)
    md  = (Base.summarysize(Fd.A) + sizeof(Fd.W)) / 1e6
    res(x) = norm(K * x - b) / norm(b)
    @printf "| %d | %d | %d | %.3f / %.3f / %.3f | %.2f / %.2f / %.2f | %.1f / %.1f / %.1f | %.0e / %.0e / %.0e |\n" r Ns N tfu tfb tfd 1e3tsu 1e3tsb 1e3tsd mu mb md res(xu) res(xb) res(xd)
    flush(stdout)
end
