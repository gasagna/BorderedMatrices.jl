# A random cyclic block-banded matrix with one dense border row and column.
# Ns slices of r unknowns; each slice couples to the h slices on each side (periodically).

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
