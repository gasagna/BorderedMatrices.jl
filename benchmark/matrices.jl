# A random cyclic block-banded problem: Ns slices of r unknowns, each slice coupled to the h
# slices on each side (periodically), with random blocks made diagonally dominant, plus one
# dense border row and column. Returns the blocks as a function f(l, k), the border b, c, d,
# and the same matrix assembled as a SparseMatrixCSC K, in the same ordering.

function cyclic_problem(r, Ns, h; rng = Random.default_rng())
    blocks = Dict((l, mod(l - 1 + j, Ns) + 1) => randn(rng, r, r) ./ sqrt(r) + (j == 0 ? (2h + 2) * I : 0I)
                  for l in 1:Ns, j in -h:h)
    n = r * Ns
    b, c, d = randn(rng, n, 1), randn(rng, n, 1), zeros(1, 1)
    Is, Js, Vs = Int[], Int[], Float64[]
    for ((l, k), F) in blocks, q in 1:r, i in 1:r
        push!(Is, (l - 1) * r + i); push!(Js, (k - 1) * r + q); push!(Vs, F[i, q])
    end
    for i in 1:n
        push!(Is, i); push!(Js, n + 1); push!(Vs, b[i])
        push!(Is, n + 1); push!(Js, i); push!(Vs, c[i])
    end
    K = sparse(Is, Js, Vs, n + 1, n + 1)
    return (l, k) -> blocks[(l, k)], b, c, d, K
end
