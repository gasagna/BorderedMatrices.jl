# Cyclic block-banded matrices, assembled directly as bordered matrices with a banded A.

"""
    blockcyclic(f, r, Ns, h, b, c, d)

The bordered matrix of the `(r Ns + q) × (r Ns + q)` matrix

    ⎡ K   b ⎤
    ⎣ cᵀ  d ⎦

where `K` has `Ns` slices of `r` unknowns, slice `l` coupled to slices `l - h`,
..., `l + h` counted modulo `Ns`, with the `r × r` block `f(l, m)` between slices
`l` and `m`; `b` and `c` are `r Ns × q` and `d` is `q × q`.

The last `h` slices and the `q` extra unknowns form the border, so that the
wrap-around blocks between the first and the last slices fall into `B` and `C`,
and `A` (the first `Ns - h` slices) is a `BandedMatrix` with bandwidths
`(h + 1) r - 1`. `f` is called once for each of the `(2h + 1) Ns` blocks.
"""
function blockcyclic(f, r::Int, Ns::Int, h::Int, b::Matrix{T}, c::Matrix{T}, d::Matrix{T}) where {T}
    Ns > 2h || throw(ArgumentError("need Ns > 2h slices"))
    q, m = size(d, 1), Ns - h               # m interior slices, in A
    n, p = r * m, h * r + q
    A = BandedMatrix{T}(n, (h + 1) * r - 1, (h + 1) * r - 1)
    B, C, D = zeros(T, n, p), zeros(T, n, p), zeros(T, p, p)
    idx(l) = (l <= m ? l - 1 : l - m - 1) * r .+ (1:r)   # rows of slice l in A, or in the border
    for l in 1:Ns, j in -h:h
        k = mod(l - 1 + j, Ns) + 1
        F = f(l, k)
        if l <= m && k <= m
            view(A, idx(l), idx(k)) .= F
        elseif l <= m
            B[idx(l), idx(k)] .= F
        elseif k <= m
            C[idx(k), idx(l)] .= transpose(F)  # the bottom block is Cᵀ
        else
            D[idx(l), idx(k)] .= F
        end
    end
    e = h * r .+ (1:q)                         # the extra unknowns, in the border
    B[:, e] .= b[1:n, :];  D[1:h*r, e] .= b[n+1:end, :]
    C[:, e] .= c[1:n, :];  D[e, 1:h*r] .= transpose(c[n+1:end, :])
    D[e, e] .= d
    return BorderedMatrix(A, B, C, D)
end
