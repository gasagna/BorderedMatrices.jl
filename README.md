# BorderedMatrices.jl

Solve bordered linear systems

```
⎡ A   B ⎤ ⎡ x ⎤   ⎡ f ⎤
⎣ Cᵀ  D ⎦ ⎣ y ⎦ = ⎣ g ⎦
```

reusing a factorisation of the `n × n` block `A`. The borders `B` and `C` are
`n × p` (or vectors, `p = 1`) and `D` is `p × p` (or a scalar).

Systems of this form appear wherever a large structured operator is augmented
by a few dense rows and columns: Newton methods for periodic orbits (the
period as an extra unknown, a phase condition as an extra equation),
pseudo-arclength continuation, Lagrange multipliers, deflation. Keeping the
border separate lets `A` be factorised with a solver that exploits its
structure (sparse, banded, ...), which a dense border would otherwise destroy.

## Algorithms

Let `S = D - Cᵀ A⁻¹ B` be the `p × p` Schur complement.

* `:BED` — plain block elimination: `W = A⁻ᵀ C`, `y = (D - WᵀB)⁻¹ (g - Wᵀ f)`,
  `x = A⁻¹ (f - B y)`. Cheap, but inaccurate when `A` is nearly singular, even if
  the bordered matrix is well conditioned.
* `:BEM` (default) — the mixed block elimination of Govaerts and Pryce: a first
  pass as `:BED`, then a correction step that uses `V = A⁻¹ B` and the second
  form of the Schur complement, `D - Cᵀ V`. It is stable when `A` is nearly
  singular, which is the usual case when the border is what makes the system
  nonsingular (e.g. the phase condition removing the time-translation
  direction at a periodic orbit).

`W`, `V` and the LU factors of the two `p × p` Schur complements are computed
once, in `lu`. Each solve then costs **one** solve with `A`, plus `O(n p)` work.

> W. Govaerts and J. D. Pryce, *Mixed block elimination for linear systems with
> wider borders*, IMA J. Numer. Anal. 13 (1993) 161–180;
> W. Govaerts, *Numerical Methods for Bifurcations of Dynamical Equilibria*,
> SIAM (2000), ch. 3.

## Usage

```julia
using BorderedMatrices, LinearAlgebra, SparseArrays

A = sprand(1000, 1000, 0.01) + 10I
b, c, d = rand(1000), rand(1000), 0.0

M = BorderedMatrix(A, b, c, d)        # [A b; cᵀ d]
F = lu(M)                             # factorises A (here sparse LU), :BEM
r = BorderedVector(rand(1000), 1.0)   # [f; g]
z = F \ r                             # z._1 = x, z._2 = y

# wider borders: B, C are n × p, D is p × p
M = BorderedMatrix(A, rand(1000, 3), rand(1000, 3), zeros(3, 3))
z = lu(M) \ rand(1003)

# partition an assembled matrix: the last p rows and columns become the border
M = BorderedMatrix(K, p)
```

`lu(M; alg = :BEM, factorize = lu)`: `factorize` is applied to `A` and may be
any factorisation that supports `F \ x` and `transpose(F) \ x`. `lu!(M)`
factorises a dense `A` in place. The original interface
`ldiv!(M, r, alg = :BEM, inplace = true)` is kept.

## Banded and cyclic block-banded matrices

`bandedlu(A)` computes a banded LU factorisation with partial pivoting
(LAPACK `gbtrf`), with storage `(2kl + ku + 1) × n` and no fill outside the
band; `bandwidths(A)` returns `(kl, ku)`. It supports `\`, `ldiv!`, and solves
with `transpose(F)` and `adjoint(F)`.

A *cyclic* block-banded matrix, e.g. a periodic finite-difference operator in
time with dense spatial blocks, is not banded because of the wrap-around
corners. Ordering the unknowns of the last `h` time slices (`h` = half-bandwidth
in slices) last, together with any extra unknowns, the interior block is
banded and the corners move into the border:

```julia
# K: cyclic block-banded (Ns slices of r unknowns, couplings up to ±h slices),
#    plus one extra row and column (period, phase condition)
M = BorderedMatrix(K, h*r + 1)
F = lu(M; factorize = bandedlu)
x = F \ rhs
```

The memory is that of the band, `≈ 3 (h + 1) r × N` numbers for the factors,
plus `2 (h r + 1) × N` for `V` and `W`, against the fill a general sparse LU
produces in the wrap-around block columns.

## Tests

```
julia --project test/runtests.jl
```
