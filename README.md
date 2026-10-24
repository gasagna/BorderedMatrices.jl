<p align="center">
  <img src="assets/logo.svg" alt="BorderedMatrices.jl" width="720">
</p>

# BorderedMatrices.jl

Direct solution of bordered linear systems

```math
\begin{bmatrix} A & B \\ C^\top & D \end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix}
=
\begin{bmatrix} f \\ g \end{bmatrix},
\qquad
A \in \mathbb{F}^{n\times n},\quad
B,\ C \in \mathbb{F}^{n\times p},\quad
D \in \mathbb{F}^{p\times p},\quad
\mathbb{F} = \mathbb{R} \text{ or } \mathbb{C},
```

by block elimination, reusing a factorisation of the large block $A$. The
border is usually thin, $p \ll n$.

The package is meant for direct linear algebra on stored arrays: every block is
a matrix and every part of a vector is a vector, all with the same element
type. $A$ can be any matrix whose factorisation supports `ldiv!` with the
matrix and with its transpose (a dense `Matrix`, a `SparseMatrixCSC`, or a
banded matrix factorised with the package's own `bandedlu`);
$B$ and $C$ are matrices of the same type, dense or sparse; $D$ is a dense
`Matrix`.

Systems of this form appear wherever a large structured operator is augmented
by a few dense rows and columns: Newton methods for periodic orbits (the period
as an extra unknown, a phase condition as an extra equation), pseudo-arclength
continuation, Lagrange multipliers. Keeping the border separate lets $A$ be
factorised by a solver that exploits its structure, which the dense border
would otherwise destroy.

## Algorithms

Eliminating $x$ gives the $p \times p$ Schur complement:

```math
S = D - C^\top A^{-1} B ,
\qquad
S\, y = g - C^\top A^{-1} f ,
\qquad
A\, x = f - B\, y .
```

There are two ways of organising this computation.

**`:BED`, plain block elimination.** With $W = A^{-\top} C$,

```math
y = \left(D - W^\top B\right)^{-1} \left(g - W^\top f\right),
\qquad
x = A^{-1}\left(f - B\,y\right).
```

It is cheap, but inaccurate when $A$ is nearly singular, even if the bordered
matrix itself is well conditioned: $W$ is then huge and $g - W^\top f$ suffers
from cancellation.

**`:BEM`, mixed block elimination** (default; Govaerts and Pryce). A first pass
as `:BED` gives $y_1$; the residual of the bordered system is then solved again
with the second form of the Schur complement, built from $V = A^{-1} B$:

```math
\begin{aligned}
y_1 &= \left(D - W^\top B\right)^{-1}\left(g - W^\top f\right), \\
f_1 &= f - B\,y_1, \qquad g_1 = g - D\,y_1, \\
\xi &= A^{-1} f_1, \\
y_2 &= \left(D - C^\top V\right)^{-1}\left(g_1 - C^\top \xi\right), \\
x &= \xi - V\,y_2, \qquad y = y_1 + y_2 .
\end{aligned}
```

In exact arithmetic $y_2 = 0$; in floating point the second pass removes the
error of the first. BEM is stable when $A$ is nearly singular, which is the
usual situation when the border is precisely what makes the system nonsingular:
at a periodic orbit, for example, the time-translation direction is in the
kernel of $A$ and the phase condition removes it.

`lu` computes $W$, $V$ (BEM only) and the LU factors of the $p \times p$ Schur
complements **once**. Each solve then costs a single solve with $A$, plus
$O(np)$ operations.

> W. Govaerts and J. D. Pryce, *Mixed block elimination for linear systems with
> wider borders*, IMA J. Numer. Anal. 13 (1993) 161–180.
> W. Govaerts, *Numerical Methods for Bifurcations of Dynamical Equilibria*,
> SIAM (2000), ch. 3.

## Usage

```julia
using BorderedMatrices, LinearAlgebra, SparseArrays

n, p = 1000, 2
A = sprand(n, n, 0.01) + 10I
M = BorderedMatrix(A, rand(n, p), rand(n, p), zeros(p, p))   # [A B; Cᵀ D]

F = lu(M)                                  # lu(A) (here UMFPACK), then BEM
r = BorderedVector(rand(n), rand(p))       # [f; g]
z = F \ r                                  # z.x = x, z.y = y
ldiv!(F, r)                                # or in place
```

* `lu(M; alg = :BEM, factorize = lu)`: `factorize` is applied to $A$; use
  `factorize = lu!` to factorise a dense $A$ in place.
* `alg = :BED` selects the plain block elimination.
* `ldiv!(F, r)` does not allocate (apart from what the factorisation of $A$
  allocates in its own `ldiv!`, as UMFPACK does). BEM keeps a work vector of
  length $p$ in the factorisation, so a factorisation should not be used by
  several threads at once.

### Sparse borders

When the border columns are mostly zero, pass $B$ and $C$ as sparse matrices:
they are then stored and multiplied as such. The package itself does not depend
on SparseArrays; any matrix type that supports `mul!` and `Matrix(C)` works.
$W = A^{-\top}C$ and $V = A^{-1}B$ are dense in any case.

## Real and complex matrices

All blocks of a `BorderedMatrix`, and both parts of a `BorderedVector`, must
have the same element type, real or complex; there is no promotion.

The bottom block is the **transpose** $C^\top$, also for complex matrices, not
the conjugate transpose $C^{\mathsf H}$: to border with $C^{\mathsf H}$, pass
`conj(C)`. The algorithms use solves with $A^\top$ consistently.

## Banded $A$

`bandedlu(A, kl, ku)` computes the LU factorisation with partial pivoting
(LAPACK `gbtrf`) of a matrix `A` of any type with $k_l$ sub- and $k_u$
super-diagonals, copying its band into LAPACK band storage; the factors take
$(2k_l + k_u + 1)\,n$ numbers and there is no fill outside the band. Pass it to
`lu` as `factorize`:

```julia
F = lu(BorderedMatrix(A, B, C, D); factorize = A -> bandedlu(A, kl, ku))
```

`bandedlu!(AB, kl, ku)` factorises a matrix assembled directly in band storage,
$A_{ij}$ in `AB[kl + ku + 1 + i - j, j]` with `AB` of size
$(2k_l + k_u + 1)\times n$, without any copy. The factorisation (a `BandedLU`)
supports `ldiv!` and `\` with the matrix and with its transpose.

### Blocked solves with many right-hand sides

Factorising the bordered matrix needs $p$ solves with $A^\top$ (for
$W = A^{-\top}C$) and, for BEM, $p$ solves with $A$ (for $V = A^{-1}B$), all at
once. LAPACK's banded solve, `gbtrs`, handles one right-hand side at a time with matrix–vector operations. For wide
bands and many right-hand sides this dominates the cost of `lu`, even though
the banded LU itself runs at matrix–matrix speed.

For a matrix of right-hand sides, `ldiv!` with a `BandedLU` therefore uses
blocked solves instead: the same factors (and so the same partial pivoting),
applied by blocks of $n_b = 64$ columns of $L$ and rows of $U$:

* **$X \leftarrow L^{-1}PX$.** For the columns $J = j_0, \dots, j_1$ of a block,
  copy the multipliers of $L$ into a dense $(n_b + k_l)\times n_b$ unit lower
  trapezoidal matrix $\begin{bmatrix} L_{11} \\ L_{21} \end{bmatrix}$, applying to
  each column the row interchanges of the later columns of the block (`gbtrf`
  stores the multipliers unpermuted). Then apply the block's interchanges to
  the rows of $X$ and
  ```math
  X_J \leftarrow L_{11}^{-1} X_J ,
  \qquad
  X_{j_1+1:\,j_1+k_l} \leftarrow X_{j_1+1:\,j_1+k_l} - L_{21} X_J ,
  ```
  a triangular solve and a matrix product (BLAS `trsm` and `gemm`).
* **$X \leftarrow U^{-1}X$.** From the bottom, for the rows $I$ of a block,
  $X_I \leftarrow U_{II}^{-1} X_I$, then subtract $U_{\cdot I} X_I$ from the
  $k_l + k_u$ rows above.
* **$X \leftarrow A^{-\top}X$** is the transpose of the two steps, in reverse
  order: $U^{-\top}$ from the top, then $L_{11}^{-\top}$ and $L_{21}^\top$ from the
  last block to the first, with the row interchanges undone at the end of each
  block.

The result equals that of `gbtrs` up to rounding. The blocked solves are used
only when $k_l \ge 48$: for narrower bands, and for a single right-hand side,
`ldiv!` calls `gbtrs`, which is as fast or faster there. The speed-up,
for the $p = hr + 1$ right-hand sides of the cyclic block-banded matrices
below, grows with the bandwidth and reaches 10–15×:

![Speed-up of the blocked banded solves over gbtrs](benchmark/speedup.svg)

Reproduce with `julia --project=benchmark benchmark/solves.jl`.

## Cyclic block-banded matrices

### The matrix

Consider unknowns organised in $N_s$ **slices** of $r$ unknowns each, plus $q$
extra unknowns at the end, so that $N = r N_s + q$. Typically, slice $l$ holds
the $r$ coefficients of a field at time $s_l$ of a periodic time grid, and the
extra unknowns are, e.g., the period. The ordering is slice by slice: unknowns
$(l-1)r + 1, \dots, l\,r$ belong to slice $l$.

A time-periodic operator discretised with a stencil that reaches $h$ slices on
each side (e.g. centred differences: $h = 1$ for second order, $h = 4$ for
eighth order) couples slice $l$ only to slices $l-h, \dots, l+h$, **counted
modulo $N_s$**. With $K_{lm}$ the $r \times r$ block that couples slice $l$ to
slice $m$, and the extra unknowns contributing dense columns $b_l$ and rows
$c_l^\top$, the matrix for $N_s = 5$, $h = 1$, $q = 1$ is

```math
K =
\left[\begin{array}{ccccc|c}
K_{11} & K_{12} &        &        & \color{red}{K_{15}} & b_1 \\
K_{21} & K_{22} & K_{23} &        &        & b_2 \\
       & K_{32} & K_{33} & K_{34} &        & b_3 \\
       &        & K_{43} & K_{44} & K_{45} & b_4 \\
\color{red}{K_{51}} &        &        & K_{54} & K_{55} & b_5 \\ \hline
c_1^\top & c_2^\top & c_3^\top & c_4^\top & c_5^\top & d
\end{array}\right].
```

The blocks in red are the **wrap-around corners**: slice 1 is a neighbour of
slice $N_s$. Because of them $K$ is not banded, and a general sparse LU fills
the block columns that connect the two ends.

### The ordering trick

Take as border the last $h$ slices together with the $q$ extra unknowns,
$p = h\,r + q$. For the example above ($h = 1$), slice 5 and the extra unknown
become the border:

```math
A =
\begin{bmatrix}
K_{11} & K_{12} &        &        \\
K_{21} & K_{22} & K_{23} &        \\
       & K_{32} & K_{33} & K_{34} \\
       &        & K_{43} & K_{44}
\end{bmatrix},
\qquad
B =
\begin{bmatrix}
\color{red}{K_{15}} & b_1 \\ 0 & b_2 \\ 0 & b_3 \\ K_{45} & b_4
\end{bmatrix},
\qquad
C^\top =
\begin{bmatrix}
\color{red}{K_{51}} & 0 & 0 & K_{54} \\
c_1^\top & c_2^\top & c_3^\top & c_4^\top
\end{bmatrix},
\qquad
D =
\begin{bmatrix}
K_{55} & b_5 \\ c_5^\top & d
\end{bmatrix}.
```

The interior slices $1, \dots, N_s - h$ only couple to interior slices at most
$h$ slices away, or to the border: every wrap-around coupling has moved into
$B$ and $C$. So $A$ is **banded**, with

```math
k_l,\ k_u \;\le\; (h + 1)\,r - 1 ,
```

and BEM takes care of the border. In code, with $n = N - p$:

```julia
using SparseArrays
# K: the N × N sparse matrix, ordered slice by slice, extra unknowns last
p = h*r + q
n = size(K, 1) - p
kb = (h + 1)*r - 1                                      # bandwidths of A
M = BorderedMatrix(K[1:n, 1:n],                         # A, banded
                   K[1:n, n+1:end],                     # B, sparse
                   sparse(transpose(K[n+1:end, 1:n])),  # C, sparse
                   Matrix(K[n+1:end, n+1:end]))         # D
F = lu(M; factorize = A -> bandedlu(A, kb, kb))
z = F \ BorderedVector(f, g)
```

Better still, assemble $A$ directly in band storage and factorise it with
`bandedlu!`, so that nothing needs converting.

### Memory

| part | numbers stored |
|---|---|
| banded LU of $A$ | $(2k_l + k_u + 1)\,n \approx 3(h+1)\,r\,n$ |
| $W = A^{-\top}C$ and $V = A^{-1}B$ (BEM; only $W$ for BED) | $2\,p\,n$ |
| $B$, $C$ (sparse) and $D$ | the nonzeros of $B$ and $C$: $O(h^2 r^2)$ in the blocks next to the border, plus $2qn$ in the $q$ dense rows and columns; $p^2$ for $D$ |

Everything grows linearly with $N_s$, and there is no fill outside the band.
The figure shows the structure for a small example: $K$, the nonzeros of
UMFPACK's factors $L + U$, and what the bordered factorisation stores.

![Sparsity of K, of UMFPACK's factors, and storage of the bordered factorisation](benchmark/sparsity.svg)

## Benchmark

Cyclic block-banded matrices as above, with $N_s = 20, \dots, 1000$ slices of
$r = 5, 10, 20, 50, 100, 200$ unknowns (up to $r^2 N_s = 6\cdot 10^6$, i.e. a few
GB), a stencil reaching $h = 1$ or $h = 4$ slices on each side, and one dense
border row and column ($q = 1$). Random blocks, made diagonally dominant. Three
solvers:

* **UMFPACK**: sparse LU of the whole $(rN_s + 1)\times(rN_s + 1)$ matrix, `lu(K)`;
* **BEM**: `lu(M; factorize = A -> bandedlu(A, kb, kb))`, with sparse $A$, $B$
  and $C$ and the last $h$ slices and the extra unknown as the border;
* **BED**: the same with `alg = :BED`, which does not compute $V = A^{-1}B$.

Memory is that of the factorisation: for UMFPACK
$16\,(\mathrm{nnz}(L) + \mathrm{nnz}(U))$ bytes (values and indices), for
BEM/BED the banded LU of $A$, $W$ (and $V$). Times are the best of up to three
runs on one core (`BLAS.set_num_threads(1)`) of an Intel Xeon at 2.8 GHz, with
Julia 1.11; the factorisation time includes building the blocks from $K$,
copying the band of $A$ into band storage, and computing
$W$ (and $V$) and the Schur complements. All three solvers give the same
residual, $\lVert Kx - b\rVert/\lVert b\rVert \approx 10^{-15}$.

![Memory of the factorisation against the number of unknowns, and ratio to UMFPACK](benchmark/memory.svg)

![Factorisation and solve times relative to UMFPACK](benchmark/time.svg)

What the figures show:

* **Memory** grows linearly with the number of unknowns for all three solvers.
  The ratio to UMFPACK is nearly constant: with $h = 4$ about 0.68 for BEM and
  0.56 for BED, for every $r$ and $N_s$; with $h = 1$ about 0.75–0.8 and
  0.63–0.7, rising towards 0.95 for the largest $N_s$, where UMFPACK's memory
  grows a little more slowly than linearly.
* **Factorisation** is faster than UMFPACK's in almost every case: with $h = 1$
  by a factor of about 1.4–5, with $h = 4$ by up to about 2.5 (BEM) and 3 (BED).
  The exceptions are the widest bands with few slices ($r = 100$, $N_s = 20$
  and $r = 200$, $N_s \le 50$, with $h = 4$), where BEM takes up to 1.6 times as
  long. This relies on the blocked solves for $W$ and $V$ described above; with
  LAPACK's `gbtrs` the factorisation was up to 6 times slower than UMFPACK's for
  large blocks.
* **One solve** takes about as long as with UMFPACK: with $h = 4$ 5–60% less;
  with $h = 1$ mostly less, up to 20% more for the largest cases (short,
  noisy timings).

In the small example of the sparsity figure the bordered factorisation stores
more numbers than UMFPACK, because there the dense $W$ and $V$ ($p/n \approx 0.1$)
weigh more than the band; for the larger cases of the benchmark it stores less.

Reproduce with

```
julia --project=benchmark benchmark/cyclic.jl 1      # and 4: writes benchmark/results_h1.csv, results_h4.csv
julia --project=benchmark benchmark/sparsity.jl
python3 benchmark/plot.py
```

## Tests

```
julia --project -e 'using Pkg; Pkg.test()'
```

The tests use SparseArrays and Random (test dependencies only). The package depends only on LinearAlgebra.
