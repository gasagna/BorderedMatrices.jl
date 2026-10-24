<p align="center">
  <img src="assets/logo.svg" alt="BorderedMatrices.jl" width="720">
</p>

# BorderedMatrices.jl

Solve bordered linear systems

```math
\begin{bmatrix} A & B \\ C^\top & D \end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix}
=
\begin{bmatrix} f \\ g \end{bmatrix},
\qquad
A \in \mathbb{R}^{n\times n},\quad
B,\ C \in \mathbb{R}^{n\times p},\quad
D \in \mathbb{R}^{p\times p},
```

reusing a factorisation of the large block $A$. The border is usually thin,
$p \ll n$; for $p = 1$, $B = b$ and $C = c$ are vectors and $D = d$ is a scalar.

Systems of this form appear wherever a large structured operator is augmented
by a few dense rows and columns: Newton methods for periodic orbits (the period
as an extra unknown, a phase condition as an extra equation), pseudo-arclength
continuation, Lagrange multipliers, deflation. Keeping the border separate lets
$A$ be factorised by a solver that exploits its structure (sparse, banded, ...),
which the dense border would otherwise destroy.

## Algorithms

Eliminating $x$ gives the $p \times p$ Schur complement

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
cancellation.

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
for example, at a periodic orbit the phase condition removes the
time-translation direction from the kernel of $A$.

In `lu`, the matrices $W = A^{-\top}C$, $V = A^{-1}B$ (BEM only) and the LU
factors of the two $p\times p$ Schur complements are computed **once**. Each
solve then costs a single solve with $A$, plus $O(np)$ operations.

> W. Govaerts and J. D. Pryce, *Mixed block elimination for linear systems with
> wider borders*, IMA J. Numer. Anal. 13 (1993) 161–180.
> W. Govaerts, *Numerical Methods for Bifurcations of Dynamical Equilibria*,
> SIAM (2000), ch. 3.

## Usage

```julia
using BorderedMatrices, LinearAlgebra, SparseArrays

n = 1000
A = sprand(n, n, 0.01) + 10I
b, c, d = rand(n), rand(n), 0.0

M = BorderedMatrix(A, b, c, d)        # [A b; cᵀ d]
F = lu(M)                             # factorises A (here sparse LU), BEM
r = BorderedVector(rand(n), 1.0)      # [f; g]
z = F \ r                             # z._1 = x, z._2 = y

# wider border: B, C are n × p, D is p × p
M = BorderedMatrix(A, rand(n, 3), rand(n, 3), zeros(3, 3))
z = lu(M) \ rand(n + 3)

# partition an assembled (N × N) matrix: its last p rows and columns become the border
M = BorderedMatrix(K, p)
```

* `lu(M; alg = :BEM, factorize = lu)`: `factorize` is applied to $A$ and can be
  any factorisation that supports `F \ x` and `transpose(F) \ x`; e.g. `lu`
  (dense or sparse), or `bandedlu` below.
* `lu!(M)` factorises a dense $A$ in place.
* The original interface `ldiv!(M, r, alg = :BEM, inplace = true)` is kept.

## Real and complex matrices

Everything works for real and complex element types (`Float32`, `Float64`,
`ComplexF32`, `ComplexF64`; `lu` also for any other number type that Julia's
`lu` supports, `bandedlu` only for these four).

* The bottom block is the **transpose** $C^\top$, also for complex matrices,
  not the conjugate transpose $C^{\mathsf H}$. To border with $C^{\mathsf H}$,
  pass `conj(C)`. The algorithms use transposed solves, $W = A^{-\top} C$,
  consistently.
* Mixed element types are promoted: a real $A$ with a complex border or corner
  gives a complex `BorderedMatrix`. A `BorderedVector` keeps the element type of
  its main part (which is not copied) unless the border is complex and the main
  part real.
* A real factorisation applied to a complex right-hand side solves for the real
  and the imaginary parts separately; a complex factorisation applied to a real
  right-hand side returns a complex solution. `ldiv!` cannot store a complex
  solution in a real vector and throws an `ArgumentError`.

## Banded matrices

`bandedlu(A)` computes an LU factorisation with partial pivoting of a matrix
with $k_l$ sub-diagonals and $k_u$ super-diagonals (LAPACK `gbtrf`). It stores
$(2k_l + k_u + 1) \times n$ numbers, with no fill outside the band.
`bandwidths(A)` returns $(k_l, k_u)$. The factorisation supports `\`, `ldiv!`,
and solves with `transpose(F)` and `adjoint(F)`, which BEM needs.

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

Declare as border the last $h$ slices together with the $q$ extra unknowns,
i.e. take $p = h\,r + q$ in `BorderedMatrix(K, p)`. For the example above
($h = 1$), slice 5 and the extra unknown become the border:

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

and can be factorised with `bandedlu`, while BEM takes care of the border.
In code:

```julia
r, Ns, h, q = 24, 583, 4, 1          # 24 unknowns per slice, 583 slices, 8th-order FD, 1 extra unknown
# K :: SparseMatrixCSC of size r*Ns + q, ordered slice by slice, extra unknowns last
M = BorderedMatrix(K, h*r + q)       # border = last h slices + extra unknowns
F = lu(M; factorize = bandedlu)      # banded LU of A, BEM for the border
x = F \ rhs
```

If the unknowns are not ordered slice by slice with the extra ones last,
permute $K$ first.

### Memory

Writing $n = r(N_s - h)$ for the size of $A$ and $p = hr + q$ for the border:

| part | numbers stored |
|---|---|
| banded LU of $A$ | $(2k_l + k_u + 1)\,n \approx 3(h+1)\,r\,n$ |
| $W = A^{-\top}C$ and $V = A^{-1}B$ | $2\,p\,n$ |
| $B$, $C$ (kept sparse if $K$ is sparse) | nonzeros only |

Both grow linearly with $N_s$, and there is no fill outside the band. For
example, with $r = 24$, $h = 4$, $N_s = 583$ ($N \approx 14\,000$) this takes
about 62 MB, against 96 MB for UMFPACK's sparse LU of the whole bordered
matrix.

## Benchmark

Cyclic block-banded matrices as above, with $N_s$ slices of $r$ unknowns, a
stencil reaching $h$ slices on each side, and one dense border row and column
($q = 1$). Random blocks, made diagonally dominant. Three solvers:

* **UMFPACK**: sparse LU of the whole $(rN_s + 1)\times(rN_s + 1)$ matrix, `lu(K)`;
* **BEM**: `lu(BorderedMatrix(K, h*r + 1); factorize = bandedlu)`;
* **BED**: the same with `alg = :BED`, which does not store $V = A^{-1}B$.

Times are the best of a few runs, on one core (`BLAS.set_num_threads(1)`) of an
Intel Xeon at 2.8 GHz, Julia 1.11. "Factor" includes partitioning $K$ and, for
BEM/BED, computing $W$ (and $V$) and the Schur complements; "solve" is one
right-hand side. Memory is that of the factors: for UMFPACK
$16\,(\mathrm{nnz}(L) + \mathrm{nnz}(U))$ bytes (values and indices), for
BEM/BED the band, $W$ (and $V$) and the sparse borders. The residual is
$\lVert Kx - b\rVert / \lVert b\rVert$. Reproduce with
`julia --project benchmark/cyclic.jl h`.

**$h = 1$** (e.g. second-order centred differences, or a one-step scheme between consecutive slices):

| r | N_s | unknowns | factor (s): UMFPACK / BEM / BED | solve (ms): UMFPACK / BEM / BED | memory (MB): UMFPACK / BEM / BED | residual: UMFPACK / BEM / BED |
|---:|---:|---:|:---:|:---:|:---:|:---:|
| 5 | 20 | 101 | 0.000 / 0.000 / 0.000 | 0.00 / 0.01 / 0.01 | 0.0 / 0.0 / 0.0 | 1e-15 / 1e-15 / 1e-15 |
| 5 | 50 | 251 | 0.000 / 0.000 / 0.000 | 0.01 / 0.01 / 0.01 | 0.1 / 0.1 / 0.1 | 3e-16 / 3e-16 / 4e-16 |
| 5 | 100 | 501 | 0.001 / 0.000 / 0.000 | 0.02 / 0.02 / 0.02 | 0.2 / 0.2 / 0.2 | 4e-15 / 4e-15 / 4e-15 |
| 10 | 20 | 201 | 0.000 / 0.000 / 0.000 | 0.01 / 0.01 / 0.01 | 0.2 / 0.1 / 0.1 | 6e-16 / 6e-16 / 6e-16 |
| 10 | 50 | 501 | 0.002 / 0.001 / 0.000 | 0.03 / 0.02 / 0.02 | 0.4 / 0.3 / 0.3 | 6e-16 / 6e-16 / 9e-16 |
| 10 | 100 | 1001 | 0.004 / 0.001 / 0.001 | 0.07 / 0.05 / 0.04 | 0.8 / 0.7 / 0.6 | 4e-16 / 4e-16 / 5e-16 |
| 50 | 20 | 1001 | 0.015 / 0.016 / 0.008 | 0.22 / 0.18 / 0.13 | 3.8 / 3.2 / 2.8 | 9e-16 / 9e-16 / 1e-15 |
| 50 | 50 | 2501 | 0.045 / 0.040 / 0.026 | 0.57 / 0.42 / 0.37 | 9.9 / 8.1 / 7.1 | 1e-14 / 2e-14 / 2e-14 |
| 50 | 100 | 5001 | 0.117 / 0.100 / 0.063 | 1.93 / 1.78 / 1.47 | 20.0 / 16.2 / 14.1 | 9e-16 / 9e-16 / 9e-16 |
| 100 | 20 | 2001 | 0.079 / 0.101 / 0.067 | 1.00 / 0.69 / 0.59 | 15.1 / 12.9 / 11.3 | 2e-15 / 1e-15 / 1e-15 |
| 100 | 50 | 5001 | 0.217 / 0.315 / 0.152 | 3.55 / 3.70 / 2.58 | 39.3 / 32.2 / 28.2 | 2e-15 / 2e-15 / 2e-15 |
| 100 | 100 | 10001 | 0.432 / 0.908 / 0.440 | 6.75 / 8.14 / 7.41 | 79.5 / 64.3 / 56.3 | 1e-15 / 1e-15 / 1e-15 |

**$h = 4$** (eighth-order centred differences):

| r | N_s | unknowns | factor (s): UMFPACK / BEM / BED | solve (ms): UMFPACK / BEM / BED | memory (MB): UMFPACK / BEM / BED | residual: UMFPACK / BEM / BED |
|---:|---:|---:|:---:|:---:|:---:|:---:|
| 5 | 20 | 101 | 0.000 / 0.000 / 0.000 | 0.01 / 0.01 / 0.01 | 0.1 / 0.1 / 0.1 | 7e-16 / 8e-16 / 7e-16 |
| 5 | 50 | 251 | 0.001 / 0.001 / 0.000 | 0.02 / 0.02 / 0.01 | 0.3 / 0.2 / 0.2 | 2e-15 / 2e-15 / 2e-15 |
| 5 | 100 | 501 | 0.002 / 0.001 / 0.001 | 0.04 / 0.03 / 0.02 | 0.7 / 0.5 / 0.4 | 5e-16 / 6e-16 / 5e-16 |
| 10 | 20 | 201 | 0.001 / 0.001 / 0.001 | 0.03 / 0.02 / 0.01 | 0.4 / 0.4 / 0.3 | 2e-15 / 2e-15 / 2e-15 |
| 10 | 50 | 501 | 0.004 / 0.003 / 0.002 | 0.07 / 0.04 / 0.04 | 1.3 / 0.9 / 0.8 | 1e-15 / 1e-15 / 1e-15 |
| 10 | 100 | 1001 | 0.009 / 0.007 / 0.004 | 0.14 / 0.09 / 0.08 | 2.7 / 1.9 / 1.5 | 8e-16 / 7e-16 / 7e-16 |
| 50 | 20 | 1001 | 0.067 / 0.092 / 0.050 | 0.67 / 0.45 / 0.33 | 10.8 / 9.0 / 7.7 | 2e-15 / 1e-15 / 1e-15 |
| 50 | 50 | 2501 | 0.180 / 0.301 / 0.153 | 2.64 / 2.31 / 1.36 | 31.2 / 22.8 / 19.1 | 5e-15 / 5e-15 / 5e-15 |
| 50 | 100 | 5001 | 0.360 / 0.668 / 0.314 | 5.32 / 4.19 / 3.47 | 65.4 / 45.9 / 38.2 | 2e-15 / 2e-15 / 2e-15 |
| 100 | 20 | 2001 | 0.270 / 0.810 / 0.387 | 3.22 / 3.16 / 2.31 | 43.0 / 35.9 / 30.8 | 2e-15 / 2e-15 / 2e-15 |
| 100 | 50 | 5001 | 0.890 / 3.447 / 1.798 | 10.10 / 12.39 / 7.79 | 124.7 / 91.2 / 76.4 | 9e-15 / 9e-15 / 9e-15 |
| 100 | 100 | 10001 | 1.779 / 8.675 / 3.852 | 19.97 / 20.26 / 17.33 | 261.0 / 183.4 / 152.6 | 5e-15 / 5e-15 / 5e-15 |

What the numbers say:

* **Memory**: the banded factorisation always takes less: by 20–30% with BEM,
  by 30–40% with BED.
* **Solves** (what a preconditioner repeats at every Krylov iteration) are
  mostly faster, up to about 2× with BED; for the largest case with $h = 1$
  ($r = N_s = 100$) they are 10–20% slower.
* **Factorisation** is as fast or faster for small blocks, but slower than
  UMFPACK for large blocks, especially with a wide stencil (up to ~5× for
  $r = 100$, $h = 4$; ~2× for $r = 100$, $h = 1$ with BEM). There
  the cost is not the banded LU itself (0.45 s for $r = N_s = 100$, $h = 4$)
  but the $2(hr + 1)$ solves with $A$ needed for $W$ and $V$ (about 3.7 s each
  for 401 right-hand sides, since LAPACK's `gbtrs` works one right-hand side at
  a time). BED halves this cost.
* Accuracy is the same for all three.

## Tests

```
julia --project test/runtests.jl
```
