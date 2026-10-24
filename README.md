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
matrix and with its transpose (a dense `Matrix`, a `SparseMatrixCSC`, a
`BandedMatrix` from [BandedMatrices.jl](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl), ...);
$B$, $C$ and $D$ are dense `Matrix`es.

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

## Real and complex matrices

All blocks of a `BorderedMatrix`, and both parts of a `BorderedVector`, must
have the same element type, real or complex; there is no promotion.

The bottom block is the **transpose** $C^\top$, also for complex matrices, not
the conjugate transpose $C^{\mathsf H}$: to border with $C^{\mathsf H}$, pass
`conj(C)`. The algorithms use solves with $A^\top$ consistently.

## Banded $A$

For a banded $A$ use a `BandedMatrix` from
[BandedMatrices.jl](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl):
`lu` then gives a banded LU (LAPACK `gbtrf`), with storage
$(2k_l + k_u + 1)\,n$ for $k_l$ sub- and $k_u$ super-diagonals and no fill
outside the band. No conversion is needed if $A$ is assembled as a
`BandedMatrix` in the first place.

```julia
using BandedMatrices
A = BandedMatrix(...)                      # n × n, bandwidths (kl, ku)
F = lu(BorderedMatrix(A, B, C, D))
```

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
using BandedMatrices
r, Ns, h, q = 24, 583, 4, 1                # 24 unknowns per slice, 583 slices, 8th-order FD, 1 extra unknown
p = h*r + q; n = r*Ns + q - p
# K: the N × N matrix, ordered slice by slice, extra unknowns last
M = BorderedMatrix(BandedMatrix(K[1:n, 1:n]),          # A, banded
                   Matrix(K[1:n, n+1:end]),            # B
                   Matrix(transpose(K[n+1:end, 1:n])), # C
                   Matrix(K[n+1:end, n+1:end]))        # D
F = lu(M)
z = F \ BorderedVector(f, g)
```

Better still, assemble the blocks directly in this form, with $A$ as a
`BandedMatrix`, so that nothing needs converting.

### Memory

| part | numbers stored |
|---|---|
| banded LU of $A$ | $(2k_l + k_u + 1)\,n \approx 3(h+1)\,r\,n$ |
| $W = A^{-\top}C$ and $V = A^{-1}B$ (BEM) | $2\,p\,n$ |
| $B$, $C$, $D$ (part of the matrix) | $2\,p\,n + p^2$ |

Everything grows linearly with $N_s$, and there is no fill outside the band.

## Benchmark

Cyclic block-banded matrices as above, with $N_s$ slices of $r$ unknowns, a
stencil reaching $h$ slices on each side, and one dense border row and column
($q = 1$). Random blocks, made diagonally dominant. Three solvers:

* **UMFPACK**: sparse LU of the whole $(rN_s + 1)\times(rN_s + 1)$ matrix, `lu(K)`;
* **BEM**: `lu(M)` with $A$ a `BandedMatrix` and the last $h$ slices and the
  extra unknown as the border;
* **BED**: the same with `alg = :BED`, which does not compute $V = A^{-1}B$.

Times are the best of a few runs, on one core (`BLAS.set_num_threads(1)`) of an
Intel Xeon at 2.8 GHz, Julia 1.11, BandedMatrices.jl 1.14. "Factor" includes
building the blocks from the sparse $K$ (including the conversion of $A$ to a
`BandedMatrix`) and, for BEM/BED, computing $W$ (and $V$) and the Schur
complements; "solve" is one right-hand side. Memory is that of the
factorisation: for UMFPACK $16\,(\mathrm{nnz}(L) + \mathrm{nnz}(U))$ bytes
(values and indices); for BEM/BED the banded LU of $A$ and $W$ (and $V$).
The residual is $\lVert Kx - b\rVert / \lVert b\rVert$. Reproduce with
`julia --project=benchmark benchmark/cyclic.jl h`.

**$h = 1$** (e.g. second-order centred differences, or a one-step scheme between consecutive slices):

| r | N_s | unknowns | factor (s): UMFPACK / BEM / BED | solve (ms): UMFPACK / BEM / BED | memory (MB): UMFPACK / BEM / BED | residual: UMFPACK / BEM / BED |
|---:|---:|---:|:---:|:---:|:---:|:---:|
| 5 | 20 | 101 | 0.000 / 0.000 / 0.000 | 0.00 / 0.01 / 0.00 | 0.0 / 0.0 / 0.0 | 1e-15 / 2e-15 / 2e-15 |
| 5 | 50 | 251 | 0.000 / 0.000 / 0.000 | 0.01 / 0.01 / 0.01 | 0.1 / 0.1 / 0.1 | 3e-16 / 3e-16 / 3e-16 |
| 5 | 100 | 501 | 0.001 / 0.000 / 0.000 | 0.02 / 0.03 / 0.02 | 0.2 / 0.2 / 0.1 | 4e-15 / 5e-15 / 4e-15 |
| 10 | 20 | 201 | 0.000 / 0.000 / 0.000 | 0.01 / 0.01 / 0.01 | 0.2 / 0.1 / 0.1 | 6e-16 / 6e-16 / 6e-16 |
| 10 | 50 | 501 | 0.002 / 0.001 / 0.001 | 0.04 / 0.03 / 0.06 | 0.4 / 0.3 / 0.3 | 6e-16 / 6e-16 / 9e-16 |
| 10 | 100 | 1001 | 0.004 / 0.002 / 0.001 | 0.07 / 0.06 / 0.05 | 0.8 / 0.6 / 0.6 | 4e-16 / 5e-16 / 4e-16 |
| 50 | 20 | 1001 | 0.015 / 0.018 / 0.013 | 0.22 / 0.19 / 0.20 | 3.8 / 3.0 / 2.7 | 9e-16 / 9e-16 / 1e-15 |
| 50 | 50 | 2501 | 0.045 / 0.046 / 0.026 | 0.70 / 0.52 / 0.43 | 9.9 / 7.9 / 6.9 | 1e-14 / 2e-14 / 2e-14 |
| 50 | 100 | 5001 | 0.099 / 0.094 / 0.082 | 1.59 / 2.05 / 1.48 | 20.0 / 15.9 / 13.9 | 9e-16 / 9e-16 / 1e-15 |
| 100 | 20 | 2001 | 0.108 / 0.108 / 0.067 | 2.35 / 1.06 / 0.70 | 15.1 / 12.2 / 10.6 | 2e-15 / 1e-15 / 1e-15 |
| 100 | 50 | 5001 | 0.257 / 0.404 / 0.257 | 3.56 / 6.45 / 3.73 | 39.3 / 31.4 / 27.4 | 2e-15 / 2e-15 / 2e-15 |
| 100 | 100 | 10001 | 0.451 / 1.079 / 0.498 | 8.90 / 9.13 / 9.94 | 79.5 / 63.4 / 55.4 | 1e-15 / 1e-15 / 1e-15 |

**$h = 4$** (eighth-order centred differences):

| r | N_s | unknowns | factor (s): UMFPACK / BEM / BED | solve (ms): UMFPACK / BEM / BED | memory (MB): UMFPACK / BEM / BED | residual: UMFPACK / BEM / BED |
|---:|---:|---:|:---:|:---:|:---:|:---:|
| 5 | 20 | 101 | 0.000 / 0.000 / 0.000 | 0.01 / 0.02 / 0.01 | 0.1 / 0.1 / 0.1 | 7e-16 / 8e-16 / 8e-16 |
| 5 | 50 | 251 | 0.001 / 0.001 / 0.001 | 0.02 / 0.02 / 0.01 | 0.3 / 0.2 / 0.2 | 2e-15 / 2e-15 / 2e-15 |
| 5 | 100 | 501 | 0.002 / 0.001 / 0.002 | 0.05 / 0.03 / 0.03 | 0.7 / 0.4 / 0.4 | 5e-16 / 5e-16 / 5e-16 |
| 10 | 20 | 201 | 0.002 / 0.001 / 0.001 | 0.04 / 0.02 / 0.04 | 0.4 / 0.3 / 0.2 | 2e-15 / 2e-15 / 2e-15 |
| 10 | 50 | 501 | 0.005 / 0.004 / 0.003 | 0.08 / 0.05 / 0.07 | 1.3 / 0.9 / 0.7 | 1e-15 / 1e-15 / 1e-15 |
| 10 | 100 | 1001 | 0.010 / 0.008 / 0.006 | 0.14 / 0.14 / 0.11 | 2.7 / 1.8 / 1.5 | 8e-16 / 8e-16 / 7e-16 |
| 50 | 20 | 1001 | 0.067 / 0.101 / 0.051 | 0.66 / 0.57 / 0.37 | 10.8 / 7.4 / 6.1 | 2e-15 / 1e-15 / 1e-15 |
| 50 | 50 | 2501 | 0.180 / 0.286 / 0.147 | 2.73 / 2.90 / 1.82 | 31.2 / 21.2 / 17.5 | 5e-15 / 5e-15 / 5e-15 |
| 50 | 100 | 5001 | 0.417 / 0.727 / 0.395 | 6.29 / 6.25 / 5.02 | 65.4 / 44.2 / 36.5 | 2e-15 / 2e-15 / 2e-15 |
| 100 | 20 | 2001 | 0.291 / 0.645 / 0.304 | 3.27 / 4.05 / 2.26 | 43.0 / 29.5 / 24.3 | 2e-15 / 2e-15 / 2e-15 |
| 100 | 50 | 5001 | 0.879 / 3.841 / 2.010 | 11.09 / 12.36 / 8.78 | 124.7 / 84.7 / 69.9 | 9e-15 / 9e-15 / 9e-15 |
| 100 | 100 | 10001 | 1.876 / 9.043 / 4.401 | 25.25 / 25.87 / 18.78 | 261.0 / 176.7 / 145.9 | 5e-15 / 5e-15 / 5e-15 |

What the numbers say:

* **Memory**: the factorisation takes less than UMFPACK's, by 20–35% with BEM
  and by 30–45% with BED.
* **Solves** (what a preconditioner repeats at every Krylov iteration) are
  comparable to UMFPACK's, within a factor of about 2 either way (these are
  short, noisy timings); BED is mostly faster.
* **Factorisation** is as fast or faster for small blocks, but slower than
  UMFPACK for large blocks, especially with a wide stencil: up to ~5× with BEM
  and ~2.3× with BED for $r = 100$, $h = 4$. The banded LU itself is cheap; the
  cost is in the $p$ solves with $A^\top$ for $W$ and, for BEM, the $p$ solves
  with $A$ for $V$, with $p = hr + 1$ right-hand sides each.
* Accuracy is the same for all three.

## Tests

```
julia --project -e 'using Pkg; Pkg.test()'
```

The tests use BandedMatrices.jl, SparseArrays and Random (test dependencies only).
