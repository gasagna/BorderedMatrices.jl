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

## Tests

```
julia --project test/runtests.jl
```
