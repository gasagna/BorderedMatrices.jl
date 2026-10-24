# Draws assets/logo.svg: the sparsity pattern of a bordered matrix [A B; Cᵀ D], with a banded A,
# the thin border B, Cᵀ, the corner D, and the wrap-around corners of a cyclic matrix moved into
# the border.
#
#     julia assets/make_logo.jl

using Printf

COLORS = (green = "#389826", blue = "#4063d8", purple = "#9558b2", red = "#cb3c33")

n    = 9        # cells of A per side
p    = 2        # border width
cell = 20.0     # cell size
gap  = 3.0      # gap between cells
sep  = 10.0     # extra gap between A and the border
kl   = 2        # bandwidth of A (cells)

# position of cell (i, j); the border is shifted by `sep`
pos(k) = (k - 1) * cell + (k > n ? sep : 0.0)

rects = String[]
cellsvg(i, j, color, opacity) = push!(rects,
    @sprintf("<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" rx=\"3.5\" fill=\"%s\" fill-opacity=\"%.2f\"/>",
             pos(j) + gap / 2, pos(i) + gap / 2, cell - gap, cell - gap, color, opacity))

# A: banded, darker on the diagonal
for i in 1:n, j in 1:n
    d = abs(i - j)
    d <= kl && cellsvg(i, j, COLORS.blue, d == 0 ? 1.0 : (d == 1 ? 0.70 : 0.40))
end
# border: B (right) and Cᵀ (bottom), mostly zero except near the ends, where the wrap-around corners
# of a cyclic matrix end up; plus a dense row/column (e.g. a phase condition and the period)
for i in 1:n, k in 1:p
    dense = k == p
    wrap  = k == 1 && (i <= kl || i > n - kl)
    (dense || wrap) || continue
    cellsvg(i, n + k, dense ? COLORS.green : COLORS.red, dense ? 0.90 : 0.80)
    cellsvg(n + k, i, dense ? COLORS.purple : COLORS.red, dense ? 0.90 : 0.80)
end
# D
for k in 1:p, l in 1:p
    cellsvg(n + k, n + l, "#242936", 0.85)
end

size_ = pos(n + p) + cell
# the partition lines
xs = n * cell + sep / 2
lines = @sprintf("<path d=\"M%.1f -6 V%.1f M-6 %.1f H%.1f\" stroke=\"#98a2b3\" stroke-width=\"2\" stroke-dasharray=\"5 4\"/>",
                 xs, size_ + 6, xs, size_ + 6)
# brackets of the matrix
br = @sprintf("<path d=\"M-8 -10 H-16 V%.1f H-8 M%.1f -10 H%.1f V%.1f H%.1f\" fill=\"none\" stroke=\"#242936\" stroke-width=\"4\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>",
              size_ + 10, size_ + 8, size_ + 16, size_ + 10, size_ + 8)

scale = 200 / (size_ + 32)

svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 300" role="img" aria-labelledby="title desc">
  <title id="title">BorderedMatrices.jl logo</title>
  <desc id="desc">The sparsity pattern of a bordered matrix: a banded block in blue, a thin border of one dense row and column in green and purple, the wrap-around corners of a cyclic matrix in red moved into the border, and the small corner block, beside the BorderedMatrices.jl wordmark.</desc>
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#f2f5f8"/>
    </linearGradient>
    <filter id="panel-shadow" x="-12%" y="-25%" width="124%" height="150%">
      <feDropShadow dx="0" dy="9" stdDeviation="11" flood-color="#172033" flood-opacity="0.14"/>
    </filter>
    <filter id="soft" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#172033" flood-opacity="0.10"/>
    </filter>
  </defs>

  <!-- Main panel -->
  <rect x="24" y="24" width="912" height="252" rx="24" fill="url(#bg)" filter="url(#panel-shadow)"/>

  <!-- Left illustration: sparsity pattern of [A B; Cᵀ D] -->
  <g transform="translate($(@sprintf("%.1f", 206 - scale * size_ / 2)) $(@sprintf("%.1f", 150 - scale * size_ / 2))) scale($(@sprintf("%.3f", scale)))" filter="url(#soft)">
    $lines
    $(join(rects, "\n    "))
    $br
  </g>

  <!-- Wordmark and subtitle -->
  <g transform="translate(372 96)">
    <text x="0" y="48" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="54" font-weight="820" fill="#222734" letter-spacing="0">BorderedMatrices.jl</text>
    <text x="2" y="90" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">Stable block elimination for bordered systems,</text>
    <text x="2" y="118" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">with sparse, dense and banded factorisations.</text>
  </g>
</svg>
"""

write(joinpath(@__DIR__, "logo.svg"), svg)
println("written assets/logo.svg")
