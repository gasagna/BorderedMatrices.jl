"""Figures for the README, from benchmark/results_h*.csv and benchmark/sparsity.txt.

    python3 benchmark/plot.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

D = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e6ea",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#8a93a3", "axes.labelcolor": "#2b3140",
                     "xtick.color": "#4f596b", "ytick.color": "#4f596b"})
INK = "#2b3140"
RCOL = {5: "#86b6ef", 10: "#5598e7", 20: "#2a78d6", 50: "#1c5cab", 100: "#104281", 200: "#081f40"}   # block size r: ordinal blue ramp, light = small
SOLV = {"umfpack": ("UMFPACK, lu(K)", "o", "-"),
        "bem": ("BorderedMatrices, BEM", "s", "--"),
        "bed": ("BorderedMatrices, BED", "^", ":")}

def load(h):
    with open(os.path.join(D, f"results_h{h}.csv")) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]

def legend(fig):
    hs = [Line2D([], [], color=RCOL[r], lw=2, label=f"r = {r}") for r in RCOL]
    hs += [Line2D([], [], color=INK, ls=ls, marker=m, ms=5, lw=1.4, label=lab) for lab, m, ls in SOLV.values()]
    fig.legend(handles=hs, loc="lower center", ncol=9, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))

def panel(ax, rows, key, scale, ylabel, title):
    for r in RCOL:
        R = sorted((x for x in rows if x["r"] == r), key=lambda x: x["N"])
        for s, (lab, m, ls) in SOLV.items():
            ax.plot([x["N"] for x in R], [scale * x[f"{key}_{s}"] for x in R], ls=ls, marker=m, ms=4,
                    lw=1.4, color=RCOL[r], mfc="white" if s == "umfpack" else RCOL[r])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("unknowns  N = r·Nₛ + 1"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9, color=INK)

rows = {h: load(h) for h in (1, 4)}

def save_preview(fig, name):                     # optional PNG copies, for checking the figures
    if os.environ.get("PREVIEW"):
        fig.savefig(os.path.join(os.environ["PREVIEW"], name + ".png"), dpi=110, bbox_inches="tight")

# memory: absolute, and relative to UMFPACK
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.4))
for j, h in enumerate((1, 4)):
    panel(axs[0, j], rows[h], "mem", 1.0, "memory of the factorisation (MB)", f"stencil reaching h = {h} slice{'s' if h > 1 else ''} on each side")
    ax = axs[1, j]
    for r in RCOL:
        R = sorted((x for x in rows[h] if x["r"] == r), key=lambda x: x["N"])
        for s_ in ("bem", "bed"):
            lab, m, ls = SOLV[s_]
            ax.plot([x["N"] for x in R], [x[f"mem_{s_}"] / x["mem_umfpack"] for x in R], ls=ls, marker=m,
                    ms=4, lw=1.4, color=RCOL[r])
    ax.axhline(1, color="#8a93a3", lw=1)
    ax.set_xscale("log"); ax.set_ylim(0, 1.2)
    ax.set_xlabel("unknowns  N = r·Nₛ + 1"); ax.set_ylabel("memory / memory of UMFPACK")
    ax.set_title(f"h = {h}: ratio to UMFPACK (below 1: less memory)", fontsize=9, color=INK)
fig.suptitle("Memory: cyclic block-banded matrix, Nₛ slices of r unknowns", fontsize=10, color=INK)
legend(fig); fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(D, "memory.svg"), bbox_inches="tight"); save_preview(fig, "memory"); plt.close(fig)

# time
fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.4), sharex=True)
for j, h in enumerate((1, 4)):
    panel(axs[0, j], rows[h], "factor", 1.0, "factorisation (s)", f"h = {h}: factorisation")
    panel(axs[1, j], rows[h], "solve", 1e3, "one solve (ms)", f"h = {h}: one solve")
fig.suptitle("Time on one core", fontsize=10, color=INK)
legend(fig); fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(D, "time.svg"), bbox_inches="tight"); save_preview(fig, "time"); plt.close(fig)

# sparsity patterns
with open(os.path.join(D, "sparsity.txt")) as f:
    f.readline()
    N, n, p, kl, ku, nnzK, nnzLU = map(int, f.readline().split())
    Ki, Kj = [list(map(int, f.readline().split())) for _ in range(2)]
    Li, Lj = [list(map(int, f.readline().split())) for _ in range(2)]

def frame(ax, title):
    ax.set_xlim(0.5, N + 0.5); ax.set_ylim(N + 0.5, 0.5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#8a93a3")
    ax.set_title(title, fontsize=9, color=INK)

def border_lines(ax):
    for v in (n + 0.5,):
        ax.axvline(v, color="#8a93a3", lw=0.8, ls="--"); ax.axhline(v, color="#8a93a3", lw=0.8, ls="--")

fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.9))
ax = axs[0]
inner = [(i, j) for i, j in zip(Ki, Kj) if i <= n and j <= n]
bord = [(i, j) for i, j in zip(Ki, Kj) if i > n or j > n]
ax.scatter([j for i, j in inner], [i for i, j in inner], s=1.2, c="#4063d8", marker="s", lw=0)
ax.scatter([j for i, j in bord], [i for i, j in bord], s=1.2, c="#e0712c", marker="s", lw=0)
border_lines(ax)
frame(ax, f"K: {nnzK} nonzeros\n(A in blue, border in red)")
ax = axs[1]
ax.scatter(Lj, Li, s=1.2, c="#9558b2", marker="s", lw=0)
frame(ax, f"UMFPACK: L + U, {nnzLU} nonzeros\n(UMFPACK's ordering)")
ax = axs[2]
# banded LU of A: kl sub-diagonals, ku super-diagonals (ku includes the fill from pivoting)
import numpy as np
I, J = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
band = (J - I <= ku) & (I - J <= kl)
ax.scatter(J[band], I[band], s=1.2, c="#4063d8", marker="s", lw=0)
ax.add_patch(Rectangle((n + 0.5, 0.5), p, n, color="#389826", alpha=0.85, lw=0))     # V = A⁻¹B
ax.add_patch(Rectangle((0.5, n + 0.5), n, p, color="#389826", alpha=0.85, lw=0))     # Wᵀ = (A⁻ᵀC)ᵀ
ax.add_patch(Rectangle((n + 0.5, n + 0.5), p, p, color="#2b3140", alpha=0.85, lw=0))  # Schur complements
border_lines(ax)
stored = int(band.sum()) + 2 * n * p + 2 * p * p
frame(ax, f"BEM: banded LU of A (blue), V, Wᵀ (green),\nSchur complements: {stored} numbers")
fig.suptitle(f"Sparsity: r = 4, Nₛ = 24, h = 2 (N = {N}, border p = {p}); dashed lines: border", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(D, "sparsity.svg"), bbox_inches="tight"); save_preview(fig, "sparsity"); plt.close(fig)
print("written memory.svg, time.svg, sparsity.svg")
