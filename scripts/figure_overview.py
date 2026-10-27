"""Fig. 1: overview of MedHyperDiff, drawn in the figures4papers house style (github.com/ChenLiu-1996/figures4papers).

Style taken from the repository's schematics (figure_ImmunoStruct, figure_VIGIL/plot_concept.py): Helvetica, pastel
rounded stage boxes colour-coded by stage, shaded block arrows, small drawn graphs and a data-driven inset. The canvas
is drawn at twice the print width of the Springer Nature template (131 mm), so 16 pt text prints at about 8 pt.

  python scripts/figure_overview.py <output folder>      # writes Figure1.pdf and Figure1.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrow, FancyBboxPatch, Rectangle

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)

INK = "#2B2B2B"
BLUE = "#0F4D92"
STAGE = {  # (fill, edge, title colour)
    "data": ("#EEEEEE", "#7A7A7A", "#4A4A4A"),
    "encode": ("#DCEBCF", "#6E9A4F", "#3F6B25"),
    "diffuse": ("#D3E5F7", "#3775BA", BLUE),
    "decode": ("#FBEFC8", "#C9A43A", "#8A6A12"),
    "eval": ("#EAE1F2", "#8A6BAE", "#5B3F80"),
}
CELL = ["#F6CFCB", "#E9A6A1", "#AADCA9", "#8BCF8B", "#CFE4F7", "#9DC3E6", "#FBEFC8", "#E6D79A"]


def box(ax, x, y, w, h, stage, dashed=False, radius=1.6):
    fill, edge, _ = STAGE[stage]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
                                edgecolor=edge, linewidth=2.2, linestyle=(0, (5, 3)) if dashed else "-", zorder=1))


def title(ax, x, y, text, stage, size=14.5):
    ax.text(x, y, text, ha="center", va="top", fontsize=size, fontweight="bold", color=STAGE[stage][2], zorder=5)


def block_arrow(ax, x0, y0, dx, dy, width=2.6):
    ax.add_patch(FancyArrow(x0, y0, dx, dy, width=width, head_width=width * 1.9, head_length=1.6,
                            length_includes_head=True, facecolor="#C9C9C9", edgecolor="#6F6F6F", linewidth=1.4,
                            zorder=3))


def node(ax, x, y, label, r=2.45, fill="white", edge=INK, color=INK, size=11):
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r, facecolor=fill, edgecolor=edge, linewidth=2.0, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=size, color=color, zorder=5)


def table(ax, x, y, w, h, rows, header, rng, synthetic=False):
    cols = len(header)
    cw, rh = w / cols, h / (rows + 1)
    for j, name in enumerate(header):
        ax.add_patch(Rectangle((x + j * cw, y + rows * rh), cw, rh, facecolor="#D6D6D6", edgecolor="white",
                               linewidth=1.2, zorder=3))
        ax.text(x + (j + 0.5) * cw, y + (rows + 0.5) * rh, name, ha="center", va="center", fontsize=10.5, zorder=4)
    for i in range(rows):
        for j in range(cols):
            c = CELL[rng.integers(len(CELL))]
            ax.add_patch(Rectangle((x + j * cw, y + i * rh), cw, rh, facecolor=c, edgecolor="white", linewidth=1.2,
                                   alpha=0.85 if synthetic else 1.0, zorder=3))


def main():
    plt.rcParams.update({"font.family": "helvetica", "font.size": 15, "mathtext.fontset": "dejavusans"})
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(10.31, 5.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 103.1)
    ax.set_ylim(0, 55.5)
    ax.axis("off")

    top, bottom = 54.5, 21.0
    # ------------------------------------------------------------- real table
    title(ax, 6.0, top - 6.0, "Real table", "data", size=13.5)
    table(ax, 0.6, 26.5, 10.8, 18.5, 7, ["RBC", "MCV", "Hct"], rng)
    ax.text(6.0, 24.6, "fit rows", ha="center", va="top", fontsize=12, color="#5A5A5A")
    block_arrow(ax, 11.9, 36.0, 3.4, 0)

    # ------------------------------------------------------------- 1 encode
    box(ax, 15.6, bottom, 22.0, top - bottom, "encode")
    title(ax, 26.6, top - 1.0, "1  Encode relations", "encode")
    ax.add_patch(Ellipse((26.6, 43.2), 17.0, 10.6, facecolor="#6E9A4F", alpha=0.18, edgecolor="#6E9A4F",
                         linewidth=1.4, zorder=2))
    node(ax, 21.6, 45.2, "Hct")
    node(ax, 31.6, 45.2, "MCV")
    node(ax, 26.6, 40.3, "RBC")
    ax.text(26.6, 35.2, r"Hct $\approx$ RBC$\,\cdot\,$MCV / 10", ha="center", va="center", fontsize=12)
    ax.annotate("", xy=(26.6, 30.2), xytext=(26.6, 33.4),
                arrowprops=dict(arrowstyle="-|>", color=STAGE["encode"][1], linewidth=2.0))
    ax.text(27.4, 31.8, "target RBC", ha="left", va="center", fontsize=11.5, color=STAGE["encode"][2])
    ax.text(26.6, 27.6, r"$r$ = RBC $-$ 10$\,\cdot\,$Hct / MCV", ha="center", va="center", fontsize=12)
    # data-driven inset: the residual keeps the rounding noise of real records
    xs = np.linspace(-3.2, 3.2, 200)
    dens = np.exp(-0.5 * xs ** 2)
    ax.fill_between(20.4 + xs * 1.25, 22.3, 22.3 + dens * 2.6, color="#6E9A4F", alpha=0.35, linewidth=0, zorder=3)
    ax.plot(20.4 + xs * 1.25, 22.3 + dens * 2.6, color="#3F6B25", linewidth=1.6, zorder=4)
    ax.text(25.2, 23.6, "residual noise\nis kept", ha="left", va="center", fontsize=10.5, color="#3F6B25",
            linespacing=1.0)
    block_arrow(ax, 38.0, 36.0, 3.4, 0)

    # ------------------------------------------------------------- 2 diffusion
    box(ax, 41.8, bottom, 24.6, top - bottom, "diffuse")
    title(ax, 54.1, top - 1.0, "2  Hypergraph diffusion", "diffuse")
    # noise to data: one record as a row of cells
    clean = ["#9DC3E6", "#CFE4F7", "#3775BA", "#9DC3E6", "#0F4D92"]
    for k, (x0, lab) in enumerate([(43.6, r"$x_T$"), (57.6, r"$x_0$")]):
        for j in range(5):
            if k == 0:
                c = plt.cm.Greys(rng.uniform(0.25, 0.7))
            else:
                c = clean[j]
            ax.add_patch(Rectangle((x0 + j * 1.45, 47.0), 1.45, 1.9, facecolor=c, edgecolor="white", linewidth=1.0,
                                   zorder=3))
        ax.text(x0 + 3.6, 45.9, lab, ha="center", va="top", fontsize=13)
    ax.annotate("", xy=(57.0, 47.95), xytext=(51.4, 47.95),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, linewidth=2.0))
    ax.text(54.2, 46.4, "denoise", ha="center", va="top", fontsize=10.5, color=BLUE)
    # denoiser over variables and residuals
    pos = {"a": (45.6, 38.6), "b": (49.6, 41.8), "c": (46.6, 32.8), "h": (52.6, 34.4), "m": (58.0, 39.6),
           "r": (60.6, 32.6)}
    for u, v in [("a", "b"), ("a", "c"), ("b", "h"), ("c", "h"), ("b", "m")]:
        ax.plot(*zip(pos[u], pos[v]), color="#9A9A9A", linewidth=1.6, zorder=2)
    ax.add_patch(Ellipse((57.2, 35.3), 13.4, 10.6, angle=-12, facecolor=BLUE, alpha=0.14, edgecolor=BLUE,
                         linewidth=1.4, zorder=2))
    node(ax, *pos["a"], r"$x_1$", r=1.9)
    node(ax, *pos["b"], r"$x_2$", r=1.9)
    node(ax, *pos["c"], r"$x_3$", r=1.9)
    node(ax, *pos["h"], "Hct", r=2.45)
    node(ax, *pos["m"], "MCV", r=2.45, size=10.5)
    node(ax, *pos["r"], r"$r$", r=2.0, fill=BLUE, edge=BLUE, color="white")
    ax.plot([43.6, 46.0], [27.6, 27.6], color="#9A9A9A", linewidth=1.8)
    ax.text(46.6, 27.6, "association graph", ha="left", va="center", fontsize=10.5)
    ax.add_patch(Ellipse((44.8, 24.6), 2.6, 1.6, facecolor=BLUE, alpha=0.14, edgecolor=BLUE, linewidth=1.2))
    ax.text(46.6, 24.6, "relation hyperedge", ha="left", va="center", fontsize=10.5)
    ax.text(54.1, 22.0, "Gaussian and masked diffusion", ha="center", va="center", fontsize=10.5,
            color="#3D3D3D")
    block_arrow(ax, 66.8, 36.0, 3.4, 0)

    # ------------------------------------------------------------- 3 decode
    box(ax, 70.6, bottom, 21.4, top - bottom, "decode")
    title(ax, 81.3, top - 1.0, "3  Decode", "decode")
    steps = [("Round to recorded grid", r"$\hat{x}_j \leftarrow \rho_j(\hat{x}_j)$"),
             ("Rebuild the target", r"$\hat{x}_t = \rho_t\,(g(\hat{x}_{P \backslash t}) + \hat{r})$"),
             ("Re-derive exact formulas", r"$\hat{x}_d = \rho_d\,(f(\hat{x}_O))$")]
    for i, (head, eq) in enumerate(steps):
        y = 47.2 - i * 7.6
        ax.add_patch(Ellipse((72.9, y), 2.4, 2.4, facecolor=STAGE["decode"][1], edgecolor="none", zorder=3))
        ax.text(72.9, y, str(i + 1), ha="center", va="center", fontsize=11, color="white", fontweight="bold",
                zorder=4)
        ax.text(74.6, y, head, ha="left", va="center", fontsize=11)
        ax.text(74.6, y - 3.0, eq, ha="left", va="center", fontsize=11.5)
    ax.add_patch(FancyBboxPatch((72.0, 22.4), 18.6, 3.9, boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor="white", edgecolor=STAGE["decode"][1], linewidth=1.6, zorder=3))
    ax.text(81.3, 24.35, r"RBC = 10$\,\cdot\,$Hct / MCV + $\hat{r}$", ha="center", va="center", fontsize=10.5,
            zorder=4)
    block_arrow(ax, 92.4, 36.0, 3.4, 0)

    # ------------------------------------------------------------- synthetic table
    ax.text(103.0, top - 6.0, "Synthetic", ha="right", va="top", fontsize=13.5, fontweight="bold", color=STAGE["data"][2])
    table(ax, 96.2, 26.5, 6.6, 18.5, 7, ["", ""], np.random.default_rng(11), synthetic=True)

    # ------------------------------------------------------------- 4 evaluate
    box(ax, 0.6, 0.6, 101.8, 15.2, "eval", dashed=True)
    ax.text(1.9, 14.6, "4  Evaluate on held-out records", ha="left", va="top", fontsize=17, fontweight="bold",
            color=STAGE["eval"][2])
    ax.text(40.5, 14.4, "against published generators and controls with the same relation knowledge",
            ha="left", va="top", fontsize=12, color="#3D3D3D")
    metrics = [("Realism", "two-sample test"), ("Relation fidelity", "residual distance"),
               ("Utility", "train synthetic, test real"), ("Privacy", "membership attacks")]
    w, gap = 23.2, 1.85
    for i, (head, sub) in enumerate(metrics):
        x = 2.0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 2.0), w, 7.6, boxstyle="round,pad=0,rounding_size=1.2", facecolor="white",
                                    edgecolor=STAGE["eval"][1], linewidth=1.8, zorder=3))
        ax.text(x + w / 2, 7.1, head, ha="center", va="center", fontsize=13.5, fontweight="bold", zorder=4)
        ax.text(x + w / 2, 4.2, sub, ha="center", va="center", fontsize=12, zorder=4)
    block_arrow(ax, 99.5, 25.5, 0, -9.4, width=2.2)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Figure1.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
