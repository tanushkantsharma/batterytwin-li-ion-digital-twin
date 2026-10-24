"""Shared Matplotlib style: fixed categorical order, thin lines, recessive grid, single y-axis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "lines.linewidth": 1.4, "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5,
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
