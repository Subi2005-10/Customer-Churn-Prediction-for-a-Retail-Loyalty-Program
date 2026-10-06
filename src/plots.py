"""Shared chart style: one palette, thin marks, recessive axes, large readable type."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import FIG_DIR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 12, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110,
})


def save(fig, name):
    fig.tight_layout()
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def bar_labels(ax, bars, fmt="{:.0%}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", fontsize=11, color=INK)


def hbar_labels(ax, bars, fmt="{:.0%}"):
    for b in bars:
        w = b.get_width()
        ax.annotate(fmt.format(w), (w, b.get_y() + b.get_height() / 2), ha="left", va="center",
                    xytext=(4, 0), textcoords="offset points", fontsize=11, color=INK)
