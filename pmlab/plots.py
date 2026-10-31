"""One chart style for both notebooks."""

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
COIN_COLORS = {"btc": "#2a78d6", "eth": "#eb6834", "sol": "#1baf7a", "xrp": "#eda100",
               "doge": "#e87ba4", "bnb": "#008300"}


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "figure.constrained_layout.use": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9,
    })


def save(fig, name):
    from .data import ROOT

    fig.savefig(ROOT / "figures" / f"{name}.png")
    plt.show()
