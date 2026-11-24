"""Shared look for the single-panel talk figures (general-audience deck on run B, 2026-11-19).
Large sans-serif text for slides, plain-language labels, no panel letters. Same colours as the paper figures.
Each fig_talk_*.py saves figures/<name>.png (200 dpi, for the slides), <name>.pdf and <name>_preview.png.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import sprott_common as sc

PNG_DPI = 200
LAB_BOARD, LAB_MODEL = "My circuit", "Model"
C_XD = "#2a9d8f"  # x' trace, as in fig_B_attractor

# (x, x') density grid and view, as in fig_B_attractor
EDGES = [np.arange(-2.0, 1.2, 0.04) + 0.01, np.arange(-1.2, 1.0, 0.04) + 0.01]
XLIM, YLIM = (-1.8, 0.9), (-1.05, 1.0)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "lines.linewidth": 1.6,
    }
)


def run_b():
    r = sc.load(sc.B_RUN)
    return r["x"], r["xd"], r["dt"]


def model_like_scope(x, xd):
    """1N4148 model at A = 0.982 with the measured noise and 8-bit steps (cached, nothing fitted)."""
    return sc.scope_like_sim("simple", x, xd)


def density(x, xd):
    return np.histogram2d(x.ravel(), xd.ravel(), EDGES, density=True)[0]


def shared_norm(*hs):
    vmax = max(h.max() for h in hs)
    return LogNorm(vmin=vmax * 1e-4, vmax=vmax)


def show_density(ax, h, norm):
    im = ax.pcolormesh(
        EDGES[0],
        EDGES[1],
        np.ma.masked_equal(h.T, 0),
        cmap=sc.CMAP,
        norm=norm,
        rasterized=True,
    )
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    return im


def density_colorbar(fig, im, ax, norm):
    cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=25)
    cb.minorticks_off()
    cb.set_ticks([norm.vmin, norm.vmax], labels=["rarely", "often"])
    cb.set_label("How often the circuit is here")
    return cb


def save(fig, name):
    sc.FIGS.mkdir(exist_ok=True)
    fig.savefig(sc.FIGS / f"{name}.png", dpi=PNG_DPI)
    return sc.save(fig, name)  # pdf + 100-dpi preview, closes the figure
