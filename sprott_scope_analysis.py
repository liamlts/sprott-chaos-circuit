"""Analyse Sprott-circuit runs recorded with sprott_scope.py.

Usage
  python3 sprott_scope_analysis.py data/sprott_R4_10000_<time>.h5        # one run: 4-panel figure
  python3 sprott_scope_analysis.py --sweep data/sprott_R4_*.h5           # several R4 values: bifurcation plot

One run gives fig_<run>.pdf (+ _preview.png) in figures/ with
 (a) x' vs x: all captures as a density map, simulation with the run's part values as a thin line
 (b) one capture against time
 (c) Poincare section: (x, x') at each local maximum of x' (Sprott 2011, Fig. 6)
 (d) spectrum of x, averaged over captures, with the linear-theory frequency 0.7926/(2 pi RC) dashed
The sweep gives fig_sweep.pdf: x at each x' maximum against A = R/R4 (Sprott 2011, Fig. 7).
"""

import argparse
from pathlib import Path

import cmcrameri.cm as cmc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the 'science' style)
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
IS, NID, VT = 2.52e-9, 1.752, 0.02585  # 1N4148 model for the simulation line
PEAK_PROM = (
    0.05  # V: minimum prominence of an x' maximum (scope steps are 20 mV at 0.5 V/div)
)
F_LIN = 0.792552 / (
    2 * np.pi
)  # dominant frequency x RC from the linear part (Sprott Eq. 4)

plt.style.use(["science", "no-latex"])


def load(path):
    with h5py.File(path, "r") as f:
        run: dict = {k: np.asarray(f[k]) for k in ("x", "xd", "t")}
        run.update(
            {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        )
    run["name"] = Path(path).stem
    return run


def simulate(r_ohm, c_f, r4_ohm, n_rc=3000.0):
    A = r_ohm / r4_ohm

    def rhs(_, s):
        x, y, z = s
        return [y, z, -A * z - x - r_ohm * IS * np.expm1(min(y / (NID * VT), 60.0))]

    sol = solve_ivp(
        rhs,
        (0, 200 + n_rc),
        [0.0, 0.4, 0.0],
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
        max_step=0.02,
    )
    keep = sol.t > 200
    return (sol.t[keep] - 200) * r_ohm * c_f, sol.y[0, keep], sol.y[1, keep]


def xd_maxima(x, xd, smooth=5):
    """(x, x') at local maxima of x' in one record; a short running mean removes the 8-bit steps."""
    k = np.ones(smooth) / smooth
    xs = np.convolve(xd, k, mode="same")
    idx, _ = find_peaks(xs, prominence=PEAK_PROM, distance=smooth)
    idx = idx[(idx > smooth) & (idx < len(xd) - smooth)]
    return x[idx], xd[idx]


def step_edges(v, lim):
    """Bin edges centred on the scope's voltage steps (8-bit, so values sit on a fixed grid)."""
    u = np.unique(v)
    step = np.min(np.diff(u)[np.diff(u) > 1e-9])
    lo = u[0] - step / 2 - step * np.ceil((u[0] - lim[0]) / step)
    return np.arange(lo, lim[1] + step, step)


def spectrum(x, dt):
    w = np.hanning(x.shape[1])
    p = np.abs(np.fft.rfft((x - x.mean(axis=1, keepdims=True)) * w, axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], dt)
    p = p.mean(axis=0)
    return f, 10 * np.log10(p / p.max())


def one_run(path):
    r = load(path)
    x, xd, t = r["x"], r["xd"], r["t"]
    dt = t[1] - t[0]
    _, xs, xds = simulate(r["R_ohm"], r["C_F"], r["R4_ohm"])

    fig, ax = plt.subplots(2, 2, figsize=(7.0, 6.0))
    a, b, c, d = ax.ravel()

    lim_x = (min(x.min(), xs.min()) - 0.1, max(x.max(), xs.max()) + 0.1)
    lim_y = (min(xd.min(), xds.min()) - 0.1, max(xd.max(), xds.max()) + 0.1)
    h = a.hist2d(
        x.ravel(),
        xd.ravel(),
        bins=[step_edges(x, lim_x), step_edges(xd, lim_y)],  # one bin per scope step
        cmap=cmc.cmaps["devon_r"],
        norm=LogNorm(),
        rasterized=True,
    )
    nshow = len(xs) // 10  # about 30 loops of the simulation, so the data stay visible
    a.plot(xs[:nshow], xds[:nshow], color="#c0392b", lw=0.4, alpha=0.8)
    a.set_xlabel("x, Ch1 (V)")
    a.set_ylabel("x′, Ch2 (V)")
    fig.colorbar(h[3], ax=a, label="samples")

    b.plot(t * 1e3, x[0], lw=0.8, label="x")
    b.plot(t * 1e3, xd[0], lw=0.8, label="x′")
    b.set_xlabel("Time (ms)")
    b.set_ylabel("Voltage (V)")
    b.set_xlim(0, t[-1] * 1e3)
    b.set_ylim(min(x[0].min(), xd[0].min()) - 0.6, None)
    b.legend(frameon=False, loc="lower right", ncol=2)

    peaks = [xd_maxima(xi, yi) for xi, yi in zip(x, xd)]
    px = np.concatenate([pk[0] for pk in peaks])
    py = np.concatenate([pk[1] for pk in peaks])
    sx, sy = xd_maxima(xs, xds, smooth=1)
    c.plot(sx, sy, ".", color="#c0392b", ms=1.5, alpha=0.5, label="simulation")
    c.plot(px, py, ".", color="#1f4e9c", ms=2.0, label="measured")
    c.set_xlabel("x at x′ maximum (V)")
    c.set_ylabel("x′ maximum (V)")
    c.legend(frameon=False, loc="lower left", markerscale=4)

    f, pdb = spectrum(x, dt)
    f_lin = F_LIN / (r["R_ohm"] * r["C_F"])
    d.plot(f, pdb, lw=0.8)
    d.axvline(f_lin, color="grey", ls="--", lw=0.8)
    d.set_xscale("log")
    d.set_xlim(f[1], min(f[-1], 50 * f_lin))
    d.set_ylim(-60, 2)
    d.set_xlabel("Frequency (Hz)")
    d.set_ylabel("Power of x (dB)")

    for axx, lab in zip(ax.ravel(), "abcd"):
        right = lab == "c"  # (c) has data in its top-left corner
        axx.text(
            0.97 if right else 0.03,
            0.97,
            f"({lab})",
            transform=axx.transAxes,
            ha="right" if right else "left",
            va="top",
            fontweight="bold",
        )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"fig_{r['name']}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_name(out.stem + "_preview.png"), dpi=100)
    plt.close(fig)

    fpk = f[1:][np.argmax(pdb[1:])]
    print(
        f"{r['name']}: {x.shape[0]} captures, {dt * 1e6:.0f} us/sample, clipped captures: {r.get('clipped_captures', 0)}"
    )
    print(
        f"  measured  x [{x.min():+.2f}, {x.max():+.2f}] V   x' [{xd.min():+.2f}, {xd.max():+.2f}] V   peak {fpk:.0f} Hz"
    )
    print(
        f"  simulated x [{xs.min():+.2f}, {xs.max():+.2f}] V   x' [{xds.min():+.2f}, {xds.max():+.2f}] V   "
        f"linear theory {f_lin:.0f} Hz"
    )
    print(f"  Poincare points: {len(px)} measured, {len(sx)} simulated")
    print(f"  figure -> {out}")


def sweep(paths):
    runs = sorted((load(p) for p in paths), key=lambda r: r["A"])
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for r in runs:
        px = np.concatenate([xd_maxima(xi, yi)[0] for xi, yi in zip(r["x"], r["xd"])])
        ax.plot(np.full(px.size, r["A"]), px, ".", color="#1f4e9c", ms=1.0, alpha=0.5)
    ax.set_xlabel("A = R / R4")
    ax.set_ylabel("x at x′ maximum (V)")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "fig_sweep.pdf"
    fig.savefig(out)
    fig.savefig(out.with_name(out.stem + "_preview.png"), dpi=100)
    print(
        f"{len(runs)} runs, A from {runs[0]['A']:.2f} to {runs[-1]['A']:.2f} -> {out}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("files", nargs="+")
    p.add_argument(
        "--sweep", action="store_true", help="bifurcation plot over several runs"
    )
    a = p.parse_args()
    if a.sweep:
        sweep(a.files)
    else:
        for fpath in a.files:
            one_run(fpath)
