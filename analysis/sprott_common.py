"""Shared helpers for the robust analysis of the Sprott (2011) jerk circuit (analysis/fig_*.py).

Model: x''' = -A x'' - x - R1 Is (exp(V2 / (n VT)) - 1), time in units of tau = (R1C1 R2C2 R3C3)^(1/3).
With unequal stages the scope's Ch2 voltage is V2 = g x' (g = (R6/R5)(R3C3/tau)); see writeup/sprott_writeup.md.
  'simple' : R = mean R, A = 0.982, g = 1        (lambda_1 = +0.0986 per tau, lyap_spectrum.py)
  'exact'  : R = R1,     A = 0.982, g = 1.025    (same 1N4148 model, every measured part value used)
Neither has a fitted parameter.
"""

from pathlib import Path

import cmcrameri.cm as cmc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
CACHE = HERE / "cache"

MAIN_RUN = DATA / "sprott_R4_10030_20261119_134608.h5"
NOISE_RUN = DATA / "sprott_noisecheck_20261119_135813.h5"
TEST_RUN = DATA / "sprott_test_20261119_133128.h5"
B_RUN = DATA / "sprott_after_swaps_20261119_181028.h5"  # state B: matches the model (18:10)

# measured parts (DVM, 2026-11-19)
R1, R2, R3, R4, R5, R6 = 9850.0, 9950.0, 9850.0, 10030.0, 9860.0, 9930.0
C1, C2, C3 = 102.1e-9, 99.4e-9, 103.9e-9
TAU = (R1 * C1 * R2 * C2 * R3 * C3) ** (1 / 3)  # 1.006 ms
A_EFF = TAU / (R4 * C1)  # 0.982
G2 = (R6 / R5) * (R3 * C3 / TAU)  # 1.025: Ch2 volts per unit x'
R_MEAN = 9883.0
IS, NID, VT = 2.52e-9, 1.752, 0.02585  # 1N4148
# per tau at A = 0.982: full Lyapunov spectrum, 24 trajectories x 3000 tau (lyap_spectrum.py, 2026-11-19).
# Was 0.110 until then, from a single 1500 tau two-trajectory run (sprott_measured_check.log), ~10 % high.
LAMBDA_SIM = 0.0986

LSB = 0.02  # V per 8-bit step at 0.5 V/div
DT_SCOPE = 40e-6  # s per sample at 10 ms/div
NPTS = 2500

# shared look
C_DATA, C_SIM, C_SIM2 = "#1f4e9c", "#c0392b", "#e08e45"
CMAP = cmc.cmaps["devon_r"]
plt.style.use(["science", "no-latex"])

# peak finding defaults (fig_peaks.py shows the result does not depend on them)
SMOOTH = 9  # samples (0.36 ms) running mean before peak picking
HMIN = 0.25  # V: smallest x' maximum kept (the model has none below ~0.3 V)
PROM = 0.05  # V


def load(path):
    with h5py.File(path, "r") as f:
        r: dict = {k: np.asarray(f[k]) for k in ("x", "xd", "t")}
        r["tcap"] = np.asarray(f["capture_time_s"])
        for k in ("xdd", "mxd"):  # only in runs recorded with more than two channels
            if k in f:
                r[k] = np.asarray(f[k])
        r.update(
            {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        )
    r["name"] = Path(path).stem
    r["dt"] = float(r["t"][1] - r["t"][0])
    return r


def runs_matching(pattern):
    return sorted(p for p in DATA.glob(pattern) if not p.stem.endswith("_fake"))


# ---------------------------------------------------------------- peak picking
def boxcar(v, w):
    if w <= 1:
        return np.asarray(v, float)
    return np.convolve(v, np.ones(w) / w, mode="same")


def xd_maxima(x, xd, smooth=SMOOTH, hmin=HMIN, prom=PROM):
    """Local maxima of x' in one record, from a running mean of width `smooth`.

    Returns (t_idx, x_at_max, xd_max) with sub-sample positions from a parabola through the
    three smoothed points around each peak, so the values are no longer stuck on the 20 mV grid.
    """
    xs, ys = boxcar(x, smooth), boxcar(xd, smooth)
    idx, _ = find_peaks(ys, height=hmin, prominence=prom, distance=max(smooth, 3))
    edge = max(smooth, 2)
    idx = idx[(idx > edge) & (idx < len(ys) - edge - 1)]
    y0, y1, y2 = ys[idx - 1], ys[idx], ys[idx + 1]
    den = y0 - 2 * y1 + y2
    d = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den == 0, -1, den), 0.0)
    ymax = y1 - 0.25 * (y0 - y2) * d
    xmax = np.interp(idx + d, np.arange(len(xs)), xs)
    return idx + d, xmax, ymax


def all_maxima(x, xd, **kw):
    """Peaks for every capture: list of (t_idx, x, xd) arrays."""
    return [xd_maxima(a, b, **kw) for a, b in zip(x, xd)]


# ---------------------------------------------------------------- simulation
def make_rhs(model):
    if model == "simple":
        R, A, g = R_MEAN, A_EFF, 1.0
    elif model == "exact":
        R, A, g = R1, A_EFF, G2
    else:
        raise ValueError(model)

    def rhs(_, s):
        x, y, z = s
        return [y, z, -A * z - x - R * IS * np.expm1(min(g * y / (NID * VT), 60.0))]

    return rhs, g


def sim_windows(
    model="simple", n_win=400, dt_s=DT_SCOPE, npts=NPTS, gap_rc=60.0, seed=1
):
    """Simulated scope windows (volts) sampled like the scope, without noise or steps.

    Windows are separated by `gap_rc` time units so they are nearly independent, like the real
    captures (2.75 s apart = 2700 time units). Cached in analysis/cache/.
    """
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"sim_{model}_{n_win}_{dt_s * 1e6:.0f}us_{npts}_{gap_rc:.0f}_{seed}.npz"
    if f.exists():
        d = np.load(f)
        return d["x"], d["xd"]
    rhs, g = make_rhs(model)
    rng = np.random.default_rng(seed)
    dt = dt_s / TAU
    span = (npts - 1) * dt
    s = solve_ivp(
        rhs, (0, 300), [0.0, 0.4, 0.0], method="LSODA", rtol=1e-8, atol=1e-10
    ).y[:, -1]
    xs, ys = np.empty((n_win, npts)), np.empty((n_win, npts))
    for k in range(n_win):
        gap = gap_rc * (0.5 + rng.random())
        s = solve_ivp(rhs, (0, gap), s, method="LSODA", rtol=1e-8, atol=1e-10).y[:, -1]
        tt = np.arange(npts) * dt
        sol = solve_ivp(
            rhs,
            (0, span),
            s,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
            t_eval=tt,
            max_step=0.05,
        )
        xs[k], ys[k] = sol.y[0], g * sol.y[1]
        s = sol.y[:, -1]
    np.savez_compressed(f, x=xs, xd=ys)
    return xs, ys


def as_scope(x, sigma, seed=2, lsb=LSB):
    """What the scope would record: add white noise of rms `sigma` and round to steps of `lsb`
    (V/div / 25; 20 mV at 0.5 V/div)."""
    rng = np.random.default_rng(seed)
    return np.round((x + rng.normal(0, sigma, x.shape)) / lsb) * lsb


def scope_sigma(v, lsb=LSB):
    """Extra white noise to add to a simulation so its hf_noise matches the data channel v."""
    return float(np.sqrt(max(hf_noise(v) ** 2 - lsb**2 / 12, 0.0)))


def scope_like_sim(model, data_x, data_xd, n_win=400, seed=1, dt_s=DT_SCOPE, lsb=LSB):
    """Simulated windows plus the measured noise and the scope's steps (`lsb`), channel by channel."""
    x, xd = sim_windows(model, n_win=n_win, dt_s=dt_s, seed=seed)
    return as_scope(x, scope_sigma(data_x, lsb), seed=seed + 10, lsb=lsb), as_scope(
        xd, scope_sigma(data_xd, lsb), seed=seed + 20, lsb=lsb
    )


def hf_noise(v):
    """rms of white noise from the second difference (signal curvature is ~1 mV per sample here)."""
    d2 = v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]
    return d2.std() / np.sqrt(6)


# ---------------------------------------------------------------- statistics
def bootstrap(stat, n_items, n_boot=1000, seed=0):
    """Resample items (captures) with replacement; stat(idx) -> array. Returns (n_boot, ...)."""
    rng = np.random.default_rng(seed)
    return np.array([stat(rng.integers(0, n_items, n_items)) for _ in range(n_boot)])


def spectrum(x, dt):
    """Per-capture Hann-window power of x (mean removed); returns f, P (captures x freqs)."""
    w = np.hanning(x.shape[1])
    p = np.abs(np.fft.rfft((x - x.mean(axis=1, keepdims=True)) * w, axis=1)) ** 2
    return np.fft.rfftfreq(x.shape[1], dt), p


def peak_freq(f, p, fmin=20.0, fmax=300.0):
    """Power-weighted centroid of the main lobe (f in fmin..fmax): finer than the 10 Hz bins."""
    m = (f >= fmin) & (f <= fmax)
    ff, pp = f[m], p[..., m]
    k = np.argmax(pp, axis=-1)
    lo, hi = np.maximum(k - 2, 0), np.minimum(k + 3, len(ff))
    out = []
    for row, a, b in zip(np.atleast_2d(pp), np.atleast_1d(lo), np.atleast_1d(hi)):
        out.append(np.sum(ff[a:b] * row[a:b]) / np.sum(row[a:b]))
    return np.array(out) if pp.ndim > 1 else out[0]


def dwell_times(x, xd, r0, dt, smooth=SMOOTH):
    """Lengths (s) of the stretches with sqrt(x^2 + x'^2) < r0 that start and end inside a capture."""
    out = []
    for a, b in zip(x, xd):
        r = np.hypot(boxcar(a, smooth), boxcar(b, smooth))[smooth:-smooth]
        inside = np.concatenate([[0], (r < r0).astype(int), [0]])
        d = np.diff(inside)
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        ok = (st > 0) & (en < len(r))
        out.append((en[ok] - st[ok]) * dt)
    return out


# ---------------------------------------------------------------- figure helpers
def panel(ax, lab, right=False):
    ax.text(
        0.97 if right else 0.03,
        0.97,
        f"({lab})",
        transform=ax.transAxes,
        ha="right" if right else "left",
        va="top",
        fontweight="bold",
    )


def save(fig, name, png=False):
    """Save figures/<name>.pdf and a 100-dpi preview; png=True also writes figures/<name>.png (200 dpi) for
    the GitHub pages, which can't show PDFs inline."""
    FIGS.mkdir(exist_ok=True)
    out = FIGS / f"{name}.pdf"
    fig.savefig(out)
    if png:
        fig.savefig(FIGS / f"{name}.png", dpi=200)
    fig.savefig(FIGS / f"{name}_preview.png", dpi=100)
    plt.close(fig)
    print(f"figure -> {out}")
    return out


def save_results(name, d):
    """Key numbers of one analysis script -> analysis/results/<name>.json (read by fig_summary.py)."""
    import json

    out = HERE / "results"
    out.mkdir(exist_ok=True)

    def conv(v):
        if isinstance(v, dict):
            return {str(k): conv(u) for k, u in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [conv(u) for u in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    (out / f"{name}.json").write_text(json.dumps(conv(d), indent=1))
