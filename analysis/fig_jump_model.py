"""Does adding the measured jumps of x to the model reproduce state A?

Model ('simple' 1N4148, A = 0.982, unchanged) plus one hidden two-state element sigma in {0, 1}:
    x_int' = x',  x'' = z,  z' = -A z - x_c - R Is (exp(x' / n VT) - 1),   x_c = x_int + DELTA * sigma
x_c is the circuit's x (feeds integrator 1 and is what Ch1 records); Ch2 records x'. Nothing is tuned:
  DELTA       per-flip jump of x - integral(x') in state A (check_telegraph.log: up median +0.48 V; the down
              median, 0.24 V, is run as a bracket; DELTA = 0 is the plain model)
  flip rates  the hazard measured in state A against x (check_flip_hazard, the 'x' rows), per ms, used as a
              function of x_c, linearly interpolated (empty bins take the nearest value).
Fixed-step RK4 (dt = scope step / 10 = 0.004 tau, as in the A scan), N_WIN independent windows of 2500 scope
samples after a 200 tau transient, then the measured state-A noise and 8-bit steps (sc.as_scope).
Every statistic is computed with the same code on data and simulation.
(a)-(c) density of (x, x') for state A, the model with jumps, the plain model    (d) x - integral(x'), one window each
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import median_filter

import fig_diode_inferred as fd
import sprott_common as sc
from check_flip_hazard import BINS, collect
from check_telegraph import MED, levels
from check_x_vs_xd import derivs
from fig_diode_inferred import RUNS
from fig_x_minus_int import u_of

DELTAS = (0.48, 0.24, 0.0)
N_WIN, T_TRANS, SUB = 200, 200.0, 10
RUN_A = "A (D1, 13:46)"
EDGES = (np.arange(-1.81, 1.0, 0.04), np.arange(-1.11, 1.0, 0.04))  # 2 scope steps, centred on them


def measured_rates():
    """Flip hazard (per tau) at the bin centres of check_flip_hazard, for leaving low (up) and high (down)."""
    dt, state, var, flips, keep = collect(RUN_A)
    c = 0.5 * (BINS[1:] + BINS[:-1])
    out = {}
    for lv in (0, 1):
        idx = np.array(flips[lv])
        occ = np.histogram(var["x"][(state == lv) & keep], BINS)[0] * dt * 1e3
        hit = np.histogram(var["x"][idx[:, 0], idx[:, 1]], BINS)[0]
        hz = np.where(occ > 20, hit / np.maximum(occ, 1e-9), np.nan)
        ok = np.isfinite(hz)
        out[lv] = np.interp(c, c[ok], hz[ok]) * sc.TAU * 1e3  # per ms -> per tau
    return c, out


def simulate(delta, c, rates, seed=3):
    rng = np.random.default_rng(seed)
    h = sc.DT_SCOPE / sc.TAU / SUB
    R, A, k = sc.R_MEAN, sc.A_EFF, sc.R_MEAN * sc.IS

    def f(xi, y, z, sig):
        xc = xi + delta * sig
        return y, z, -A * z - xc - k * np.expm1(np.minimum(y / (sc.NID * sc.VT), 60.0))

    xi = rng.normal(0, 0.3, N_WIN)
    y = 0.4 + rng.normal(0, 0.1, N_WIN)
    z = np.zeros(N_WIN)
    sig = rng.integers(0, 2, N_WIN).astype(float)
    n_tr = int(T_TRANS / h)
    xs, ys = np.empty((N_WIN, sc.NPTS)), np.empty((N_WIN, sc.NPTS))
    for i in range(n_tr + sc.NPTS * SUB):
        if i >= n_tr and (i - n_tr) % SUB == 0:
            j = (i - n_tr) // SUB
            xs[:, j], ys[:, j] = xi + delta * sig, y
        k1 = f(xi, y, z, sig)
        k2 = f(xi + h / 2 * k1[0], y + h / 2 * k1[1], z + h / 2 * k1[2], sig)
        k3 = f(xi + h / 2 * k2[0], y + h / 2 * k2[1], z + h / 2 * k2[2], sig)
        k4 = f(xi + h * k3[0], y + h * k3[1], z + h * k3[2], sig)
        xi = xi + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y = y + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z = z + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        if delta:
            xc = xi + delta * sig
            rate = np.where(
                sig == 0, np.interp(xc, c, rates[0]), np.interp(xc, c, rates[1])
            )
            sig = np.where(rng.random(N_WIN) < rate * h, 1 - sig, sig)
    assert np.all(np.isfinite(xs)), "diverged"
    return xs, ys


def stats(x, xd, dt):
    """The state-A fingerprints, same code for data and simulation."""
    f, p = sc.spectrum(xd, dt)
    r = (
        np.hypot(sc.boxcar(x, sc.SMOOTH), sc.boxcar(xd, sc.SMOOTH))
        if x.ndim == 1
        else None
    )
    near = np.mean(
        [
            np.mean(np.hypot(sc.boxcar(a, sc.SMOOTH), sc.boxcar(b, sc.SMOOTH)) < 0.2)
            for a, b in zip(x, xd)
        ]
    )
    y, _, _, xsm, dxdt = derivs(x, xd, dt)
    s = np.s_[:, fd.W : -fd.W]
    corr = np.corrcoef(y[s].ravel(), dxdt[s].ravel())[0, 1]
    run = {"x": x, "xd": xd, "dt": dt}
    us = np.array([median_filter(u_of(run, c)[2], MED) for c in range(x.shape[0])])
    lo, hi = levels(us.ravel())
    _, _, _, D = fd.terms(x, xd, dt)
    st = (us > 0.5 * (lo + hi))[:, fd.W3 : -fd.W3].ravel()
    yy = y[:, fd.W3 : -fd.W3].ravel()
    off = yy < fd.OFF
    del r
    return dict(
        rms_x=float(x.std()),
        rms_xd=float(xd.std()),
        mean_x=float(x.mean()),
        mean_xd_mV=float(xd.mean() * 1e3),
        f_Hz=float(np.median(sc.peak_freq(f, p))),
        near_origin=float(near),
        corr_xd_dxdt=float(corr),
        u_step_mV=float((hi - lo) * 1e3),
        Doff_split_mV=float((D[off & st].mean() - D[off & ~st].mean()) * 1e3),
    )


def density(ax, x, xd, title):
    H, _, _ = np.histogram2d(x.ravel(), xd.ravel(), EDGES, density=True)
    ax.pcolormesh(
        EDGES[0],
        EDGES[1],
        H.T,
        norm=LogNorm(H.max() * 1e-4, H.max()),
        cmap=sc.CMAP,
        rasterized=True,
    )
    ax.set_xlabel("x (V)")
    ax.set_ylabel("x′ (V)")
    ax.set_title(title, fontsize=8)


def main():
    c, rates = measured_rates()
    print("rates per ms, x bins " + " ".join(f"{q:+.1f}" for q in c))
    for lv in (0, 1):
        print(
            f"  {'up  ' if lv == 0 else 'down'} "
            + " ".join(f"{q / sc.TAU / 1e3:.2f}" for q in rates[lv])
        )
    ra = sc.load(sc.DATA / RUNS[RUN_A])
    sx, sxd = sc.scope_sigma(ra["x"]), sc.scope_sigma(ra["xd"])
    sets = {"state A (data)": (ra["x"], ra["xd"])}
    rb = sc.load(sc.DATA / RUNS["B (D1 reseated)"])
    sets["state B (data)"] = (rb["x"], rb["xd"])
    for d in DELTAS:
        xs, ys = simulate(d, c, rates)
        sets[f"model, jumps {d:.2f} V" if d else "model, no jumps"] = (
            sc.as_scope(xs, sx, seed=11),
            sc.as_scope(ys, sxd, seed=12),
        )
    res = {}
    keys = None
    for k, (x, xd) in sets.items():
        res[k] = stats(x, xd, sc.DT_SCOPE)
        if keys is None:
            keys = list(res[k])
            print(f"{'':24s}" + "".join(f"{q:>14s}" for q in keys))
        print(f"{k:24s}" + "".join(f"{res[k][q]:14.3f}" for q in keys))
    sc.save_results("jump_model", res)

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6), layout="constrained")
    density(axs[0, 0], *sets["state A (data)"], "State A (data)")
    density(axs[0, 1], *sets["model, jumps 0.48 V"], "Model + measured jumps (0.48 V)")
    density(axs[1, 0], *sets["model, no jumps"], "Model, no jumps")
    ax = axs[1, 1]
    t = np.arange(sc.NPTS) * sc.DT_SCOPE * 1e3
    for k, off, col in (
        ("state A (data)", 0.0, sc.C_DATA),
        ("model, jumps 0.48 V", -1.2, sc.C_SIM),
        ("model, no jumps", -2.2, "grey"),
    ):
        x, xd = sets[k]
        ax.plot(
            t,
            u_of({"x": x, "xd": xd, "dt": sc.DT_SCOPE}, 3)[2] + off,
            color=col,
            lw=0.6,
            label=k,
        )
    ax.set_xlim(0, 100)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("x − ∫x′ (V, shifted)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    for axx, lab in zip(axs.ravel(), "abcd"):
        sc.panel(axx, lab)
    sc.save(fig, "fig_jump_model")


if __name__ == "__main__":
    main()
