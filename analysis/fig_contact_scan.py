"""Contact resistance in series with D1: can it give the board's states A, B and C at A = 0.982?

Tests the idea that a poor D1 contact made states A and C. Model as everywhere else ('simple': R = R_MEAN,
1N4148 SPICE values, A = 0.982, g = 1) plus an ohmic resistance Rc in series with the diode. Rc is scanned, not
fitted; Rc = 0 is the usual model (check: rms x, time near the origin, lambda_1). With Rc the diode current solves
I = Is (exp((x' - I Rc) / n VT) - 1), which has the closed form
    I = (n VT / Rc) omega(z) - Is,   z = ln(Is Rc / n VT) + (x' + Is Rc) / n VT,
omega = Wright omega function, W(e^z) (no overflow for large x').
Same RK4 as fig_model_scan.py (dt 0.004 tau), N_IC random starts per Rc, T_TRANS transient, T_REC recorded.
Largest Lyapunov exponent from one tangent vector (Jacobian with c = R dI/dx'), renormalised every 0.5 tau.

(a) x' maxima against Rc
(b) rms x, (c) time within 0.2 V of the origin, (d) mean diode current (= -<x>/R), (e) lambda_1, against Rc.
Board states A, B, C (same parts, A = 0.982) as grey dashed lines in (b)-(d); B's lambda_1 in (e).
Rc on a log axis: it spans four decades.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wrightomega

import sprott_common as sc
from check_window_C import levels
from fig_model_scan import DT, NVT, R0, STATES, classify, data_maxima, starts

RC = np.concatenate([[0.0], np.geomspace(1.0, 3e4, 76)])  # ohm
N_IC = 6
T_TRANS, T_REC = 1000.0, 1000.0  # tau
T_REN = 0.5  # tau between tangent renormalisations
CACHE_F = sc.CACHE / "contact_scan.npz"
X_PLOT = (1.0, 2e3)  # ohm: above ~2 kOhm the orbit grows to volts, above ~5 kOhm it runs away
A_WIN = 0.955  # the model's period-4 window in A (Rc = 0), for comparison with the contact's period 4
CACHE_W = sc.CACHE / "contact_scan_Awin.npz"
BOARD_KEYS = {"A (lingering)": "A", "B (D1 reseated)": "B", "C (reseated again)": "C"}


def diode_i(y, rc, logk):
    """Diode current (A) for voltage y across diode + contact. rc, logk = ln(Is rc / n VT) per trajectory."""
    out = sc.IS * np.expm1(np.minimum(y / NVT, 60.0))
    nz = rc > 0
    z = logk[nz] + (y[nz] + sc.IS * rc[nz]) / NVT
    out[nz] = NVT / rc[nz] * wrightomega(z) - sc.IS
    return out


def integrate(rc, s0, A=sc.A_EFF):
    s = s0.astype(float).copy()
    n = rc.size
    logk = np.log(np.where(rc > 0, sc.IS * rc / NVT, 1.0))
    rng = np.random.default_rng(7)
    q = rng.normal(size=(3, n))
    q /= np.linalg.norm(q, axis=0)
    bad = np.zeros(n, bool)

    def f(s, q):
        x, y, z = s
        i = diode_i(y, rc, logk)
        c = sc.R_MEAN * (i + sc.IS) / (NVT + (i + sc.IS) * rc)  # R dI/dy
        ds = np.array([y, z, -A * z - x - sc.R_MEAN * i])
        dq = np.array([q[1], q[2], -q[0] - c * q[1] - A * q[2]])
        return ds, dq

    def step(s, q):
        a1, b1 = f(s, q)
        a2, b2 = f(s + 0.5 * DT * a1, q + 0.5 * DT * b1)
        a3, b3 = f(s + 0.5 * DT * a2, q + 0.5 * DT * b2)
        a4, b4 = f(s + DT * a3, q + DT * b3)
        return (
            s + DT / 6 * (a1 + 2 * a2 + 2 * a3 + a4),
            q + DT / 6 * (b1 + 2 * b2 + 2 * b3 + b4),
        )

    n_ren = int(round(T_REN / DT))
    for k in range(int(T_TRANS / DT)):
        s, q = step(s, q)
        if k % n_ren == 0:
            q /= np.linalg.norm(q, axis=0)
            b = ~np.isfinite(s).all(0) | (np.abs(s).max(0) > 50)
            bad |= b
            s[:, b] = 0.0
            q[:, b] = 1.0
    q /= np.linalg.norm(q, axis=0)
    n_rec = int(T_REC / DT)
    sx, sx2, near, logsum = (np.zeros(n) for _ in range(4))
    y2, y1 = s[1].copy(), s[1].copy()
    mi, mh = [], []
    for k in range(n_rec):
        s, q = step(s, q)
        x, y = s[0], s[1]
        sx += x
        sx2 += x * x
        near += np.hypot(x, y) < R0
        pk = (y1 > y2) & (y1 >= y) & (y1 > 0.005)
        if pk.any():
            j = np.nonzero(pk)[0]
            mi.append(j)
            mh.append(y1[j])
        y2, y1 = y1, y.copy()
        if (k + 1) % n_ren == 0:
            nq = np.linalg.norm(q, axis=0)
            logsum += np.log(nq)
            q /= nq
            b = ~np.isfinite(s).all(0) | (np.abs(s).max(0) > 50)
            bad |= b
            s[:, b] = 0.0
            q[:, b] = 1.0
        if k % 50000 == 0:
            print(f"  recorded {k * DT:.0f} / {T_REC:.0f} tau", flush=True)
    mean = sx / n_rec
    return dict(
        mean_x=mean,
        rms=np.sqrt(np.maximum(sx2 / n_rec - mean**2, 0)),
        near=near / n_rec,
        lam=logsum / T_REC,
        bad=bad,
        mi=np.concatenate(mi),
        mh=np.concatenate(mh),
    )


def run():
    if CACHE_F.exists():
        return dict(np.load(CACHE_F))
    rc = np.repeat(RC, N_IC)
    print(
        f"integrating {rc.size} trajectories ({RC.size} Rc x {N_IC} starts)", flush=True
    )
    out = integrate(rc, starts(rc.size, np.random.default_rng(3)))
    out["rc"] = rc
    np.savez_compressed(CACHE_F, **out)
    return out


def window_ref():
    """Rc = 0 at A = A_WIN (the period-4 window in A), same starts and lengths, to compare with the contact's P4."""
    if CACHE_W.exists():
        return dict(np.load(CACHE_W))
    out = integrate(np.zeros(N_IC), starts(N_IC, np.random.default_rng(3)), A=A_WIN)
    np.savez_compressed(CACHE_W, **out)
    return out


def summarise_p4(mi, mh, idx, rms, near, mean_x):
    lv = np.mean([levels(mh[mi == j][-300:], 4) for j in idx], 0)
    return dict(
        levels=lv.tolist(),
        rms=float(rms[idx].mean()),
        near=float(near[idx].mean()),
        i_uA=float(-mean_x[idx].mean() / sc.R_MEAN * 1e6),
    )


def board_states():
    res = {}
    for name, key in BOARD_KEYS.items():
        ys, rms, near = data_maxima(STATES[name])
        mx = float(sc.load(sc.DATA / STATES[name])["x"].mean())
        res[key] = dict(rms=rms, near=near, i_mean_uA=-mx / sc.R_MEAN * 1e6, maxima=ys)
    return res


def main():
    d = run()
    rc, n = d["rc"], d["rc"].size
    per = classify(d["mi"], d["mh"], n)
    ok = ~d["bad"]
    i_mean = -d["mean_x"] / sc.R_MEAN * 1e6  # uA
    board = board_states()
    lam_b = json.loads((sc.HERE / "results" / "B_lyapunov.json").read_text())

    rows = []
    print(f"{'Rc (ohm)':>9s}  type           rms x   near   I (uA)  lambda_1")
    for r in RC:
        j = np.nonzero((rc == r) & ok)[0]
        if j.size == 0:
            rows.append(dict(rc=float(r), types=["diverged"], n_ok=0))
            continue
        lab = sorted(set("chaos" if p > 20 else f"P{p}" for p in per[j]))
        row = dict(
            rc=float(r),
            types=lab,
            n_ok=int(j.size),
            rms=float(d["rms"][j].mean()),
            near=float(d["near"][j].mean()),
            i_uA=float(i_mean[j].mean()),
            lam=float(d["lam"][j].mean()),
            lam_sd=float(d["lam"][j].std()),
        )
        rows.append(row)
        print(
            f"{r:9.1f}  {','.join(lab):13s} {row['rms']:.3f}  {row['near']:.3f}  {row['i_uA']:6.1f}  "
            f"{row['lam']:+.4f}"
        )
    print(f"diverged: {int(d['bad'].sum())} of {n}")
    for k, b in board.items():
        print(
            f"board {k}: rms x {b['rms']:.3f}, near {b['near']:.3f}, mean diode current {b['i_mean_uA']:.1f} uA"
        )

    # periodic orbits against state C's four x' maxima levels
    lev_c = levels(board["C"]["maxima"][board["C"]["maxima"] > 0.3])
    per_rows = []
    for r in RC:
        j = np.nonzero((rc == r) & ok & (per >= 2) & (per <= 8))[0]
        for jj in j:
            h = d["mh"][d["mi"] == jj][-300:]
            lv = levels(h, per[jj])
            per_rows.append(
                dict(
                    rc=float(r), period=int(per[jj]), levels=lv, rms=float(d["rms"][jj])
                )
            )
    if per_rows:
        print("periodic trajectories (period 2-8):")
        for p in per_rows:
            print(
                f"  Rc {p['rc']:8.1f}: P{p['period']} levels {np.round(p['levels'], 3).tolist()} rms {p['rms']:.3f}"
            )
    print(f"board C levels {np.round(lev_c, 3).tolist()}")

    # state C against the two period-4 orbits: contact Rc (P4 window) and A = A_WIN (Rc = 0)
    p4 = np.nonzero(ok & (per == 4) & (rc > 10) & (rc < 200))[0]
    w = window_ref()
    w_per = classify(w["mi"], w["mh"], N_IC)
    w_idx = np.nonzero(~w["bad"] & (w_per == 4))[0]
    comp = {
        f"contact Rc {rc[p4].min():.0f}-{rc[p4].max():.0f} ohm, A = {sc.A_EFF:.3f}": summarise_p4(
            d["mi"], d["mh"], p4, d["rms"], d["near"], d["mean_x"]
        ),
        f"Rc = 0, A = {A_WIN}": summarise_p4(w["mi"], w["mh"], w_idx, w["rms"], w["near"], w["mean_x"]),
        "board C": dict(
            levels=lev_c, rms=board["C"]["rms"], near=board["C"]["near"], i_uA=board["C"]["i_mean_uA"]
        ),
    }
    print(f"A = {A_WIN}, Rc = 0: periods {w_per.tolist()}")
    for k, v in comp.items():
        print(
            f"  {k:38s} levels {np.round(v['levels'], 3).tolist()}  rms {v['rms']:.3f}  near {v['near']:.3f}  "
            f"I {v['i_uA']:.1f} uA"
        )

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(7.0, 7.4), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    xr = X_PLOT
    good = [row for row in rows[1:] if row["n_ok"] > 0]

    pos = rc[d["mi"]] > 0
    keep = ok[d["mi"]] & pos & (d["mh"] > 0.2)
    ax_a.plot(
        rc[d["mi"]][keep],
        d["mh"][keep],
        ".",
        ms=0.8,
        mew=0,
        color=sc.C_SIM,
        alpha=0.4,
        rasterized=True,
    )
    ax_a.hlines(lev_c, 50, 110, color="grey", ls="--", lw=0.8)  # board C's four x' maxima levels
    ax_a.set_ylim(0.2, 1.2)
    ax_a.set_ylabel("x′ maxima (V)")

    rr = np.array([row["rc"] for row in good])
    for axx, key, ylab in (
        (ax_b, "rms", "rms x (V)"),
        (ax_c, "near", "Time within 0.2 V"),
        (ax_d, "i_uA", "Mean diode current (µA)"),
    ):
        jj = (rc > 0) & ok
        axx.plot(
            rc[jj],
            {"rms": d["rms"], "near": d["near"], "i_uA": i_mean}[key][jj],
            "o",
            ms=1.6,
            mew=0,
            color=sc.C_SIM,
            alpha=0.5,
        )
        axx.plot(rr, [row[key] for row in good], "-", color=sc.C_SIM, lw=0.9)
        bkey = {"rms": "rms", "near": "near", "i_uA": "i_mean_uA"}[key]
        for k, b in board.items():
            axx.axhline(b[bkey], color="grey", ls="--", lw=0.7)
            axx.annotate(
                k,
                (1.0, b[bkey]),
                xycoords=("axes fraction", "data"),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color="grey",
            )
        axx.set_ylabel(ylab)

    jj = (rc > 0) & ok
    ax_e.plot(rc[jj], d["lam"][jj], "o", ms=1.6, mew=0, color=sc.C_SIM, alpha=0.5)
    ax_e.plot(rr, [row["lam"] for row in good], "-", color=sc.C_SIM, lw=0.9)
    ax_b.set_ylim(0.35, 0.8)
    ax_d.set_ylim(0, 45)
    ax_e.axhline(
        lam_b["board_calibrated_mean"],
        color="grey",
        ls="--",
        lw=0.7,
    )
    ax_e.annotate(
        "B",
        (1.0, lam_b["board_calibrated_mean"]),
        xycoords=("axes fraction", "data"),
        xytext=(3, 0),
        textcoords="offset points",
        va="center",
        fontsize=7,
        color="grey",
    )
    ax_e.axhline(0, color="k", lw=0.5)
    ax_e.set_ylabel(r"λ$_1$ (1/τ)")
    ax_e.axhline(lam_b["lambda1_model"], color="k", ls=":", lw=0.8)  # Rc = 0, full spectrum

    for axx in (ax_a, ax_b, ax_c, ax_d, ax_e):
        axx.set_xscale("log")
        axx.set_xlim(*xr)
        axx.set_xlabel("Contact resistance Rc (Ω)")
    for axx, lab in zip((ax_a, ax_b, ax_c, ax_d, ax_e), "abcde"):
        sc.panel(axx, lab)
    sc.save(fig, "fig_contact_scan", png=True)

    sc.save_results(
        "contact_scan",
        dict(
            note="model 'simple' + ohmic Rc in series with the diode, A = 0.982, Rc scanned (not fitted)",
            n_ic=N_IC,
            t_trans=T_TRANS,
            t_rec=T_REC,
            rows=rows,
            periodic=per_rows,
            state_C_comparison=comp,
            board={
                k: {kk: vv for kk, vv in b.items() if kk != "maxima"}
                for k, b in board.items()
            },
            board_C_levels=lev_c,
            diverged=int(d["bad"].sum()),
        ),
    )


if __name__ == "__main__":
    main()
