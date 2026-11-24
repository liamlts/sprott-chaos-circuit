"""Expected readings for the build guide (docs/build.md), from the model at the measured parts and from run B.

- model: one long run of the 'simple' model (1N4148, A = 0.982, R = 9.88 kOhm), 200 tau transient then 3000 tau.
  Ranges are the 0.1 and 99.9 % points, in volts, for x (U2 pin 1), x' (U2 pin 7) and x'' (U1 pin 7).
- board_B: run B (D1 reseated, 18:10), the run that matches the model; same percentiles of Ch1 (x) and Ch2 (x').
- peak frequencies of x from B_attractor.json (fig_B_attractor.py).
- r4_sweep: what the model does at standard R4 values (model_scan.json, fig_model_scan.py).
- periodic_window_R4_ohm: R4 range of the periodic window just above the board's R4 (A = 0.95-0.96 in
  model_scan.json near_board, every start periodic, periods 4-6; state C's period-4 orbit sits in it).

Writes analysis/results/build_numbers.json.
"""

import json

import numpy as np
from scipy.integrate import solve_ivp

import sprott_common as sc

T_SKIP, T_RUN, DT = 200.0, 3000.0, 0.01  # tau
PCT = (0.1, 99.9)
RES = sc.HERE / "results"


def lo_hi(v):
    lo, hi = np.percentile(v, PCT)
    return round(float(lo), 2), round(float(hi), 2)


def model_ranges():
    rhs, _ = sc.make_rhs("simple")
    kw = dict(method="LSODA", rtol=1e-8, atol=1e-10)
    s0 = solve_ivp(rhs, (0, T_SKIP), [0.0, 0.4, 0.0], **kw).y[:, -1]
    t = np.arange(0, T_RUN, DT)
    x, xd, xdd = solve_ivp(rhs, (0, T_RUN), s0, t_eval=t, max_step=0.05, **kw).y
    out = {}
    for name, v in (("x", x), ("xd", xd), ("xdd", xdd)):
        out[f"{name}_lo"], out[f"{name}_hi"] = lo_hi(v)
    return out


def board_ranges():
    r = sc.load(sc.B_RUN)
    out = {}
    for name in ("x", "xd"):
        out[f"{name}_lo"], out[f"{name}_hi"] = lo_hi(r[name])
    return out


def r4_of(a):
    return sc.TAU / (a * sc.C1)


def main():
    scan = json.loads((RES / "model_scan.json").read_text())
    attr = json.loads((RES / "B_attractor.json").read_text())
    win = [e["A"] for e in scan["near_board"] if 0.95 < e["A"] < 0.96 and "chaos" not in e["model"]]
    d = {
        "tau_ms": round(sc.TAU * 1e3, 3),
        "A": round(sc.A_EFF, 4),
        "model": {**model_ranges(), "peak_f_Hz": round(attr["peak_f_model"], 1)},
        "board_B": {**board_ranges(), "peak_f_Hz": round(attr["peak_f_board"], 1)},
        "r4_sweep": [
            {"R4_ohm": e["R4"], "A": round(e["A"], 3), "model": e["model"]}
            for e in scan["sweep"]
        ],
        "periodic_window_R4_ohm": [round(r4_of(max(win))), round(r4_of(min(win)))],
    }
    sc.save_results("build_numbers", d)
    print(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
