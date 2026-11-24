"""Quick check: Sprott 2011 jerk circuit with a 1N4148 diode stays chaotic.

x''' = -x'' - x - f(x'),  f(y) = R * Is * (exp(y / (n*VT)) - 1),  time in units of RC.
Compares R = 1 kOhm (paper values) and 10 kOhm (same RC with 100 nF).
"""

import numpy as np
from scipy.integrate import solve_ivp

IS, N, VT = 2.52e-9, 1.752, 0.02585  # 1N4148 SPICE values


def rhs(t, s, R):
    x, y, z = s
    f = R * IS * np.expm1(min(y / (N * VT), 60.0))
    return [y, z, -z - x - f]


def lyap(R, T=4000.0, dt=1.0, d0=1e-7):
    a = np.array([0.0, 0.4, 0.0])
    a = solve_ivp(rhs, (0, 200), a, args=(R,), method="LSODA", rtol=1e-9, atol=1e-11).y[:, -1]
    b = a + [d0, 0, 0]
    s = 0.0
    for _ in range(int(T / dt)):
        a = solve_ivp(rhs, (0, dt), a, args=(R,), method="LSODA", rtol=1e-9, atol=1e-11).y[:, -1]
        b = solve_ivp(rhs, (0, dt), b, args=(R,), method="LSODA", rtol=1e-9, atol=1e-11).y[:, -1]
        d = np.linalg.norm(b - a)
        s += np.log(d / d0)
        b = a + (b - a) * d0 / d
    return s / T, a


for R in (1e3, 1e4):
    lam, _ = lyap(R)
    sol = solve_ivp(rhs, (0, 600), [0, 0.4, 0], args=(R,), method="LSODA", rtol=1e-9, atol=1e-11, max_step=0.02)
    x, y, z = sol.y[:, sol.t > 200]
    print(f"R={R:>6.0f}  lambda={lam:.3f} per RC   x [{x.min():.2f},{x.max():.2f}]  "
          f"x' [{y.min():.2f},{y.max():.2f}]  x'' [{z.min():.2f},{z.max():.2f}] V")
