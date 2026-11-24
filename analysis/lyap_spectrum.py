"""Full Lyapunov spectrum of the model at the board's A (no change to the model), for the run-B chaos figures.

Model 'simple' as everywhere: x''' = -A x'' - x - R Is expm1(x' / (n VT)), time in tau, A = 0.982.
Method: RK4 for the state together with the tangent equations dQ/dt = J Q, re-orthonormalised by QR every
RENORM tau (Benettin et al. 1980); lambda_i = time average of ln |R_ii|. N_TRAJ independent trajectories from
random starts give the error bar. Check: lambda_1 + lambda_2 + lambda_3 must equal the trace of J, which is -A.
Kaplan-Yorke dimension D_KY = 2 + lambda_1 / |lambda_3|.
Output: results/lyap_spectrum.json (also read by fig_B_lyapunov.py and fig_B_dimension.py).
"""

import time

import numpy as np

import sprott_common as sc

A = sc.A_EFF
RIS = sc.R_MEAN * sc.IS
NVT = sc.NID * sc.VT
DT = 0.004
T_TRANS = 300.0
T_RUN = 3000.0
RENORM = 0.5  # tau between QR steps
N_TRAJ = 24
SEED = 7


def f(s):
    x, y, z = s
    return np.array([y, z, -A * z - x - RIS * np.expm1(np.minimum(y / NVT, 60.0))])


def jq(s, q):
    """J(s) Q for every trajectory; q[i, k, n] = component i of tangent vector k, trajectory n."""
    c = RIS / NVT * np.exp(np.minimum(s[1] / NVT, 60.0))
    return np.stack([q[1], q[2], -q[0] - c * q[1] - A * q[2]])


def step(s, q):
    k1s, k1q = f(s), jq(s, q)
    s2, q2 = s + 0.5 * DT * k1s, q + 0.5 * DT * k1q
    k2s, k2q = f(s2), jq(s2, q2)
    s3, q3 = s + 0.5 * DT * k2s, q + 0.5 * DT * k2q
    k3s, k3q = f(s3), jq(s3, q3)
    s4, q4 = s + DT * k3s, q + DT * k3q
    k4s, k4q = f(s4), jq(s4, q4)
    return (
        s + DT / 6 * (k1s + 2 * k2s + 2 * k3s + k4s),
        q + DT / 6 * (k1q + 2 * k2q + 2 * k3q + k4q),
    )


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    s = np.vstack(
        [
            rng.uniform(-1, 0.5, N_TRAJ),
            rng.uniform(-0.5, 0.5, N_TRAJ),
            rng.uniform(-0.5, 0.5, N_TRAJ),
        ]
    )
    q = np.repeat(np.eye(3)[:, :, None], N_TRAJ, axis=2)
    for _ in range(int(T_TRANS / DT)):  # transient, state only
        k1 = f(s)
        k2 = f(s + 0.5 * DT * k1)
        k3 = f(s + 0.5 * DT * k2)
        k4 = f(s + DT * k3)
        s = s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    print(f"transient done ({time.time() - t0:.0f} s)", flush=True)

    n_sub = int(round(RENORM / DT))
    n_ren = int(T_RUN / RENORM)
    acc = np.zeros((N_TRAJ, 3))
    for k in range(n_ren):
        for _ in range(n_sub):
            s, q = step(s, q)
        qq, rr = np.linalg.qr(np.moveaxis(q, 2, 0))  # (N, 3, 3)
        acc += np.log(np.abs(np.diagonal(rr, axis1=1, axis2=2)))
        q = np.moveaxis(qq, 0, 2)
        if (k + 1) % (n_ren // 10) == 0:
            lam = acc.mean(0) / ((k + 1) * RENORM)
            print(
                f"t = {(k + 1) * RENORM:6.0f} tau  lambda = {lam[0]:+.4f} {lam[1]:+.4f} {lam[2]:+.4f}"
                f"  ({time.time() - t0:.0f} s)",
                flush=True,
            )
    lam_each = acc / T_RUN  # (N, 3)
    lam = lam_each.mean(0)
    err = lam_each.std(0, ddof=1) / np.sqrt(N_TRAJ)
    dky_each = 2 + lam_each[:, 0] / np.abs(lam_each[:, 2])
    out = dict(
        A=A,
        lam=lam.tolist(),
        lam_err=err.tolist(),
        sum=float(lam.sum()),
        trace=-A,
        d_ky=float(dky_each.mean()),
        d_ky_err=float(dky_each.std(ddof=1) / np.sqrt(N_TRAJ)),
        lyapunov_time_ms=float(sc.TAU * 1e3 / lam[0]),
        n_traj=N_TRAJ,
        t_run_tau=T_RUN,
        dt=DT,
    )
    print(
        f"lambda = {lam[0]:+.4f} ± {err[0]:.4f}, {lam[1]:+.4f} ± {err[1]:.4f}, "
        f"{lam[2]:+.4f} ± {err[2]:.4f} per tau; sum {lam.sum():+.4f} (trace -A = {-A:+.4f})"
    )
    print(
        f"D_KY = {out['d_ky']:.4f} ± {out['d_ky_err']:.4f}; Lyapunov time {out['lyapunov_time_ms']:.2f} ms"
    )
    sc.save_results("lyap_spectrum", out)


if __name__ == "__main__":
    main()
