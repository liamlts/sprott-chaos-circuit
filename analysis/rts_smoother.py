"""Model-free Kalman (RTS) smoother for the scope channels.

State: x and its first n-1 time derivatives (time in tau). The last one is driven by white noise of intensity q,
so the prior only says "x is smooth"; it knows nothing about the circuit, and x'', x''' from it can be used to test
the circuit equation. Each scope channel measures one derivative (with a sign) plus white noise of known rms
(from sc.hf_noise, which includes the 8-bit steps). q is set by maximum likelihood.

Every window has the same length and noise, so covariances and gains are the same for all windows: they are
computed once and the states of all windows are updated together.

Channels (CHANNELS): "x" = Ch1 (U2 pin 1), "xd" = Ch2 (U2 pin 7, drives D1), "xdd" = U1 pin 7, "mxd" = U1 pin 1.
A derivative channel is used only after calibrate() has measured its gain and offset against dx/dt from Ch1.
"""

from math import factorial

import numpy as np
from scipy.optimize import minimize_scalar

CHANNELS = {
    "x": (0, 1.0),
    "xd": (1, 1.0),
    "xdd": (2, 1.0),
    "mxd": (1, -1.0),
}  # (derivative, sign)
P0 = 1e4  # prior variance of every state at the first sample (effectively unknown)
SKIP = 20  # samples left out of the likelihood while the filter forgets the prior
EDGE = 50  # samples trimmed at both ends when calibrating


def transition(n, h, q):
    """F and Q for n states over one step h, white noise of intensity q on the n-th derivative."""
    F, Q = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j >= i:
                F[i, j] = h ** (j - i) / factorial(j - i)
            p = 2 * n - 1 - i - j
            Q[i, j] = q * h**p / (p * factorial(n - 1 - i) * factorial(n - 1 - j))
    return F, Q


def _obs(chans, n):
    H = np.zeros((len(chans), n))
    for r, c in enumerate(chans):
        d, s = CHANNELS[c]
        H[r, d] = s
    return H


def _covariances(F, Q, H, R, npts):
    n = F.shape[0]
    eye = np.eye(n)
    P = eye * P0
    Pp, Pf = np.empty((npts, n, n)), np.empty((npts, n, n))
    K = np.empty((npts, n, H.shape[0]))
    Sinv, logdet = np.empty((npts, H.shape[0], H.shape[0])), np.empty(npts)
    for k in range(npts):
        if k:
            P = F @ P @ F.T + Q
        Pp[k] = P
        S = H @ P @ H.T + R
        Si = np.linalg.inv(S)
        Kk = P @ H.T @ Si
        A = eye - Kk @ H
        P = A @ P @ A.T + Kk @ R @ Kk.T  # Joseph form, stays symmetric
        Pf[k], K[k], Sinv[k], logdet[k] = P, Kk, Si, np.linalg.slogdet(S)[1]
    return Pp, Pf, K, Sinv, logdet


def _forward(Y, F, H, R, cov):
    Pp, Pf, K, Sinv, logdet = cov
    nwin, npts, _ = Y.shape
    n = F.shape[0]
    xp, xf = np.zeros((nwin, npts, n)), np.zeros((nwin, npts, n))
    ll = 0.0
    for k in range(npts):
        if k:
            xp[:, k] = xf[:, k - 1] @ F.T
        e = Y[:, k] - xp[:, k] @ H.T
        xf[:, k] = xp[:, k] + e @ K[k].T
        if k >= SKIP:
            ll -= 0.5 * (nwin * logdet[k] + np.einsum("wi,ij,wj->", e, Sinv[k], e))
    return xp, xf, ll / (nwin * (npts - SKIP))


def _setup(ys, chans, sigmas, h, q, order):
    Y = np.stack(ys, axis=-1)
    F, Q = transition(order, h, q)
    H = _obs(chans, order)
    R = np.diag(np.asarray(sigmas, float) ** 2)
    return Y, F, H, R, _covariances(F, Q, H, R, Y.shape[1])


def loglik(ys, chans, sigmas, h, q, order=4):
    """Log-likelihood per window and sample of the data under the smoothness prior with intensity q."""
    Y, F, H, R, cov = _setup(ys, chans, sigmas, h, q, order)
    return _forward(Y, F, H, R, cov)[2]


def smooth(ys, chans, sigmas, h, q, order=4):
    """ys: list of (n_win, npts) arrays in volts, one per channel name in `chans`, already calibrated.

    Returns (states, sd, loglik): states (n_win, npts, order) = x, x', x'', ... in V and V/tau^k;
    sd (npts, order) their posterior standard deviation; loglik per window and sample.
    """
    Y, F, H, R, cov = _setup(ys, chans, sigmas, h, q, order)
    Pp, Pf = cov[0], cov[1]
    xp, xf, ll = _forward(Y, F, H, R, cov)
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(Y.shape[1] - 2, -1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[:, k] = xf[:, k] + (xs[:, k + 1] - xp[:, k + 1]) @ C.T
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    sd = np.sqrt(np.einsum("kii->ki", Ps))
    return xs, sd, ll


def fit_q(ys, chans, sigmas, h, order=4, n_fit=40, bounds=(-4.0, 10.0)):
    """Maximum-likelihood q (on the first n_fit windows). Returns q and the log-likelihood per sample."""
    sub = [y[:n_fit] for y in ys]
    res = minimize_scalar(
        lambda lq: -loglik(sub, chans, sigmas, h, 10**lq, order),
        bounds=bounds,
        method="bounded",
        options={"xatol": 0.02},
    )
    if min(abs(res.x - b) for b in bounds) < 0.1:
        print(f"  warning: ML q at the edge of its range (log10 q = {res.x:.2f})")
    return 10**res.x, -res.fun


def calibrate(y, chan, states, sd, noise, bins=None):
    """Gain g and offset b of a derivative channel against the Ch1-only estimate: y = s*g*x^(d) + b.

    Also returns the rms left over, the rms expected if the channel were only g*x^(d) + b + noise
    (noise plus the Ch1-only estimate's own error), and the mean left over binned against x^(d):
    a curve here means the channel is not a scaled derivative of Ch1 (e.g. it sags when the diode conducts).
    """
    d, s = CHANNELS[chan]
    ref = s * states[:, EDGE:-EDGE, d].ravel()
    yy = y[:, EDGE:-EDGE].ravel()
    g, b = np.polyfit(ref, yy, 1)
    left = yy - g * ref - b
    expect = np.sqrt(noise**2 + np.mean((g * sd[EDGE:-EDGE, d]) ** 2))
    out: dict = dict(
        gain=float(g),
        offset_V=float(b),
        left_rms_V=float(left.std()),
        expected_rms_V=float(expect),
    )
    if bins is not None:
        idx = np.digitize(ref, bins) - 1
        c = 0.5 * (bins[1:] + bins[:-1])
        m = np.array(
            [
                left[idx == k].mean() if np.sum(idx == k) > 200 else np.nan
                for k in range(c.size)
            ]
        )
        out["binned"] = (c, m)
    return out


def run(chan_data, sigmas, h, order=4, n_fit=40, verbose=True, bins=None):
    """Full pipeline. Ch1 alone first; then the channels are calibrated in order of derivative, each against the
    best estimate so far (Ch2 against Ch1 alone, x'' against Ch1 + the first-derivative channels), and the
    smoother is rerun with every calibrated channel.

    chan_data: dict name -> (n_win, npts) volts, must contain "x". sigmas: dict name -> rms noise (V).
    Returns dict with "x_only" and "all" results (states, sd, q, loglik) and the calibrations.
    """
    out: dict = {}
    chans, ys, sig, cal = ["x"], [chan_data["x"]], [sigmas["x"]], {}
    q, ll = fit_q(ys, chans, sig, h, order, n_fit)
    st, sd, _ = smooth(ys, chans, sig, h, q, order)
    out["x_only"] = dict(states=st, sd=sd, q=q, loglik=ll)
    if verbose:
        print(f"  Ch1 only: q = {q:.3g}, loglik {ll:.3f}")
    for deriv in (1, 2):
        new = [c for c in chan_data if c != "x" and CHANNELS[c][0] == deriv]
        if not new:
            continue
        for c in new:
            y = chan_data[c]
            cal[c] = calibrate(y, c, st, sd, sigmas[c], bins)
            g, b = cal[c]["gain"], cal[c]["offset_V"]
            if verbose:
                print(
                    f"  {c}: gain {g:.4f}, offset {b * 1e3:+.1f} mV, left over {cal[c]['left_rms_V'] * 1e3:.1f} mV "
                    f"rms (expected {cal[c]['expected_rms_V'] * 1e3:.1f} mV if it is only gain, offset and noise)"
                )
            chans.append(c)
            ys.append((y - b) / g)
            sig.append(sigmas[c] / abs(g))
        q, ll = fit_q(ys, chans, sig, h, order, n_fit)
        st, sd, _ = smooth(ys, chans, sig, h, q, order)
        out["all"] = dict(states=st, sd=sd, q=q, loglik=ll, chans=list(chans))
        if verbose:
            print(f"  {'+'.join(chans)}: q = {q:.3g}")
    out["cal"] = cal
    return out
