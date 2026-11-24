"""Control a Tektronix scope over USB and record the Sprott circuit.

Written for the TBS1052B-EDU. Also runs the 4-channel scopes that use the same commands (TBS1000, TBS1000B,
TDS1000, TDS2000 families); with any other model it stops before recording.

Usage
  python3 sprott_scope.py list                         # find the scope, print its ID
  python3 sprott_scope.py view                         # XY mode (Ch1 against Ch2) + infinite persistence
  python3 sprott_scope.py capture --n 200 --r4 10000   # 200 single-shot captures of Ch1 and Ch2, saved to data/
  python3 sprott_scope.py capture --n 200 --channels x,xd,xdd,mxd   # all four channels
  python3 sprott_scope.py capture --n 50 --fake        # no scope: simulated captures, to test the pipeline

--channels says what each scope channel is on, in channel order (Ch1 first). The default is x,xd.
  x    x      U2 pin 1 (hole b21)
  xd   x'     U2 pin 7 (hole j22), the diode's anode
  xdd  x''    U1 pin 7
  mxd  -x'    U1 pin 1
x and xd must be in the list (the analysis reads them). The scope triggers on Ch1.

Each capture is one single-shot acquisition (2,500 points per channel, every channel from the same trigger).
Reading one channel takes ~1.2 s, so a capture takes ~2.75 s with two channels and ~5 s with four.
A run is saved as data/sprott_<label>_<time>.h5 with one dataset per channel, named as above (volts,
captures x points), t (seconds), capture_time_s, and attributes holding the part values, scope ID and settings.
Plug the scope's rear USB port (device port) into the laptop.
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
TEK_VENDOR = (
    "0x0699",
    "::1689::",
)  # Tektronix USB vendor ID, hex (NI-VISA) or decimal (pyvisa-py)

# default scope settings for the 100 nF build (about 100 Hz)
TDIV = 10e-3  # s/div -> 100 ms per capture, about 10 loops, 25 kS/s
VDIV = 0.5  # V/div on both channels (x spans about -1.5..+0.6 V, x' about -0.9..+0.6 V)
NPTS = 2500

# what a scope channel can be on (see the docstring)
SIGNALS = {
    "x": "x, U2 pin 1 (b21)",
    "xd": "x', U2 pin 7 (j22)",
    "xdd": "x'', U1 pin 7",
    "mxd": "-x', U1 pin 1",
}
# models with the same programming commands as the TBS1052B; the last digit of the number is the channel count
KNOWN_MODELS = re.compile(r"\b(TBS1\d{3}|TDS[12]\d{3})")

# part values on the board (measure them with the DVM and pass the real values on the command line)
R_OHM, C_F, R4_OHM = 10_000.0, 100e-9, 10_000.0


# ---------------------------------------------------------------- scope
def is_tek(resource):
    return any(v in resource.lower() for v in TEK_VENDOR)


def open_scope(resource=None):
    import pyvisa

    rm = pyvisa.ResourceManager("@py")
    if resource is None:
        tek = [r for r in rm.list_resources() if is_tek(r)]
        if not tek:
            sys.exit(
                "No Tektronix scope found on USB. Check the cable is in the rear port and the scope is on."
            )
        resource = tek[0]
    usb_reset()
    scope = rm.open_resource(resource)
    scope.timeout = 10_000
    scope.write("HEADer OFF")
    return scope, resource


def usb_reset():
    """Reset the scope's USB link. A transfer cut off mid-way (Ctrl-C, crash) leaves the TBS
    refusing every later write with 'Operation timed out'; pyvisa-py cannot send a device
    clear, so a USB port reset is the way to get it back."""
    import usb.core

    dev = usb.core.find(idVendor=0x0699)
    if dev is not None:
        dev.reset()
        time.sleep(2.0)


def parse_channels(text):
    names = [c.strip() for c in text.split(",") if c.strip()]
    ok = (
        2 <= len(names) <= 4
        and len(set(names)) == len(names)
        and all(c in SIGNALS for c in names)
        and {"x", "xd"} <= set(names)
    )
    if not ok:
        sys.exit(
            f"--channels {text!r}: give 2 to 4 different names from {', '.join(SIGNALS)}, including x and xd."
        )
    return names


def check_model(idn, n_used):
    """Stop unless the scope uses the TBS1052B's commands and has enough channels."""
    m = KNOWN_MODELS.search(idn.upper().replace(" ", ""))
    if m is None:
        sys.exit(
            f"The scope reports '{idn}'. This script only knows the TBS1000/TDS2000 commands, so it stops here "
            "rather than record wrong data. It needs adapting to this model first."
        )
    n_scope = int(m.group(1)[-1])
    if n_used > n_scope:
        sys.exit(f"{m.group(1)} has {n_scope} channels but --channels lists {n_used}.")


def cmd_list(args):
    import pyvisa

    rm = pyvisa.ResourceManager("@py")
    res = rm.list_resources()
    print("VISA resources:", res or "none")
    for r in res:
        if is_tek(r):
            s = rm.open_resource(r)
            s.timeout = 5000
            print(r, "->", s.query("*IDN?").strip())


def setup_channels(scope, vdiv, tdiv, chans=("CH1", "CH2")):
    for ch in chans:
        scope.write(f"SELect:{ch} ON")
        scope.write(f"{ch}:COUPling DC")
        scope.write(f"{ch}:SCAle {vdiv}")
        scope.write(f"{ch}:POSition 0")
    scope.write(f"HORizontal:MAIn:SCAle {tdiv}")
    scope.write(
        "ACQuire:MODe SAMple"
    )  # never Average: a chaotic signal does not repeat
    scope.write("TRIGger:MAIn:MODe AUTO")
    scope.write("TRIGger:MAIn:EDGE:SOUrce CH1")
    scope.write("TRIGger:MAIn:LEVel 0")
    # waveform transfer format: set once, it persists (each command costs the TBS ~0.4 s)
    for c in (
        "DATa:ENCdg RIBinary",
        "DATa:WIDth 1",
        "DATa:STARt 1",
        f"DATa:STOP {NPTS}",
    ):
        scope.write(c)
    scope.query("*OPC?")


def channel_scale(scope, ch):
    """Volts per code for one channel; fixed for the whole run since V/div does not change."""
    scope.write(f"DATa:SOUrce {ch}")
    q = [
        float(scope.query(f"WFMPre:{k}?")) for k in ("YMUlt", "YZEro", "YOFf", "XINcr")
    ]
    return dict(zip(("ymult", "yzero", "yoff", "xincr"), q))


def cmd_view(args):
    scope, res = open_scope(args.resource)
    setup_channels(scope, args.vdiv, args.tdiv)
    scope.write("ACQuire:STOPAfter RUNSTop")
    scope.write("DISplay:FORMat XY")
    scope.write("DISplay:PERSistence INFInite")
    scope.write("ACQuire:STATE RUN")
    print(
        "Scope in XY mode with infinite persistence. Press the scope's DISPLAY menu to clear or turn it off."
    )


def read_channel(scope, ch, sc):
    """One channel of the last acquisition; sc from channel_scale. CURVe? takes ~1.2 s on the TBS."""
    scope.write(f"DATa:SOUrce {ch}")
    raw = scope.query_binary_values(
        "CURVe?", datatype="b", is_big_endian=True, container=np.array
    )
    clipped = bool(np.any(raw >= 127) or np.any(raw <= -127))
    return (raw - sc["yoff"]) * sc["ymult"] + sc["yzero"], sc["xincr"], clipped


def single_shot(scope, timeout=5.0):
    scope.write("ACQuire:STOPAfter SEQuence")
    scope.write("ACQuire:STATE RUN")
    t0 = time.time()
    while scope.query("ACQuire:STATE?").strip() not in ("0", "OFF"):
        if time.time() - t0 > timeout:
            raise TimeoutError("scope did not finish a single acquisition")
        time.sleep(0.02)


# ---------------------------------------------------------------- fake captures (no scope)
def fake_captures(n, tdiv, vdiv, r_ohm, c_f, r4_ohm, seed=0):
    """Simulated captures of every signal in SIGNALS, with the scope's 8-bit steps and some noise."""
    from scipy.integrate import solve_ivp

    IS, NID, VT = 2.52e-9, 1.752, 0.02585
    A = r_ohm / r4_ohm

    def rhs(_, s):
        x, y, z = s
        return [y, z, -A * z - x - r_ohm * IS * np.expm1(min(y / (NID * VT), 60.0))]

    rc = r_ohm * c_f
    dt = 10 * tdiv / NPTS / rc  # sample step in units of RC
    span = NPTS * dt
    gap = 7.3  # dead time between captures (RC units), like the USB transfer time
    sol = solve_ivp(
        rhs,
        (0, 200 + n * (span + gap)),
        [0.0, 0.4, 0.0],
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )
    rng = np.random.default_rng(seed)
    lsb = vdiv / 25.0
    out = {k: np.empty((n, NPTS)) for k in SIGNALS}
    for k in range(n):
        tt = 200 + k * (span + gap) + np.arange(NPTS) * dt
        x, y, z = sol.sol(tt)
        for name, v in (("x", x), ("xd", y), ("xdd", z), ("mxd", -y)):
            out[name][k] = np.round((v + rng.normal(0, 0.005, NPTS)) / lsb) * lsb
    return out, dt * rc


# ---------------------------------------------------------------- capture run
def cmd_capture(args):
    DATA_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    names = parse_channels(args.channels)
    chans = [f"CH{i + 1}" for i in range(len(names))]
    label = args.label or f"R4_{int(args.r4)}"
    path = DATA_DIR / f"sprott_{label}_{stamp}{'_fake' if args.fake else ''}.h5"
    meta: dict[str, object] = dict(
        R_ohm=args.r,
        C_F=args.c,
        R4_ohm=args.r4,
        A=args.r / args.r4,
        tdiv=args.tdiv,
        vdiv=args.vdiv,
        note=args.note,
        started=datetime.now().isoformat(timespec="seconds"),
        fake=args.fake,
        channels=",".join(names),
    )

    if args.fake:
        sim, dt = fake_captures(args.n, args.tdiv, args.vdiv, args.r, args.c, args.r4)
        data = {nm: sim[nm] for nm in names}
        meta["scope"] = "simulated"
        tcap = np.arange(args.n) * 0.3
        nclip = 0
    else:
        scope, res = open_scope(args.resource)
        idn = scope.query("*IDN?").strip()
        check_model(idn, len(names))
        meta["scope"] = idn
        meta["resource"] = res
        for ch in chans:
            meta[f"{ch}_probe"] = scope.query(f"{ch}:PRObe?").strip()
        scope.write("DISplay:FORMat YT")
        setup_channels(scope, args.vdiv, args.tdiv, chans)
        time.sleep(0.5)
        single_shot(
            scope
        )  # one throw-away shot so the waveform preamble matches these settings
        scales = {ch: channel_scale(scope, ch) for ch in chans}
        data = {nm: np.empty((args.n, NPTS)) for nm in names}
        tcap = np.empty(args.n)
        nclip, dt, t0, k = 0, 0.0, time.time(), 0
        try:
            for k in range(args.n):
                single_shot(scope)
                tcap[k] = time.time() - t0
                clipped = False
                for ch, nm in zip(chans, names):
                    data[nm][k], dt, c = read_channel(scope, ch, scales[ch])
                    clipped = clipped or c
                nclip += clipped
                if (k + 1) % 20 == 0 or k + 1 == args.n:
                    print(f"  {k + 1}/{args.n} captures, {tcap[k]:.0f} s", flush=True)
        except KeyboardInterrupt:
            print(f"stopped by hand after {k} captures")
            data, tcap = {nm: v[:k] for nm, v in data.items()}, tcap[:k]
        finally:
            scope.write("ACQuire:STOPAfter RUNSTop")
            scope.write("ACQuire:STATE RUN")
        if nclip:
            print(
                f"WARNING: {nclip} captures hit the edge of the screen. Use a larger --vdiv."
            )

    meta["clipped_captures"] = int(nclip)
    with h5py.File(path, "w") as f:
        for nm, v in data.items():
            f[nm] = v
        f["t"] = np.arange(NPTS) * dt
        f["capture_time_s"] = tcap
        for k, v in meta.items():
            f.attrs[k] = v
    print(f"saved {len(tcap)} captures of {', '.join(names)} -> {path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("view", "capture"):
        s = sub.add_parser(name)
        s.add_argument(
            "--resource", help="VISA resource string (default: first Tektronix on USB)"
        )
        s.add_argument("--tdiv", type=float, default=TDIV, help="seconds per division")
        s.add_argument(
            "--vdiv", type=float, default=VDIV, help="volts per division, both channels"
        )
        if name == "capture":
            s.add_argument(
                "--n", type=int, default=200, help="number of single-shot captures"
            )
            s.add_argument(
                "--r", type=float, default=R_OHM, help="measured R (ohm), R1-R3, R5, R6"
            )
            s.add_argument(
                "--c", type=float, default=C_F, help="measured C1-C3 (farad)"
            )
            s.add_argument("--r4", type=float, default=R4_OHM, help="measured R4 (ohm)")
            s.add_argument("--label", help="name for the run (default R4_<value>)")
            s.add_argument("--note", default="", help="free text saved with the run")
            s.add_argument(
                "--channels",
                default="x,xd",
                help="what Ch1, Ch2, ... are on, from: x, xd, xdd, mxd (default x,xd)",
            )
            s.add_argument(
                "--fake",
                action="store_true",
                help="simulate instead of using the scope",
            )
    args = p.parse_args()
    {"list": cmd_list, "view": cmd_view, "capture": cmd_capture}[args.cmd](args)


if __name__ == "__main__":
    main()
