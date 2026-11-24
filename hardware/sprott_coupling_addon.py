"""Coupling add-on: two Sprott boards, the response (B) locked to the drive (A). See analysis/sync_check.py.

Coupling into integrator 2 of board B (node N2, U1 pin 2): one 10k from B's own -x' (a leak across C2) and one
10k from A's x'. Together they add -k (x'_B - x'_A) to board B's x'' equation, k = tau / (Rc C2) ~ 1.
On the build layout N2 has one usable hole (a9; b9 is under the orange jumper), so the two resistors sit on a
small unpowered strip, joined to the boards by three wires. Board A only drives a 10k load, so it is not changed.
Page: integrator-2 schematic with the new parts, simulated XY pictures, board B (existing parts faded) with the
three new connection points, the coupling strip, steps and expected numbers.
Writes sprott_coupling_addon.pdf and sprott_coupling_addon_preview.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from scipy.integrate import solve_ivp

import sprott_build as sb
from sprott_build import ROW_Y, H, RL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import sprott_common as sc  # noqa: E402
import sync_check as syn  # noqa: E402

OUT_PDF = "sprott_coupling_addon.pdf"
OUT_PREVIEW = "sprott_coupling_addon_preview.png"
RC_OHM = 10_000
C_N2 = "#8d5524"  # new wire: strip -> B's N2
# board B connection points, and A's
B_N2, B_MXD, B_PROBE = "a9", "j19", "b21"
A_XD, A_PROBE = "i22", "b21"
# coupling strip: hole spacing 1, columns 1-10, rows a-e (a at the bottom); placed right of board B
STRIP_X0, STRIP_Y0 = 35.0, 4.0
STRIP_COLS = 10


def S(hole):
    """Strip hole -> drawing coordinates."""
    return STRIP_X0 + int(hole[1:]) - 1, STRIP_Y0 + "abcde".index(hole[0])


# ---------------------------------------------------------------- simulated XY pictures
def xy_traces(rc, is_fac, t_rec=150.0):
    k = 0.0 if rc is None else sc.TAU / (rc * sc.C2)
    sol = solve_ivp(
        syn.rhs,
        (0, 300.0 + t_rec),
        [0.1, 0.0, 0.0, -0.3, 0.2, 0.1],
        args=(k, 0.0, is_fac, "N2"),
        t_eval=np.linspace(300.0, 300.0 + t_rec, 15000),
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    return sol.y[0], sol.y[3]


def xy_panel(ax, xa, xb, title, color):
    ax.plot(xa, xb, color=color, lw=0.4)
    ax.set_xlim(-2.0, 1.0)
    ax.set_ylim(-2.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_yticks([-2, -1, 0, 1])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Ch1: x of board A (V)", fontsize=8)
    ax.set_ylabel("Ch2: x of board B (V)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(color="#ddd", lw=0.5)


# ---------------------------------------------------------------- schematic of integrator 2 on board B
def schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")
    k = dict(color="#333", lw=1.4, solid_capstyle="round")

    def res(x0, y0, x1, y1, label, color="#333", above=True):
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.4)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.add_patch(
            Rectangle(
                (cx - 0.45, cy - 0.16),
                0.9,
                0.32,
                fc="white",
                ec=color,
                lw=1.4,
                zorder=3,
            )
        )
        ax.text(
            cx,
            cy + (0.32 if above else -0.32),
            label,
            ha="center",
            va="bottom" if above else "top",
            fontsize=8,
            color=color,
        )

    def dot(x, y, c="#333"):
        ax.add_patch(Circle((x, y), 0.07, color=c, zorder=4))

    # op-amp U1 half A (integrator 2), output on the right
    ax.add_patch(
        plt.Polygon(
            [[5.0, 1.0], [5.0, 3.0], [6.6, 2.0]],
            closed=True,
            fc="white",
            ec="#333",
            lw=1.4,
            zorder=2,
        )
    )
    ax.text(5.15, 2.45, "−", fontsize=11, va="center")
    ax.text(5.15, 1.5, "+", fontsize=9, va="center")
    ax.text(5.6, 2.0, "U1\npin 1–3", fontsize=6.5, va="center", ha="center")
    ax.plot([4.5, 5.0], [2.45, 2.45], **k)
    ax.plot([4.5, 4.5], [2.45, 5.2], **k)  # N2 node riser
    ax.plot([4.7, 5.0], [1.5, 1.5], **k)
    ax.plot([4.7, 4.7], [1.5, 1.2], **k)
    for i, w in enumerate((0.3, 0.2, 0.1)):
        ax.plot(
            [4.7 - w, 4.7 + w], [1.2 - 0.1 * i, 1.2 - 0.1 * i], color="#333", lw=1.2
        )
    ax.plot([6.6, 8.6], [2.0, 2.0], **k)
    ax.plot([7.6, 7.6], [2.0, 4.4], **k)
    dot(7.6, 2.0)
    ax.text(
        8.7,
        2.0,
        "−x′\n(U1 pin 1)",
        va="center",
        fontsize=8,
        color=sb.C_MXD,
        fontweight="bold",
    )
    # C2 feedback
    ax.plot([4.5, 5.85], [3.4, 3.4], **k)
    ax.plot([6.25, 7.6], [3.4, 3.4], **k)
    ax.plot([5.85, 5.85], [3.1, 3.7], color="#333", lw=2)
    ax.plot([6.25, 6.25], [3.1, 3.7], color="#333", lw=2)
    ax.text(6.05, 3.0, "C2", ha="center", va="top", fontsize=8)
    dot(4.5, 3.4)
    dot(7.6, 3.4)
    # new: Rc1 across C2 (leak), drawn above
    res(4.5, 4.4, 7.6, 4.4, "Rc1  10k  (new)", color=C_N2)
    ax.plot([4.5, 4.5], [3.4, 4.4], color="#333", lw=1.4)
    dot(4.5, 4.4)
    dot(7.6, 4.4)
    # inputs into N2
    res(0.6, 2.45, 4.5, 2.45, "R2  10k  (from x″)", above=False)
    dot(4.5, 2.45)
    res(0.6, 5.2, 4.5, 5.2, "Rc2  10k  (new)", color=C_N2)
    ax.plot([4.5, 4.5], [4.4, 5.2], color="#333", lw=1.4)
    ax.text(
        0.5,
        5.2,
        "x′ of board A",
        ha="right",
        va="center",
        fontsize=8,
        color=sb.C_XD,
        fontweight="bold",
    )
    ax.text(
        0.5,
        2.45,
        "x″",
        ha="right",
        va="center",
        fontsize=8,
        color=sb.C_XDD,
        fontweight="bold",
    )
    ax.text(
        4.35,
        2.75,
        "N2",
        ha="right",
        va="bottom",
        fontsize=8,
        color=C_N2,
        fontweight="bold",
    )
    ax.text(
        5.0,
        0.2,
        "Board B, integrator 2 (half A of chip U1). Board A is not changed.",
        ha="center",
        fontsize=7.5,
        color="#555",
    )
    # the riser from Rc1's left end to Rc2's right end is the same node; Rc2 ends on the N2 riser


# ---------------------------------------------------------------- board B, faded, plus the strip
def flying(ax, p0, p1, c0, c1, color, lw=2.8):
    verts = [p0, c0, c1, p1]
    path = MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(
        PathPatch(path, fc="none", ec="white", lw=lw + 2.4, zorder=13, capstyle="round")
    )
    ax.add_patch(
        PathPatch(path, fc="none", ec=color, lw=lw, zorder=13.5, capstyle="round")
    )
    for p in (p0, p1):
        ax.add_patch(Circle(p, 0.2, color=color, zorder=14))


def mark(ax, p, color):
    ax.add_patch(Circle(p, 0.48, fc="none", ec=color, lw=2.2, zorder=14))


def box(ax, xy, text, color, ha="center", fs=8):
    ax.text(
        *xy,
        text,
        ha=ha,
        va="center",
        fontsize=fs,
        color=color,
        zorder=15,
        bbox=dict(fc="white", ec=color, lw=1.0, boxstyle="round,pad=0.3"),
    )


def board_b(ax):
    sb.board(ax)
    for name, c0 in sb.CHIPS.items():
        sb.chip(ax, c0, f"chip {name}")
    for hole, rail, color in sb.POWER:
        sb.wire(ax, H(hole), RL(rail, sb.group(hole)[0]), color)
    for r1, r2, c in sb.BYPASS:
        sb.capacitor(ax, RL(r1, c), RL(r2, c), fc=sb.CERAMIC)
    for name, kind, h1, h2, val in sb.PARTS:
        if kind == "R":
            sb.resistor(ax, H(h1), H(h2), val)
        elif kind == "C":
            sb.capacitor(ax, H(h1), H(h2))
        else:
            sb.diode(ax, H(h1), H(h2))
    for h1, h2, color in sb.WIRES:
        sb.wire(ax, H(h1), H(h2), color)
    # fade everything that is already on the board
    ax.add_patch(
        FancyBboxPatch(
            (0.0, -2.3),
            sb.NCOL + 1.0,
            16.8,
            boxstyle="round,pad=0,rounding_size=0.4",
            fc="white",
            ec="none",
            alpha=0.55,
            zorder=10,
        )
    )
    ax.text(
        0.2,
        15.3,
        "Board B (response), built like board A. Faded = already there.",
        ha="left",
        fontsize=9,
        fontweight="bold",
    )


def strip(ax):
    x0, y0 = STRIP_X0, STRIP_Y0
    ax.add_patch(
        FancyBboxPatch(
            (x0 - 1.0, y0 - 1.2),
            STRIP_COLS + 1.0,
            6.4,
            boxstyle="round,pad=0,rounding_size=0.4",
            fc="#f7f5ef",
            ec="#c9c4b5",
            zorder=0,
        )
    )
    for i, r in enumerate("abcde"):
        for c in range(1, STRIP_COLS + 1):
            ax.add_patch(
                Rectangle(
                    (x0 + c - 1 - 0.1, y0 + i - 0.1),
                    0.2,
                    0.2,
                    fc="#3a3a3a",
                    ec="none",
                    zorder=1,
                )
            )
        ax.text(
            x0 - 0.55, y0 + i, r, ha="center", va="center", fontsize=6.5, color="#777"
        )
    for c in (1, 5, 10):
        ax.text(
            x0 + c - 1,
            y0 - 0.75,
            str(c),
            ha="center",
            va="center",
            fontsize=6,
            color="#777",
        )
    r2 = sb.resistor(ax, S("b2"), S("b5"), RC_OHM)
    r1 = sb.resistor(ax, S("d8"), S("d5"), RC_OHM)
    lab = dict(ha="center", va="center", fontsize=8, fontweight="bold", color=C_N2, zorder=12)
    ax.text(r2[0], r2[1] + 0.55, "Rc2", **lab)
    ax.text(r1[0], r1[1] - 0.55, "Rc1", **lab)
    ax.text(
        x0 + STRIP_COLS + 0.6,
        -5.7,
        "Coupling strip: any small breadboard, no power\nRc1, Rc2: 10,000 ohms (brown, black, orange)",
        ha="right",
        va="center",
        fontsize=8,
    )


def page():
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(
        0.03,
        0.965,
        "Syncing two Sprott boards: board B follows board A",
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.03,
        0.93,
        "Two 10k resistors and three wires. Board A is unchanged; only board B is steered.",
        fontsize=9.5,
        va="top",
        color="#444",
    )

    axs = fig.add_axes((0.02, 0.60, 0.40, 0.30))
    schematic(axs)

    xa, xb = xy_traces(RC_OHM, 2.0)
    ax1 = fig.add_axes((0.47, 0.62, 0.17, 0.25))
    xy_panel(ax1, xa, xb, "Coupled (10k), diodes differ", C_N2)
    xa, xb = xy_traces(None, 2.0)
    ax2 = fig.add_axes((0.67, 0.62, 0.17, 0.25))
    xy_panel(ax2, xa, xb, "Brown wire out", "#999")
    ax2.set_ylabel("")
    fig.text(
        0.865,
        0.86,
        "Simulated scope, XY mode.\nLocked: a thin diagonal\nline. The line's thickness\n"
        "shows how different the\nboards are (here diode\nIs differs by 2×).\nUnlocked: a filled blob.",
        fontsize=7.5,
        va="top",
        color="#444",
    )

    ax = fig.add_axes((0.01, 0.215, 0.98, 0.385))
    ax.set_xlim(-1.0, 47.5)
    ax.set_ylim(-6.6, 17.6)
    ax.set_aspect("equal")
    ax.axis("off")
    board_b(ax)
    strip(ax)
    # three new wires
    flying(ax, H(B_N2), S("a5"), (9.0, -9.0), (36.0, -9.0), C_N2)
    flying(ax, H(B_MXD), S("e8"), (19.0, 17.5), (48.0, 12.0), sb.C_MXD)
    a_end = (S("e2")[0], 13.4)
    flying(ax, a_end, S("e2"), (a_end[0], 11.0), (S("e2")[0], 10.0), sb.C_XD)
    box(ax, (a_end[0] + 4.6, 14.3), f"from board A, hole {A_XD} (x′)", sb.C_XD)
    mark(ax, H(B_N2), C_N2)
    mark(ax, H(B_MXD), sb.C_MXD)
    sb.probe(
        ax,
        H(B_PROBE),
        "Channel 2 probe tip: x of board B\n(Ch1 tip on board A, b21)",
        (24.6, -3.5),
        sb.C_X,
    )
    box(ax, (4.2, -3.6), f"brown wire: {B_N2} (N2, U1 pin 2)\nto strip a5", C_N2)
    box(ax, (12.5, 17.0), f"orange wire: {B_MXD} (−x′) to strip e8", sb.C_MXD)

    steps = (
        "1. Build board B from the same sheet as board A. Check each board alone on the scope (x vs x′) before coupling.\n"
        "2. Both boards on ±9 V. Join the two boards' black ground posts with a wire, and use the same ground for the scope.\n"
        "3. Strip: Rc2 from b2 to b5, Rc1 from d8 to d5 (both 10,000 ohms, measure them; within 1 % of each other is best).\n"
        f"4. Wires: purple from board A {A_XD} to strip e2; orange from board B {B_MXD} to strip e8; brown from strip a5 to board B {B_N2}.\n"
        "    Keep them short (under about 30 cm).\n"
        "5. Scope: Ch1 tip on board A b21, Ch2 tip on board B b21, both 0.5 V/div, DISPLAY → Format XY. Locked = thin diagonal line.\n"
        "6. On/off switch: pull the brown wire at board B (blob), push it back in (line within a few loops). Don't pull only the\n"
        "    purple wire: board B keeps Rc1 as a leak across C2 and becomes a different circuit."
    )
    fig.text(0.03, 0.205, steps, fontsize=8, va="top", linespacing=1.45)
    fig.text(
        0.03,
        0.012,
        "Expected (simulation, sync_check.py, 1N4148 model, measured parts): identical boards lock for Rc up to ~22k and not at 47k. "
        "With 10k, rms gap / rms x:\n0 for identical boards, 1 % if one board's A is 3 % off, 6 % if the two diodes' Is differ by 2×. "
        "Board B's diode must sit firmly; reseating changed board A's behaviour on 19 Nov.",
        fontsize=7.3,
        color="#555",
        va="bottom",
    )
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PREVIEW, dpi=100)
    plt.close(fig)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    sb.check_layout()
    used = (
        {h for p in sb.PARTS for h in p[2:4]}
        | {h for w in sb.WIRES for h in w[:2]}
        | {p[0] for p in sb.POWER}
    )
    used |= set(sb.PROBES.values()) | {
        h for c in sb.CHIPS for h in sb.chip_pins(c).values()
    }
    assert B_N2 not in used and B_MXD not in used and A_XD not in used, (
        "connection hole already taken"
    )
    page()
