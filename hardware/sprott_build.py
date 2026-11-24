"""Breadboard build sheet for Sprott's diode jerk circuit (Sprott, IEEE TCAS-II 58, 240, 2011, Fig. 1).

x''' = -x'' - x - f(x'): three integrators in a loop, an inverter, and one diode.
Page 1: hole-by-hole layout on the PBB-509 (same board drawing as the Chua build sheet).
Page 2: parts list, what each op-amp does, bring-up steps, and the simulated x' vs x picture.
A netlist check runs first and stops the script if the wiring does not match the schematic.
Writes sprott_build.pdf, sprott_build_page1.png (150 dpi), sprott_build_preview.png (page 1),
sprott_build_p2_preview.png.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge
from matplotlib.transforms import Affine2D
from scipy.integrate import solve_ivp

OUT_PDF = "sprott_build.pdf"
OUT_P1 = "sprott_build_preview.png"
OUT_PAGE1 = "sprott_build_page1.png"  # page 1 at 150 dpi, shown in the README and docs/build.md
OUT_P2 = "sprott_build_p2_preview.png"

# component values (all resistors equal, all three integrator capacitors equal; RC sets the speed)
R_OHM = 10_000
C_NF = 100
R4_OHM = 10_000  # damping resistor across C1 ("the leftmost resistor" in the paper); A = R / R4
SUPPLY_V = 9.0
DIODE = "1N4148"
IS, NIDEAL, VT = 2.52e-9, 1.752, 0.02585  # 1N4148 model used for the simulated picture

NCOL = 30
ROWS = "abcdefghij"
ROW_Y = {r: (i + 1 if i < 5 else i + 2.2) for i, r in enumerate(ROWS)}
RAIL_Y = {"bot_neg": -1.6, "bot_gnd": -0.6, "top_gnd": 12.8, "top_pos": 13.8}
RAIL_STYLE = {
    "top_pos": ("#c0392b", f"+{SUPPLY_V:.0f} volts"),
    "top_gnd": ("#1f4e9c", "Ground"),
    "bot_gnd": ("#1f4e9c", "Ground"),
    "bot_neg": ("#c0392b", f"−{SUPPLY_V:.0f} volts"),
}
BAND = {
    0: "black",
    1: "#7b3f00",
    2: "#d62728",
    3: "#ff7f0e",
    4: "#f1c40f",
    5: "#2e8b57",
    6: "#1f5fbf",
    7: "#8e44ad",
    8: "#8c8c8c",
    9: "white",
}
COLOR_NAME = {
    0: "black",
    1: "brown",
    2: "red",
    3: "orange",
    4: "yellow",
    5: "green",
    6: "blue",
    7: "violet",
    8: "grey",
    9: "white",
}
LABEL_KW = dict(fontsize=8, ha="center", va="center", zorder=12)
FILM, CERAMIC = "#c62828", "#ef8a17"

# signal colours: one colour per signal, used for its wires and its labels
C_X, C_XD, C_MXD, C_XDD = "#1f5fbf", "#8e44ad", "#e67e22", "#16a085"

# ---------------------------------------------------------------- the layout, as data
# chips: name -> first column (pin 1 / pin 8 column); notch faces left
CHIPS = {"U1": 8, "U2": 21}
# op-amp roles: (chip, half) -> (job, output node, inverting-input node)
OPAMPS = {
    ("U1", "B"): ("integrator 1", "xdd", "N1"),
    ("U1", "A"): ("integrator 2", "mxd", "N2"),
    ("U2", "A"): ("integrator 3", "x", "N3"),
    ("U2", "B"): ("inverter", "xd", "N4"),
}
# two-lead parts: (name, kind, hole1, hole2, value).  diode: hole1 = anode, hole2 = cathode (band)
PARTS = [
    ("R1", "R", "i15", "i10", R_OHM),
    ("R2", "R", "c6", "c9", R_OHM),
    ("R3", "R", "c19", "c22", R_OHM),
    ("R4", "R", "h9", "h10", R4_OHM),
    ("R5", "R", "i19", "i23", R_OHM),
    ("R6", "R", "g22", "g23", R_OHM),
    ("C1", "C", "j9", "j10", C_NF),
    ("C2", "C", "d8", "d9", C_NF),
    ("C3", "C", "d21", "d22", C_NF),
    ("D1", "D", "g14", "g10", DIODE),
]
# signal wires: (hole1, hole2, colour)
WIRES = [
    ("i9", "i6", C_XDD),
    ("g6", "d6", C_XDD),
    ("b8", "b19", C_MXD),
    ("d19", "g19", C_MXD),
    ("h22", "h14", C_XD),
    ("a21", "a15", C_X),
    ("d15", "g15", C_X),
]
# power wires: (hole, rail, colour)
POWER = [
    ("j8", "top_pos", "#d62728"),
    ("j21", "top_pos", "#d62728"),
    ("j11", "top_gnd", "black"),
    ("j24", "top_gnd", "black"),
    ("a10", "bot_gnd", "black"),
    ("a23", "bot_gnd", "black"),
    ("a11", "bot_neg", "#27ae60"),
    ("a24", "bot_neg", "#27ae60"),
]
BYPASS = [
    ("top_pos", "top_gnd", 14),
    ("top_pos", "top_gnd", 27),
    ("bot_gnd", "bot_neg", 14),
    ("bot_gnd", "bot_neg", 27),
]
PROBES = {"Channel 1 (x)": "b21", "Channel 2 (x′)": "j22"}

# what the schematic says each node connects to (Sprott 2011, Fig. 1)
EXPECTED = {
    "N1": {
        "U1.6",
        "R1",
        "R4",
        "C1",
        "D1-",
    },
    "xdd": {"U1.7", "R4", "C1", "R2"},
    "N2": {"U1.2", "R2", "C2"},
    "mxd": {"U1.1", "C2", "R3", "R5"},
    "N3": {"U2.2", "R3", "C3"},
    "x": {"U2.1", "C3", "R1", "Channel 1 (x)"},
    "N4": {"U2.6", "R5", "R6"},
    "xd": {"U2.7", "R6", "D1+", "Channel 2 (x′)"},
    "gnd": {"U1.3", "U1.5", "U2.3", "U2.5", "top_gnd", "bot_gnd"},
    "vpos": {"U1.8", "U2.8", "top_pos"},
    "vneg": {"U1.4", "U2.4", "bot_neg"},
}


def H(hole):
    return float(hole[1:]), ROW_Y[hole[0]]


def RL(rail, col):
    return float(col), RAIL_Y[rail]


def group(hole):
    """Electrical strip a hole belongs to: column + half (a-e or f-j)."""
    return (int(hole[1:]), "bot" if hole[0] in "abcde" else "top")


def chip_pins(name):
    c0 = CHIPS[name]
    pins = {}
    for k in range(4):
        pins[k + 1] = f"e{c0 + k}"
        pins[8 - k] = f"f{c0 + k}"
    return pins


# ---------------------------------------------------------------- netlist check
def check_layout():
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    used = {}

    def take(hole, who):
        if hole in used:
            raise SystemExit(f"hole {hole} used twice: {used[hole]} and {who}")
        used[hole] = who

    members = []  # (terminal name, group)
    for chip in CHIPS:
        for pin, hole in chip_pins(chip).items():
            take(hole, f"{chip}.{pin}")
            members.append((f"{chip}.{pin}", group(hole)))
    for name, kind, h1, h2, _ in PARTS:
        take(h1, name)
        take(h2, name)
        t1, t2 = (f"{name}+", f"{name}-") if kind == "D" else (name, name)
        members += [(t1, group(h1)), (t2, group(h2))]
    for h1, h2, _ in WIRES:
        take(h1, "wire")
        take(h2, "wire")
        union(group(h1), group(h2))
    for hole, rail, _ in POWER:
        take(hole, "power wire")
        union(group(hole), rail)
    for name, hole in PROBES.items():
        take(hole, name)
        members.append((name, group(hole)))
    for rail in RAIL_Y:
        members.append((rail, rail))
    union("top_gnd", "bot_gnd")  # both ground rails go to the common post

    nets = {}
    for term, g in members:
        nets.setdefault(find(g), set()).add(term)
    found = sorted(nets.values(), key=sorted)
    want = sorted(EXPECTED.values(), key=sorted)
    if found != want:
        for n in found:
            if n not in want:
                print("unexpected net:", sorted(n))
        raise SystemExit("layout does not match the schematic")

    # nothing may sit in a hole that a wire or a part body passes over
    def passed(h1, h2):
        (x1, y1), (x2, y2) = H(h1), H(h2)
        if y1 == y2:
            return [
                f"{h1[0]}{c}" for c in range(int(min(x1, x2)) + 1, int(max(x1, x2)))
            ]
        return []

    for h1, h2, _ in WIRES:
        for h in passed(h1, h2):
            if h in used:
                raise SystemExit(f"wire {h1}-{h2} passes over {h} ({used[h]})")
    for name, _, h1, h2, _ in PARTS:
        for h in passed(h1, h2):
            if h in used:
                raise SystemExit(f"{name} body covers {h} ({used[h]})")
    print("netlist check passed:", len(found), "nets,", len(used), "holes used")


# ---------------------------------------------------------------- drawing helpers
def bands(ohms):
    s = str(int(ohms))
    return [BAND[int(s[0])], BAND[int(s[1])], BAND[len(s) - 2], "#c9a227"]


def color_code(ohms):
    s = str(int(ohms))
    return ", ".join(COLOR_NAME[d] for d in (int(s[0]), int(s[1]), len(s) - 2))


def callout(ax, xy, text, txy, color="#333"):
    ax.annotate(
        text,
        xy=xy,
        xytext=txy,
        color=color,
        **LABEL_KW,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.7, shrinkA=0, shrinkB=3),
        bbox=dict(fc="white", ec=color, lw=0.7, boxstyle="round,pad=0.25"),
    )


def lead(ax, p1, p2):
    ax.plot(*zip(p1, p2), color="#8a8a8a", lw=1.8, zorder=4, solid_capstyle="round")
    for p in (p1, p2):
        ax.add_patch(Circle(p, 0.16, color="#555", zorder=4.5))


def mid(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def body(ax, p1, p2, L, w, fc, ec, rounding):
    cx, cy = mid(p1, p2)
    ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    t = Affine2D().rotate_deg_around(cx, cy, ang) + ax.transData
    ax.add_patch(
        FancyBboxPatch(
            (cx - L / 2, cy - w / 2),
            L,
            w,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            fc=fc,
            ec=ec,
            lw=0.8,
            zorder=5,
            transform=t,
        )
    )
    return cx, cy, t


def resistor(ax, p1, p2, ohms):
    lead(ax, p1, p2)
    L = min(1.2, 0.72 * np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    w = 0.44
    cx, cy, t = body(ax, p1, p2, L, w, "#e8d3a8", "#7a6a4a", 0.15)
    for k, c in enumerate(bands(ohms)):
        x0 = cx - L / 2 + L * (0.16 + 0.19 * k)
        ax.add_patch(
            Rectangle(
                (x0, cy - w / 2), L * 0.09, w, fc=c, ec="none", zorder=6, transform=t
            )
        )
    return cx, cy


def capacitor(ax, p1, p2, fc=FILM):
    lead(ax, p1, p2)
    cx, cy = mid(p1, p2)
    film = fc == FILM
    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.32, cy - 0.25),
            0.64,
            0.5,
            boxstyle=f"round,pad=0,rounding_size={0.03 if film else 0.24}",
            fc=fc,
            ec="#5a1010" if film else "#7a4a10",
            lw=0.8,
            zorder=6,
        )
    )
    return cx, cy


def diode(ax, anode, cathode):
    """Glass diode; black band at the cathode end."""
    lead(ax, anode, cathode)
    L, w = 1.3, 0.4
    cx, cy, t = body(ax, anode, cathode, L, w, "#f0a24a", "#8a4a10", 0.12)
    ax.add_patch(
        Rectangle(
            (cx + L / 2 - 0.3, cy - w / 2),
            0.16,
            w,
            fc="black",
            ec="none",
            zorder=6,
            transform=t,
        )
    )
    return cx, cy


def wire(ax, p1, p2, color, lw=2.8):
    ax.plot(*zip(p1, p2), color="white", lw=lw + 2.4, zorder=7, solid_capstyle="round")
    ax.plot(*zip(p1, p2), color=color, lw=lw, zorder=7.5, solid_capstyle="round")
    for p in (p1, p2):
        ax.add_patch(Circle(p, 0.17, color=color, zorder=8))


def chip(ax, c0, name):
    x0, x1 = c0 - 0.45, c0 + 3.45
    y0, y1 = ROW_Y["e"] + 0.32, ROW_Y["f"] - 0.32
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#222", ec="black", zorder=6))
    ax.add_patch(Wedge((x0, (y0 + y1) / 2), 0.3, -90, 90, fc="#999", zorder=7))
    ax.text(
        (x0 + x1) / 2 + 0.15,
        (y0 + y1) / 2,
        name,
        color="white",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        zorder=8,
    )
    for k in range(4):
        x = c0 + k
        for y in (ROW_Y["e"], ROW_Y["f"]):
            ax.add_patch(
                Rectangle(
                    (x - 0.14, y - 0.14),
                    0.28,
                    0.28,
                    fc="#c8c8c8",
                    ec="#444",
                    lw=0.6,
                    zorder=7,
                )
            )
        for y, num in ((ROW_Y["e"] - 0.45, k + 1), (ROW_Y["f"] + 0.45, 8 - k)):
            ax.text(
                x,
                y,
                str(num),
                ha="center",
                va="center",
                fontsize=6,
                color="#b03a2e",
                zorder=9,
                fontweight="bold",
            )


def board(ax):
    ax.add_patch(
        FancyBboxPatch(
            (0.0, -2.3),
            NCOL + 1.0,
            16.8,
            boxstyle="round,pad=0,rounding_size=0.4",
            fc="#f7f5ef",
            ec="#c9c4b5",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, ROW_Y["e"] + 0.5),
            NCOL + 1.0,
            ROW_Y["f"] - ROW_Y["e"] - 1.0,
            fc="#e3dfd2",
            ec="none",
            zorder=0.5,
        )
    )
    for r in ROWS:
        for c in range(1, NCOL + 1):
            ax.add_patch(
                Rectangle(
                    (c - 0.1, ROW_Y[r] - 0.1),
                    0.2,
                    0.2,
                    fc="#3a3a3a",
                    ec="none",
                    zorder=1,
                )
            )
        ax.text(0.45, ROW_Y[r], r, ha="center", va="center", fontsize=6.5, color="#777")
    for c in (1, 5, 10, 15, 20, 25, 30):
        for y in (ROW_Y["a"] - 0.6, ROW_Y["j"] + 0.6):
            ax.text(c, y, str(c), ha="center", va="center", fontsize=6, color="#777")
    for rail, y in RAIL_Y.items():
        col, txt = RAIL_STYLE[rail]
        side = 0.32 if rail in ("top_pos", "bot_gnd") else -0.32
        ax.plot([0.6, NCOL + 0.4], [y + side, y + side], color=col, lw=1.2, zorder=1)
        for c in range(1, NCOL + 1):
            ax.add_patch(
                Rectangle(
                    (c - 0.1, y - 0.1), 0.2, 0.2, fc="#3a3a3a", ec="none", zorder=1
                )
            )
        ax.text(
            NCOL + 1.4,
            y,
            txt,
            ha="left",
            va="center",
            fontsize=8,
            color=col,
            fontweight="bold",
        )


def probe(ax, p, text, txy, color):
    ax.add_patch(Circle(p, 0.32, fc="none", ec=color, lw=2, zorder=10))
    ax.annotate(
        text,
        xy=p,
        xytext=txy,
        fontsize=8,
        color=color,
        ha="center",
        va="center",
        zorder=12,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkB=5),
        bbox=dict(fc="white", ec=color, lw=1.1, boxstyle="round,pad=0.3"),
    )


def rlabel(ohms):
    return f"{ohms:,} ohms"


# ---------------------------------------------------------------- page 1
def page_breadboard(pdf):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.01)
    ax.set_xlim(-5.0, NCOL + 4.5)
    ax.set_ylim(-12.0, 18.8)
    ax.set_aspect("equal")
    ax.axis("off")

    board(ax)
    for name, c0 in CHIPS.items():
        chip(ax, c0, f"chip {name}")

    # power posts (not connected to the rails on this board)
    for txt, p, c, above in (
        (
            f"Positive post (yellow)\nset to +{SUPPLY_V:.1f} volts",
            (-2.6, 13.8),
            "#f1c40f",
            True,
        ),
        ("Common post\n(ground, black)", (-2.6, 6.1), "#222", False),
        (
            f"Negative post (green)\nset to −{SUPPLY_V:.1f} volts",
            (-2.6, -1.6),
            "#27ae60",
            True,
        ),
    ):
        ax.add_patch(Circle(p, 0.5, fc=c, ec="black", zorder=8))
        ax.text(
            p[0],
            p[1] + (0.9 if above else -0.9),
            txt,
            ha="center",
            va="bottom" if above else "top",
            fontsize=7.5,
        )
    wire(ax, (-2.1, 13.8), RL("top_pos", 1), "#d62728")
    wire(ax, (-2.1, -1.6), RL("bot_neg", 1), "#27ae60")
    for yend in (12.8, -0.6):
        ax.plot(
            [-2.1, -1.0, 1],
            [6.1, yend, yend],
            color="black",
            lw=2.8,
            zorder=7.5,
            solid_joinstyle="round",
        )
        ax.add_patch(Circle((1, yend), 0.17, color="black", zorder=8))

    for hole, rail, color in POWER:
        wire(ax, H(hole), RL(rail, group(hole)[0]), color)
    byp = [capacitor(ax, RL(r1, c), RL(r2, c), fc=CERAMIC) for r1, r2, c in BYPASS]

    pos = {}
    for name, kind, h1, h2, val in PARTS:
        if kind == "R":
            pos[name] = resistor(ax, H(h1), H(h2), val)
        elif kind == "C":
            pos[name] = capacitor(ax, H(h1), H(h2))
        else:
            pos[name] = diode(ax, H(h1), H(h2))
    for h1, h2, color in WIRES:
        wire(ax, H(h1), H(h2), color)

    up, up2, dn, dn2 = 16.9, 15.4, -3.7, -5.7
    cap = f"{C_NF} nanofarads"
    callout(ax, pos["C1"], f"C1 (red box)\n{cap}", (4.8, up))
    callout(ax, pos["R4"], f"R4: {rlabel(R4_OHM)}\n(sets the damping)", (9.6, up))
    callout(ax, pos["R1"], f"R1: {rlabel(R_OHM)}", (12.6, up2))
    callout(ax, pos["D1"], f"D1: {DIODE}\nband toward column 10", (19.8, up))
    callout(ax, pos["R5"], f"R5: {rlabel(R_OHM)}", (21.2, up2))
    callout(ax, pos["R6"], f"R6: {rlabel(R_OHM)}", (24.6, up))
    callout(ax, byp[1], "Bypass (orange)\n10–100 nanofarads", (29.4, up))
    callout(ax, byp[0], "Bypass (orange)\n10–100 nanofarads", (14.6, up))
    callout(ax, pos["R2"], f"R2: {rlabel(R_OHM)}", (4.0, dn))
    callout(ax, pos["C2"], f"C2 (red box)\n{cap}", (8.6, dn2))
    callout(ax, pos["R3"], f"R3: {rlabel(R_OHM)}", (19.4, dn))
    callout(ax, pos["C3"], f"C3 (red box)\n{cap}", (22.4, dn2))
    callout(ax, byp[2], "Bypass (orange)\n10–100 nanofarads", (14.5, dn2))
    callout(ax, byp[3], "Bypass (orange)\n10–100 nanofarads", (28.4, dn2))

    # which op-amp does what (A = pins 1-3, bottom; B = pins 5-7, top)
    for (chipname, half), (job, _, _) in OPAMPS.items():
        c0 = CHIPS[chipname]
        y = ROW_Y["e"] + 0.55 if half == "A" else ROW_Y["f"] - 0.55
        ax.text(
            c0 + 1.5,
            y,
            f"{half}: {job}",
            ha="center",
            va="center",
            fontsize=5.8,
            color="#e8e8e8",
            zorder=9,
        )

    probe(
        ax,
        H(PROBES["Channel 1 (x)"]),
        "Channel 1 probe tip: x\n(left–right on screen)",
        (25.4, dn),
        C_X,
    )
    probe(
        ax,
        H(PROBES["Channel 2 (x′)"]),
        "Channel 2 probe tip: x′\n(up–down on screen)",
        (26.3, 15.4),
        C_XD,
    )

    ax.set_title(
        "Sprott jerk circuit on the powered breadboard (PBB-509): where every part goes",
        fontsize=13,
        loc="left",
        pad=0,
    )
    notes = (
        "• Chip notches face left. Small red numbers are pin numbers. Both chips are TL082 (TL072 or LF353 also fit).\n"
        "• In each column, holes a–e are one connection and holes f–j are another. Everything in one such group is one point.\n"
        "• Wire colours are the four signals:  teal = x″ (U1 pin 7),  orange = −x′ (U1 pin 1),  "
        "blue = x (U2 pin 1),  purple = x′ (U2 pin 7).\n"
        "• Red wire = +9 volts,  green wire = −9 volts,  black wire = ground. "
        "Each chip: pin 8 to +9, pin 4 to −9, pins 3 and 5 to ground.\n"
        "• The diode only conducts one way: its black band (cathode) goes in g10, the plain end in g14.\n"
        f"• All six resistors are {R_OHM:,} ohms ({color_code(R_OHM)}); all three red boxes are {C_NF} nanofarads, "
        "film (no electrolytics). RC = 1 ms, about 100 Hz.\n"
        "• The posts are not connected to the rails: add all four post wires as drawn. Build with the power off.\n"
        "• Scope: both ground clips on a ground rail, 0.5 V/div on both channels, DISPLAY → Format XY. "
        "You should see the loop on page 2."
    )
    ax.text(-4.6, -7.4, notes, fontsize=7.6, ha="left", va="top", linespacing=1.3)
    pdf.savefig(fig)
    fig.savefig(OUT_P1, dpi=100)
    fig.savefig(OUT_PAGE1, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- page 2
def simulate():
    R4ratio = R_OHM / R4_OHM

    def rhs(t, s):
        x, y, z = s
        return [
            y,
            z,
            -R4ratio * z - x - R_OHM * IS * np.expm1(min(y / (NIDEAL * VT), 60.0)),
        ]

    sol = solve_ivp(
        rhs,
        (0, 700),
        [0.0, 0.4, 0.0],
        method="LSODA",
        rtol=1e-9,
        atol=1e-11,
        max_step=0.02,
    )
    keep = sol.t > 200
    return sol.t[keep] - 200, sol.y[:, keep]


def page_parts(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(
        0.03,
        0.955,
        "Sprott jerk circuit: parts, what each op-amp does, and what you should see",
        fontsize=13,
        ha="left",
        va="top",
    )

    rows = [
        (
            "U1, U2",
            "Dual op-amp chip (8 pins)",
            "TL082 (or TL072, LF353)",
            "columns 8–11 and 21–24",
        ),
        (
            "R1",
            "Resistor",
            f"{R_OHM:,} Ω ({color_code(R_OHM)})",
            "i10–i15 (x into integrator 1)",
        ),
        ("R2", "Resistor", f"{R_OHM:,} Ω", "c6–c9 (x″ into integrator 2)"),
        ("R3", "Resistor", f"{R_OHM:,} Ω", "c19–c22 (−x′ into integrator 3)"),
        ("R4", "Resistor", f"{R4_OHM:,} Ω", "h9–h10, across C1 (damping)"),
        ("R5", "Resistor", f"{R_OHM:,} Ω", "i19–i23 (inverter input)"),
        ("R6", "Resistor", f"{R_OHM:,} Ω", "g22–g23 (inverter feedback)"),
        (
            "C1, C2, C3",
            "Film capacitor (red box)",
            f"{C_NF} nF (104), non-polar",
            "j9–j10, d8–d9, d21–d22",
        ),
        (
            "D1",
            "Diode",
            f"{DIODE} (or 1N914, 1N4001)",
            "g14 (plain end) to g10 (band)",
        ),
        (
            "—",
            "Bypass capacitors ×4",
            "10–100 nF ceramic",
            "across the rails, columns 14 and 27",
        ),
        ("—", "Jumper wires", "4 colours + power", "see page 1"),
    ]
    ax = fig.add_axes((0.03, 0.50, 0.58, 0.40))
    ax.axis("off")
    tab = ax.table(
        cellText=rows,
        colLabels=["Part", "What it is", "Value", "Holes"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.11, 0.25, 0.30, 0.34],
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.35)
    for (r, _), cell in tab.get_celld().items():
        cell.set_edgecolor("#bbb")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eee")

    jobs = [
        (
            "U1 B (pins 5–7)",
            "integrator 1",
            "x″ on pin 7",
            "sums x (R1), x″ (R4) and the diode at pin 6",
        ),
        ("U1 A (pins 1–3)", "integrator 2", "−x′ on pin 1", "integrates x″ (R2, C2)"),
        ("U2 A (pins 1–3)", "integrator 3", "x on pin 1", "integrates −x′ (R3, C3)"),
        (
            "U2 B (pins 5–7)",
            "inverter",
            "x′ on pin 7",
            "flips −x′ (R5, R6); feeds the diode",
        ),
    ]
    ax2 = fig.add_axes((0.03, 0.30, 0.58, 0.17))
    ax2.axis("off")
    t2 = ax2.table(
        cellText=jobs,
        colLabels=["Op-amp", "Job", "Output", "Input side"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.2, 0.17, 0.17, 0.46],
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(8)
    t2.scale(1, 1.35)
    for (r, _), cell in t2.get_celld().items():
        cell.set_edgecolor("#bbb")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eee")

    steps = (
        "Bring-up\n"
        "1. Power off. Set the supply to ±9 V. Place both chips (notch left), then the power wires and bypass caps.\n"
        "2. Power on with only the chips in: pin 8 of each chip at +9 V, pin 4 at −9 V, chips cool to the touch.\n"
        "   Don't check the outputs yet: with nothing on the inputs they sit near a rail.\n"
        "3. Power off, add the rest. Check the diode band (g10) and the four coloured wires.\n"
        "4. Power on. Channel 1 on b21 (x), Channel 2 on j22 (x′), 0.5 V/div, XY mode.\n"
        "   The loop should start by itself within a second and look like the picture on the right.\n"
        "5. Single clean loop or a double loop instead? You are in a periodic window. Change R4 a little.\n"
        "   With a 1N4148 the model at my measured parts is chaotic at R4 = 10, 11, 15 and 20k and periodic at\n"
        "   9.1k and below, with a narrow periodic window at 10.31–10.34k. Your own parts shift these a little.\n"
        "6. Flat line at a rail? An op-amp output is stuck: check the integrator capacitors and R4."
    )
    fig.text(0.03, 0.27, steps, fontsize=8.4, ha="left", va="top", linespacing=1.45)

    t, (x, y, _) = simulate()
    axp = fig.add_axes((0.67, 0.50, 0.30, 0.40))
    axp.plot(x, y, color=C_XD, lw=0.5)
    axp.set_xlabel("x, Channel 1 (V)")
    axp.set_ylabel("x′, Channel 2 (V)")
    axp.set_title("Simulated scope picture (XY)", fontsize=9)
    axp.set_xlim(-2.0, 1.0)
    axp.set_ylim(-1.25, 1.0)
    axp.set_aspect("equal")
    axp.grid(color="#ddd", lw=0.5)

    axt = fig.add_axes((0.67, 0.10, 0.30, 0.28))
    ms = t * R_OHM * C_NF * 1e-9 * 1e3
    m = ms < 100
    axt.plot(ms[m], x[m], color=C_X, lw=0.8, label="x")
    axt.plot(ms[m], y[m] - 2.5, color=C_XD, lw=0.8, label="x′ (shifted down)")
    axt.set_xlabel("time (ms)")
    axt.set_ylabel("volts")
    axt.set_title("Simulated waveforms (about 100 Hz)", fontsize=9)
    axt.legend(fontsize=7, loc="upper right", frameon=False)
    axt.set_xlim(0, 100)

    fig.text(
        0.67,
        0.045,
        f"Simulation: Sprott 2011, Eq. (2), {DIODE} diode model,\n"
             f"R = {R_OHM:,} ohms, C = {C_NF} nF. Circuit: J. C. Sprott,\n"
             "IEEE Trans. Circuits Syst. II 58, 240 (2011), Fig. 1.",
        fontsize=7,
        ha="left",
        va="top",
        color="#555",
    )
    pdf.savefig(fig)
    fig.savefig(OUT_P2, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    check_layout()
    with PdfPages(OUT_PDF) as pdf:
        page_breadboard(pdf)
        page_parts(pdf)
    print("wrote", OUT_PDF)
