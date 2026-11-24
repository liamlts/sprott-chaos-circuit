"""Markdown table of every run on 2026-11-19 from analysis/results/stats*.json (written by fig_stats.py).
Writes the table into docs/results.md between the <!-- RUN_TABLE --> and <!-- /RUN_TABLE --> markers, so the
page on GitHub shows it as it is; build.sh runs this first. Bootstrap errors over captures are in the JSON.
"""

import json
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
DATA = ROOT / "data"
PAGE = ROOT / "docs" / "results.md"
START, END = "<!-- RUN_TABLE -->", "<!-- /RUN_TABLE -->"
SXD, MXD = "std x'", "mean x'"  # JSON keys for x'

# run stem -> (time, what was in D1 / what changed, board state)
RUNS = {
    "sprott_R4_10030_20261119_134608": ("13:46", "original D1", "A"),
    "sprott_R4_10030_rep1_20261119_140223": ("14:02", "original D1", "A"),
    "sprott_R4_10030_rep2_20261119_141144": ("14:11", "original D1", "A"),
    "sprott_R4_10030_rep3_20261119_142103": ("14:21", "original D1", "A"),
    "sprott_R4_10030_tdiv25_20261119_143021": ("14:30", "original D1, 25 ms/div", "A"),
    "sprott_diode_4148b_20261119_164709": ("16:47", "second 1N4148", "A"),
    "sprott_diode_4004_20261119_165526": ("16:55", "1N4004", "A"),
    "sprott_diode_Ge_20261119_171152": ("17:11", "germanium", "Ge"),
    "sprott_diode_Ge_vdiv02_20261119_172031": ("17:20", "germanium, 0.2 V/div", "Ge"),
    "sprott_after_swaps_20261119_181028": ("18:10", "D1 reseated", "B"),
    "sprott_D1_reseated2_20261119_183303": ("18:33", "D1 reseated again", "C"),
}


def stats_file(stem):
    if stem == "sprott_R4_10030_20261119_134608":
        return RES / "stats.json"
    return RES / f"stats_{stem.replace('sprott_', '')}.json"


def table():
    head = (
        "| time | D1 / change | n | state | rms x (V) | rms x′ (V) | peak f (Hz) "
        "| time within 0.2 V | mean x (V) | mean x′ (mV) |"
    )
    lines = [head]
    # dash counts set the relative column widths in the PDF (pandoc pipe tables)
    widths = [5, 20, 4, 5, 7, 7, 11, 8, 7, 7]
    lines.append("|" + "|".join("-" * w for w in widths) + "|")
    model = None
    for stem, (t, what, state) in RUNS.items():
        d = json.loads(stats_file(stem).read_text())
        v, e = d["data"]["val"], d["data"]["err"]
        if model is None:
            model = d["model"]["val"]
        with h5py.File(DATA / f"{stem}.h5") as f:
            n = f["x"].shape[0]
        lines.append(
            f"| {t} | {what} | {n} | {state} | {v['std x']:.3f} | {v[SXD]:.3f} "
            f"| {v['peak f (Hz)']:.1f} ± {e['peak f (Hz)']:.1f} | {v['time frac r<0.2']:.3f} "
            f"| {v['mean x']:.3f} | {v[MXD] * 1e3:+.1f} |"
        )
    lines.append(
        f"| | model, A = 0.982 | 400 | | {model['std x']:.3f} | {model[SXD]:.3f} "
        f"| {model['peak f (Hz)']:.1f} | {model['time frac r<0.2']:.3f} | {model['mean x']:.3f} | 0 |"
    )

    return "\n".join(lines)


def main():
    text = PAGE.read_text(encoding="utf-8")
    i, j = text.index(START) + len(START), text.index(END)
    PAGE.write_text(text[:i] + "\n\n" + table() + "\n\n" + text[j:], encoding="utf-8")
    print(f"run table -> {PAGE}")


if __name__ == "__main__":
    main()
