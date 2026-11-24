# Data

Every scope recording from 19 November 2026, as saved by `sprott_scope.py`. The files have not been changed since
they were recorded. 13 runs, 1,640 captures in all, none clipped.

## Runs

All runs use the same board: R4 = 10.03 kΩ, ±9 V, Ch1 = x, Ch2 = x′, both probes 1X. Board states A, B and C are
described in the [results](../docs/results.md#three-states-with-the-same-parts).

| file | start | captures | time/div | V/div | in D1 | state | notes |
|---|---|---|---|---|---|---|---|
| `sprott_test_20261119_133128.h5` | 13:31 | 20 | 10 ms | 0.5 V | original 1N4148 | A | pipeline test; attributes hold nominal part values |
| `sprott_R4_10030_20261119_134608.h5` | 13:46 | 200 | 10 ms | 0.5 V | original 1N4148 | A | main run |
| `sprott_noisecheck_20261119_135813.h5` | 13:58 | 20 | 10 ms | 0.5 V | original 1N4148 | A | Ch2 probe and ground moved |
| `sprott_R4_10030_rep1_20261119_140223.h5` | 14:02 | 200 | 10 ms | 0.5 V | original 1N4148 | A | repeat 1 |
| `sprott_R4_10030_rep2_20261119_141144.h5` | 14:11 | 200 | 10 ms | 0.5 V | original 1N4148 | A | repeat 2 |
| `sprott_R4_10030_rep3_20261119_142103.h5` | 14:21 | 200 | 10 ms | 0.5 V | original 1N4148 | A | repeat 3 |
| `sprott_R4_10030_tdiv25_20261119_143021.h5` | 14:30 | 200 | 25 ms | 0.5 V | original 1N4148 | A | 250 ms windows, 100 µs per sample |
| `sprott_diode_4148b_20261119_164709.h5` | 16:47 | 100 | 10 ms | 0.5 V | second 1N4148 | A | |
| `sprott_diode_4004_20261119_165526.h5` | 16:55 | 100 | 10 ms | 0.5 V | 1N4004 | A | |
| `sprott_diode_Ge_20261119_171152.h5` | 17:11 | 100 | 10 ms | 0.5 V | germanium | – | attractor a third the size |
| `sprott_diode_Ge_vdiv02_20261119_172031.h5` | 17:20 | 100 | 10 ms | 0.2 V | germanium | – | finer steps for the smaller signal |
| `sprott_after_swaps_20261119_181028.h5` | 18:10 | 100 | 10 ms | 0.5 V | original 1N4148, reseated | B | matches the model; used for the chaos analysis |
| `sprott_D1_reseated2_20261119_183303.h5` | 18:33 | 100 | 10 ms | 0.5 V | original 1N4148, reseated again | C | periodic |

The power was off for every diode swap. The `note` attribute of the 18:10 run says "which one to confirm": it was
confirmed afterwards that the diode was the original, labelled D1.

`capture_*.log` are the console output of each capture session.

## File format (HDF5)

| dataset | shape | meaning |
|---|---|---|
| `x` | captures × 2500 | Ch1 in volts (x, at the output of the last integrator) |
| `xd` | captures × 2500 | Ch2 in volts (x′, at the inverter output that also drives the diode) |
| `t` | 2500 | time of each sample within a capture, seconds |
| `capture_time_s` | captures | when each capture was taken, seconds from the start of the run |

Attributes: `R_ohm` and `C_F` (mean of the three integrator R and C), `R4_ohm`, `A` (= `R_ohm / R4_ohm`, the nominal
value), `tdiv` and `vdiv` (scope settings), `CH1_probe`, `CH2_probe`, `clipped_captures`, `scope` (scope ID string),
`started`, `note`, and `fake` (true only for simulated test captures).

The analysis does not use the `A` attribute. It uses A = τ/(R4·C1) = 0.982 with
$\tau = (R_1C_1\,R_2C_2\,R_3C_3)^{1/3}$ = 1.006 ms, from the measured parts: R1–R6 = 9.85, 9.95, 9.85, 10.03, 9.86, 9.93 kΩ; C1–C3 = 102.1, 99.4, 103.9 nF.

```python
import h5py

with h5py.File("data/sprott_after_swaps_20261119_181028.h5") as f:
    x, xd, t = f["x"][:], f["xd"][:], f["t"][:]   # volts, volts, seconds
    print(dict(f.attrs))
```
