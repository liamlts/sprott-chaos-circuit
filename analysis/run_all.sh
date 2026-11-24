#!/bin/bash
# Re-run the analysis behind the pages in order. Logs go to analysis/logs/, figures to figures/,
# key numbers to analysis/results/*.json. Takes ~20 min (simulations are cached in analysis/cache/).
# Side studies (fig_jump_model, fig_diode_inferred, fig_smoother, ...) run on their own.
set -e
cd "$(dirname "$0")"
mkdir -p logs
# fig_model_scan integrates ~2,650 trajectories; it is slow only when analysis/cache/model_scan.npz is missing.
# check_window_C reads that cache.
for s in fig_peaks fig_returnmap fig_origin fig_compare fig_repeat fig_diodes fig_ge fig_states \
         fig_model_scan check_window_C check_diode_range fig_contact_scan; do
  echo "=== $s $(date +%T)"
  python3 "$s.py" > "logs/$s.log" 2>&1
done
echo "=== fig_stats (main run) $(date +%T)"
python3 fig_stats.py > logs/fig_stats.log 2>&1
for f in ../data/sprott_R4_10030_rep*.h5 ../data/sprott_R4_10030_tdiv25_*.h5 \
         ../data/sprott_diode_*.h5 ../data/sprott_R4_10030_D1back_*.h5 \
         ../data/sprott_after_swaps_*.h5 ../data/sprott_D1_reseated2_*.h5; do
  [ -e "$f" ] || continue
  n=$(basename "$f" .h5)
  echo "=== fig_stats $n $(date +%T)"
  python3 fig_stats.py "$f" > "logs/fig_stats_$n.log" 2>&1
done
echo "=== fig_summary $(date +%T)"
python3 fig_summary.py > logs/fig_summary.log 2>&1
# run-B chaos figures; lyap_spectrum first (the others read its JSON). Slow parts are cached in cache/.
for s in lyap_spectrum fig_B_attractor fig_B_returnmap fig_B_lyapunov fig_B_dimension; do
  echo "=== $s $(date +%T)"
  python3 "$s.py" > "logs/$s.log" 2>&1
done
# figures and numbers for the GitHub pages (docs/) that the loops above don't make; build_numbers reads
# model_scan.json and B_attractor.json, so it runs after them
for s in fig_x_minus_int build_numbers fig_talk_trace fig_talk_board_model fig_talk_bifurcation fig_talk_divergence \
         fig_talk_states; do
  echo "=== $s $(date +%T)"
  python3 "$s.py" > "logs/$s.log" 2>&1
done
echo "=== build sheet and schematic $(date +%T)"
(cd ../hardware && python3 sprott_build.py) > logs/sprott_build.log 2>&1
../hardware/make_schematic.sh > logs/make_schematic.log 2>&1
echo "=== write-up $(date +%T)"
../writeup/build.sh > logs/build_writeup.log 2>&1
echo "=== all done $(date +%T)"
