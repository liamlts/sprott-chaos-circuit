#!/bin/bash
# Repeatability runs on the measured board (2026-11-19): 3 x 200 captures at 10 ms/div, then 1 x 200 at 25 ms/div.
# Same part values as the main run. Runs one after another (only one process may hold the scope).
cd "$(dirname "$0")/.."
PARTS="--r 9883 --c 101.8e-9 --r4 10030"
NOTE="R1 9.85k R2 9.95k R3 9.85k R4 10.03k R5 9.86k R6 9.93k C1 102.1n C2 99.4n C3 103.9n, +-9 V"
for i in 1 2 3; do
  echo "=== rep$i start $(date +%T)"
  python3 sprott_scope.py capture --n 200 $PARTS --label R4_10030_rep$i --note "$NOTE; repeatability run $i"
done
echo "=== tdiv25 start $(date +%T)"
python3 sprott_scope.py capture --n 200 $PARTS --tdiv 25e-3 --label R4_10030_tdiv25 --note "$NOTE; 25 ms/div long windows"
echo "=== all done $(date +%T)"
