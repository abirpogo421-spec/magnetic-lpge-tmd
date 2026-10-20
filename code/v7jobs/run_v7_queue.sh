#!/bin/bash
# v7 reruns with the fixed engine (lpge.py sha256 ff4a692d...). Launch from this folder:
#   nohup bash run_v7_queue.sh > run_v7_queue.log 2>&1 &
# Lanes 1-8: Fig. 9/11 edge runs (GGA, 2 T steps), then LDA threshold runs (10 T steps).
# Lane 9: Fig. 8 maps (4 materials), then Fig. 10 chi(omega) runs (8).
# Lanes 10-11: Fig. 12 lambda sweeps (2 materials each), then the decomposition for those materials.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo START $(date)
lane() { m=$1; mo=$2; python3 edge_v7.py $m $mo 0.0007 2 GGA > edge7_${m}_${mo}.log 2>&1; python3 edge_v7.py $m $mo 0.0007 10 LDA > edge7_${m}_${mo}_LDA.log 2>&1; echo lane $m $mo done $(date); }
for m in MoS2 MoSe2 WS2 WSe2; do for mo in TNN NN; do lane $m $mo & done; done
( for m in MoS2 MoSe2 WS2 WSe2; do python3 maps_v7.py $m > maps7_${m}.log 2>&1; done
  for m in MoS2 MoSe2 WS2 WSe2; do for mo in TNN NN; do python3 chiw9_v7.py $m $mo F 0.001 > chiw9v7_${m}_${mo}.log 2>&1; done; done; echo lane maps+chiw done $(date) ) &
( for m in MoS2 WS2; do python3 lam_v7.py $m F lam > lam7_${m}.log 2>&1; python3 decomp_v7.py $m > decomp7_${m}.log 2>&1; done; echo lane lamA done $(date) ) &
( for m in MoSe2 WSe2; do python3 lam_v7.py $m F lam > lam7_${m}.log 2>&1; python3 decomp_v7.py $m > decomp7_${m}.log 2>&1; done; echo lane lamB done $(date) ) &
wait
echo ALL DONE $(date)
