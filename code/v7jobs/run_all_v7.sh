#!/bin/bash
# Launch from this folder:  nohup bash run_all_v7.sh > run_all_v7.log 2>&1 &
export OMP_NUM_THREADS=1
for m in MoS2 MoSe2 WS2 WSe2; do
  python3 edge_v7.py $m TNN > edge7_${m}_TNN.log 2>&1 &
  python3 edge_v7.py $m NN  > edge7_${m}_NN.log 2>&1 &
done; wait
for m in MoS2 MoSe2 WS2 WSe2; do python3 maps_v7.py $m > maps7_${m}.log 2>&1 & done; wait
echo ALL DONE
