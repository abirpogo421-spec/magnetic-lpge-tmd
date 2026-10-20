#!/bin/zsh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python3}
mkdir -p "$(dirname $0)/runs"; cd "$(dirname $0)/runs"
P=../
for m in MoS2 MoSe2 WS2 WSe2; do (nohup $PY -u $P/run_prod.py $m F > prodF_$m.log 2>&1 &); done
(nohup zsh -c "for m in MoS2 MoSe2 WS2 WSe2; do $PY -u $P/run_prod.py \$m Z; done; for m in MoS2 MoSe2 WS2 WSe2; do $PY -u $P/run_edge.py \$m Z B; done; for m in MoS2 MoSe2 WS2 WSe2; do $PY -u $P/run_edge.py \$m Z lam; done" > chainZ.log 2>&1 &)
(nohup zsh -c "for m in MoS2 MoSe2; do $PY -u $P/run_edge.py \$m F B; done" > edgeF_a.log 2>&1 &)
(nohup zsh -c "for m in WS2 WSe2; do $PY -u $P/run_edge.py \$m F B; done" > edgeF_b.log 2>&1 &)
(nohup zsh -c "for m in MoS2 MoSe2 MoTe2; do $PY -u $P/run_edge.py \$m F lam; done" > lamF_a.log 2>&1 &)
(nohup zsh -c "for m in WS2 WSe2 WTe2; do $PY -u $P/run_edge.py \$m F lam; done" > lamF_b.log 2>&1 &)
