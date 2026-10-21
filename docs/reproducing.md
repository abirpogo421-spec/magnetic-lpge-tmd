# Reproducing the paper

Every figure and table can be rebuilt two ways. The **notebooks** are the clean, readable version and run on a laptop at quick settings. The **production scripts** are the exact code that made the paper's numbers, with their stored outputs and provenance sidecars.

| Item | Notebook | Production script | Stored output |
|---|---|---|---|
| Figs. 1-4: bands, Berry curvature, quantum metric (NN vs TNN) | `nn_tnn.ipynb` | — | — |
| Fig. 5: second-order channels, four materials | `response.ipynb` | `code/v6/v6jobs/f5/f5.py` (v6 engine) | `code/v6/figure_data/f5_*_240.npz` |
| Fig. 6: k.p spectra | `kp.ipynb` | `code/kp/kp_figs_data.py`, `code/kp/kp_figs.py` | `code/kp/kp_spectra.json`, `kp_table.json` |
| Table II: $v$ and $\kappa$ by Löwdin reduction | `kp.ipynb` (uses `lowdin` in `engine.ipynb`) | `code/verification/ref/selftest.py` checks it | — |
| Fig. 7: van Hove slopes | `field.ipynb` | `code/v7jobs/maps_v7.py` | `code/v7jobs/prod7_*_F.npz` |
| Fig. 8: $\eta(\hbar\omega, B)$ maps | `field.ipynb` | `code/v7jobs/maps_v7.py` | `code/v7jobs/prod7_*_F.npz` |
| Fig. 9: $\eta(B)$ at the band edge | `edge.ipynb` | `code/v7jobs/edge_v7.py` | `code/v7jobs/edge7_*.json` |
| Fig. 10: $\chi$ vs photon energy | `edge.ipynb` | `code/v7jobs/chiw9_v7.py` | `code/v7jobs/chiwR9v7_*.json` |
| Fig. 11: threshold $\chi$ across models | `edge.ipynb` | `code/v7jobs/edge_v7.py`, `code/kp/kp_fp_all_v7.py` | `code/v7jobs/edge7_*.json`, `code/kp/kp_fp_all_v7.json` |
| Fig. 12: dependence on $\lambda$ | `soc.ipynb` | `code/v7jobs/lam_v7.py` | `code/v7jobs/lam7_*_F.npz` |
| Field-term decomposition (Sec. III B) | `field.ipynb` | `code/v7jobs/decomp_v7.py` | `code/v7jobs/decomp7_*.json` |
| Convergence checks | — | `code/v7jobs/checks/c1-c4` | `code/v7jobs/checks/` |

All production figures are drawn by `code/v7jobs/fig_v7_all.py` into `code/v7jobs/figs_v7/`. Each PDF has a `.prov.json` sidecar with the engine hash, the script and its parameters.

## Notes

- `code/kp/kp_params.py` is kept for the record. It imports a module from an earlier development tree and does not run on its own. The same Löwdin fit is the `lowdin` function in `notebooks/engine.ipynb`.
- Fig. 5 was made with the v6-era engine (`code/v6/`), before the v7 production runs. The verification suite's G6 check reads its parameters from there. The claims checks (G1-G3) need a local copy of the paper source.
- Production runs are CPU-bound. Run one process per core with `OMP_NUM_THREADS=1`, as `code/v7jobs/run_v7_queue.sh` does.
