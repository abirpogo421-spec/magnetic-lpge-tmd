# v7jobs: reruns for Figs. 8 and 9 (prepared Oct 13, NOT run yet)

Run from this folder on the Mac (it imports `../pub_pipeline`):
`nohup bash run_all_v7.sh > run_all_v7.log 2>&1 &`, then `python3 fig_v7.py`.

| Script | Purpose | Settings | Output |
|---|---|---|---|
| `maps_v7.py` | Fig. 8 maps, eta(omega, B) | TNN GGA, model F, N = 600, 900 omega points from 0.95 E_g to 1.05 x largest interband energy, B = 0..60 T in 1 T steps (61 fields, mirrored to 121), nu = 10 meV | `prod7_{m}_F.npz` |
| `edge_v7.py` | Fig. 9 band edge, eta(B) at E_g + 0.01 eV, plus threshold chi for Fig. 11 | TNN and NN, local disks radius 0.9/a (same as Fig. 10), spacing 0.0007/a, nu = 5 meV, eta averaged over E_g + 0.0075..0.0125 eV (5 points), B = 0..60 T in 2 T steps | `edge7_{m}_{model}.json` (with chi and % deviation from linear at 60 T) |
| `fig_v7.py` | Redraws Figs. 8 and 9 as 1x4 strips | | `lpge_map_grid.pdf`, `lpge_edge_sign_grid.pdf`, `threshold_chi_TNN_v7.json` |

Why these changes:
- Old maps: 21 fields (3 T steps) and omega from E_g - 0.15 to E_g + 2.6 eV. The new run uses 1 T steps and the original window (0.95 E_g to 1.05 x largest gap); N lowered 900 -> 600 to pay for the extra fields (N = 240 already agrees with N = 480 to 0.2%).
- Old band-edge run: radius 0.6/a, spacing 0.001/a, single omega -> visible wiggle in WS2. The finer spacing plus 5 meV omega averaging removes it, and radius 0.9/a makes Figs. 9, 10 and 11 use the same grid.
- After the run, update the Fig. 11 TNN/NN points and the threshold numbers in the text from `edge7_*.json` (they may shift slightly for TNN MoSe2 and WS2).

## v7 runs, done 2026-10-14 (fixed engine lpge.py sha256 ff4a692d..., exact Lorentzian sum)
Launched by `run_v7_queue.sh` (23:49 to 07:47 PT). All outputs record the engine hash, script and parameters.
| Script | Figure / item | Output |
|---|---|---|
| `maps_v7.py` | Fig. 8 maps and Fig. 7 (van Hove = omega of max abs(eta) at 60 T; slope = eta(1 T)/1 T) | `prod7_{m}_F.npz` |
| `edge_v7.py` (GGA, 2 T steps; LDA, 10 T steps) | Fig. 9, Fig. 11, threshold numbers | `edge7_{m}_{TNN,NN}[_LDA].json` |
| `chiw9_v7.py` | Fig. 10 | `chiwR9v7_{m}_{model}_F_0.001.json` |
| `lam_v7.py` (R = 0.9/a) | Fig. 12 | `lam7_{m}_F.npz` (chi in CODE sign; x(-2) for physical) |
| `decomp_v7.py` (vH from prod7, guarded against partial maps) | field-term decomposition | `decomp7_{m}.json` |
| `fig_v7_all.py` | draws Figs. 7-12 + Fig. 9 residual check, provenance sidecars | `figs_v7/*.pdf`, `*.prov.json`, `numbers_v7.json` |
| `checks/c1..c4` | engine fix vs independent exact values; N, h convergence; full-BZ hotspots (velocity-shift divergence in WSe2) | `checks/*.json`, `*.out` |
