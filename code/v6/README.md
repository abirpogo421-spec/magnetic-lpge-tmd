# v6: Fig. 5 (second-order channels)

Fig. 5 was produced with the v6-era engine, before the v7 production runs. It is kept here so the figure and the verification suite's G6 parameter check can be traced.

| Path | What |
|---|---|
| `v6jobs/f5/f5.py` | Fig. 5 run used for the paper (TNN, GGA, $\mu_B B = 0$ and 80 meV), with the v6 engine files beside it |
| `v6_work_code/runs/f5_extended_omega.py` | the same run over an extended photon-energy window |
| `figure_data/f5_*_240.npz` | stored Fig. 5 data ($N = 240$) |
| `figure_data/f5_base_omega.npz` | photon-energy grid of the zero-field survey run that the scripts extend |

Run from `v6jobs/f5/`: `python3 f5.py MoS2 240 1.0`. The clean, current version of this figure is `notebooks/response.ipynb`.
