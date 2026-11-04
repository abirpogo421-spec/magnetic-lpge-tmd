# Reproducing the paper

Every figure and table comes from one notebook. `python3 reproduce.py` runs them all.

| Item | Notebook | Stored run (`notebooks/data/`) |
|---|---|---|
| Figs. 1-4: bands, Berry curvature, quantum metric (NN and TNN) | `nn_tnn.ipynb` | none, about a minute |
| Fig. 5: injection and shift current, four materials | `response.ipynb` | none, about an hour |
| Fig. 6: k.p spectra | `kp.ipynb` | none, a few minutes |
| Table II: $v$ and $\kappa$ from the Löwdin reduction | `kp.ipynb` | none |
| Table V: k.p tensor components at $E_g + 0.1$ eV | `kp.ipynb` | none |
| Fig. 7: van Hove slopes | `field.ipynb` | `map_*.npz` |
| Fig. 8: $\eta(\hbar\omega, B)$ maps | `field.ipynb` | `map_*.npz` |
| Field-term decomposition (Sec. III B) | `field.ipynb` | `decomp_*.npz` |
| Fig. 9: $\eta(B)$ at the band edge | `edge.ipynb` | `edge_*.npz` |
| Fig. 10: $\chi$ against photon energy | `edge.ipynb` | `chi_omega_*.npz` |
| Fig. 11: threshold $\chi$ across models | `edge.ipynb` | `edge_*.npz` |
| Moment at $K$, largest $\|D_n - 1\|$, phase-space change of threshold $\chi$ | `edge.ipynb` | `phase_space_*.npz` |
| Convergence in $N$ and $h$ | `field.ipynb`, `edge.ipynb` | `convergence_*.npz` |

## Notes

- The stored runs were made with a script version of the engine, which uses the same slanted mesh, valley disks and central-difference step. On the same grids the notebooks reproduce the stored maps to about $10^{-10}$ and the band-edge values to about $10^{-6}$ or better.
- To repeat a stored run, delete its file and rerun the notebook. Use one process per core with `OMP_NUM_THREADS=1`, as `reproduce.py` does.
