# Magnetic injection current in monolayer TMDs

Code for the paper

> **Magnetic activation of the injection current in the linear photogalvanic effect of monolayer transition-metal dichalcogenides**
> Abir Mehta and Jyesta Adhidewata (in preparation)

An out-of-plane magnetic field breaks time-reversal symmetry in monolayer $\mathrm{MX_2}$ and switches on a magnetic injection current under linearly polarized light. The code computes this response with the three-band tight-binding model of Liu *et al.* (NN and TNN, GGA and LDA) for $\mathrm{MoS_2}$, $\mathrm{MoSe_2}$, $\mathrm{WS_2}$ and $\mathrm{WSe_2}$. It includes the Zeeman coupling, the orbital moment of the Bloch bands and the field-corrected phase-space measure. A two-band warped-Dirac $\mathbf{k}\cdot\mathbf{p}$ reduction gives the near-edge response in closed form. Every number in the paper is checked by an independent verification suite.

## Repository layout

```
notebooks/            clean, self-contained notebooks, one per topic (start here)
  engine.ipynb        shared setup: constants, Liu parameters, Hamiltonians, grids, field terms, Löwdin reduction
  nn_tnn.ipynb        Figs. 1-4   bands, Berry curvature and quantum metric, NN vs TNN
  response.ipynb      Fig. 5      second-order channels of the four materials
  kp.ipynb            Fig. 6, Table II   two-band k.p model and trigonal warping
  field.ipynb         Figs. 7-8   field dependence over the full zone
  edge.ipynb          Figs. 9-11  band-edge response and threshold coefficient
  soc.ipynb           Fig. 12     dependence on the Ising coupling
code/
  pub_pipeline/       production engine: tb_core.py, tmd_params.py, lpge.py
  kp/                 two-band k.p engine (kp2.py) and first-principles k.p runs
  v7jobs/             production run scripts, results (JSON/NPZ) and figure script for Figs. 7-12
  verification/       independent reference implementation and test suite (see VERIFICATION.md)
docs/
  reproducing.md      which script or notebook produces each figure, table and number
```

## Getting started

Python 3.10 or newer.

```bash
git clone https://github.com/abirpogo421-spec/magnetic-lpge-tmd.git
cd magnetic-lpge-tmd
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Each topic notebook starts with `%run engine.ipynb` and then computes its own figures. The **Resolution** cell near the top of every notebook holds the paper settings, with a quick setting in the comment beside each line. Results are cached in `notebooks/data/` under file names that include the resolution, so changing a setting starts a fresh run for that section only. `nn_tnn.ipynb` takes about 20 s. The band-edge runs at paper resolution take hours.

## Production pipeline

The paper's Figs. 7-12 come from the production engine in `code/pub_pipeline/` (engine hash `ff4a692d…`, exact Lorentzian sum), driven by the scripts in `code/v7jobs/`:

```bash
cd code/v7jobs
bash run_v7_queue.sh          # all production runs (many hours; set OMP_NUM_THREADS=1 per process)
python3 fig_v7_all.py --data . --kp ../kp/kp_fp_all_v7.json --out figs_v7
```

The results of the paper runs are already in `code/v7jobs/` (`prod7_*.npz`, `edge7_*.json`, `chiwR9v7_*.json`, `decomp7_*.json`) and the figures in `code/v7jobs/figs_v7/`, each with a `.prov.json` sidecar recording the engine hash, script and parameters. The first-principles k.p threshold coefficients come from `code/kp/kp_fp_all_v7.py` (seconds).

## Verification

```bash
cd code/verification
python3 verify_all.py --fast      # a few minutes
python3 verify_all.py --full      # production grids and convergence (hours)
python3 verify_all.py --mutants   # planted-bug test
```

The suite compares the engine against a reference implementation written only from the paper and Liu *et al.*, reruns short copies of every production script, checks symmetry, units and convergence, and checks each printed number against `claims.json`. `VERIFICATION.md` explains every check. The claims checks need a local copy of the paper source (`paths.paper_root` in `config.json`). Without it they are skipped, and G6 reports four Fig. 5 parameters as unchecked because the v6-era Fig. 5 scripts are not included (see `docs/reproducing.md`).

## Conventions

- Physical current: $\eta_{\rm phys} = -2\,\eta_{\rm code}$ (spin sum and electron charge).
- Threshold: $\hbar\omega = E_g + 0.01$ eV. Field coefficient $\chi = \partial_B\eta^{x;xx}$ in $\mathrm{A\,m\,V^{-2}s^{-1}T^{-1}}$.
- Field model "F": Zeeman coupling plus the orbital-moment energy shift, phase-space factor and velocity shift.
- In the notebooks momenta are in units of $1/a$ and energies in eV. `kp/kp2.py` uses Å.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). The three-band model is from G.-B. Liu *et al.*, Phys. Rev. B **88**, 085433 (2013), and the first-principles k.p parameters from A. Kormányos *et al.*, 2D Mater. **2**, 022001 (2015).

## License

MIT. See `LICENSE`.
