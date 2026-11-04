# Magnetic injection current in monolayer TMDs

Code for the paper

> Magnetic activation of the injection current in the linear photogalvanic effect of monolayer transition-metal dichalcogenides
> Abir Mehta and Jyesta Adhidewata (in preparation)

An out-of-plane magnetic field breaks time-reversal symmetry in monolayer $\mathrm{MX_2}$, and under linearly polarized light this turns on a magnetic injection current. We compute it with the three-band tight-binding model of Liu *et al.* (NN and TNN, GGA and LDA) for $\mathrm{MoS_2}$, $\mathrm{MoSe_2}$, $\mathrm{WS_2}$ and $\mathrm{WSe_2}$, including the Zeeman coupling, the orbital moment of the Bloch bands and the field-corrected phase-space measure. A two-band warped-Dirac $\mathbf{k}\cdot\mathbf{p}$ model gives the response near the band edge in closed form.

## Layout

```
notebooks/            the calculation, one notebook per topic
  engine.ipynb        shared code: model, grids, band geometry, currents, Löwdin reduction, paper style
  nn_tnn.ipynb        Figs. 1-4    bands, Berry curvature and quantum metric, NN and TNN
  response.ipynb      Fig. 5       injection and shift current of the four materials
  kp.ipynb            Fig. 6, Tables II and V    Löwdin reduction and the two-band k.p model
  field.ipynb         Figs. 7-8    field dependence over the whole zone
  edge.ipynb          Figs. 9-11   band-edge response and threshold coefficient
  soc.ipynb           dependence on the Ising coupling (not in the current draft)
  data/               results of the long runs at the paper settings
  figures/            the paper's figures as PDFs
reproduce.py          runs the notebooks top to bottom
docs/
  reproducing.md      where each figure, table and number comes from
```

## Getting started

Python 3.10 or newer.

```bash
git clone https://github.com/abirpogo421-spec/magnetic-lpge-tmd.git
cd magnetic-lpge-tmd
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
jupyter lab notebooks/
```

Each notebook starts with `%run engine.ipynb` and runs on its own. The Settings cell near the top holds the paper settings, with a quick setting in the comment next to each line. Runs are cached in `notebooks/data/` under file names that include the settings, so changing one setting reruns only that part.

## Reproducing the figures

```bash
reproduce-lpge                 # the notebooks behind the paper, about 1 h
reproduce-lpge --quick         # the quick settings from the Settings cells, about 2 h
reproduce-lpge field edge      # selected notebooks
reproduce-lpge soc             # the Ising-coupling sweep, not in the paper, about 10 h
```

`python3 reproduce.py` does the same without installing. Each executed notebook is saved to `notebooks/executed/`, and the figures go to `notebooks/figures/`. `--quick` overwrites those figures; `git checkout notebooks/figures` restores them.

The long runs behind Figs. 7-11 take about 70 core-hours, so their results are stored in `notebooks/data/` and a plain run redraws from them. To repeat a run, delete its file. All derivatives are central differences with a step of $10^{-4}/a$, and the zone sums use a slanted $N\times N$ mesh along the reciprocal lattice vectors. The stored results were computed with a script version of this engine on the same grids with the same step; on those grids the notebooks reproduce the stored maps to about $10^{-10}$.

## Conventions

- Physical current: $\eta_{\rm phys} = -2\,(e^3a/\hbar^2)\,\eta_{\rm code}$. The engine sums both spin blocks with prefactor $\pi$; the paper's prefactor $-2\pi e^3a/\hbar^2$ comes from the golden-rule $2\pi$ and the electron charge $-e$.
- Threshold: $\hbar\omega = E_g + 0.01$ eV. Field coefficient $\chi = \partial_B\eta^{x;xx}$ in $\mathrm{A\,m\,V^{-2}s^{-1}T^{-1}}$.
- Field model "F": the Zeeman coupling plus the orbital-moment energy shift, phase-space factor and velocity shift.
- Momenta are in units of $1/a$ and energies in eV. The first-principles k.p model uses Å.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). The three-band model is from G.-B. Liu *et al.*, Phys. Rev. B **88**, 085433 (2013), and the first-principles k.p parameters are from A. Kormányos *et al.*, 2D Mater. **2**, 022001 (2015).

## License

MIT. See `LICENSE`.
