# kp: two-band k.p code (numerical)
- `kp2.py`: two-band warped-Dirac engine (physical units, Angstrom), Kormanyos 2015 table, response(); same kernel and prefactor as the TB engine.
- `kp_params.py` -> `kp_params_TNN_soc.json`: k.p parameters from Lowdin reduction of the Liu TNN model (units of a).
- `kp_fp_all.py` -> `kp_fp_all.json`: chi with first-principles (Kormanyos) parameters; SI = code x 1e-10 e^3/hbar^2, physical sign x(-2) applied in the figure scripts.
- `kp_sweeps_korm.py`, `kp_figs_data.py`, `kp_check.py`: parameter sweeps, spectra (Fig. kp_spectra) and checks.
The symbolic (sympy) derivations are project docs `claude/kp/*.py` and `claude/kp/*.tex`.
Note: these were run in the cloud with paths /home/claude/pub/...; fix sys.path before rerunning on the Mac.
