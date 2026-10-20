"""Figures 6 (k.p scaling) and 7 (k.p spectra) and the appendix k.p table from saved JSON data."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11, "axes.linewidth": 0.9,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.labelsize": 9, "ytick.labelsize": 9})
import os; D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "figs"); os.makedirs(OUT, exist_ok=True)
E = 1.602176634e-19; HB = 1.054571817e-34
T = json.load(open(f"{D}/kp_sweeps.json")); K = json.load(open(f"{D}/kp_sweeps_korm.json"))
bT, bK = T['base'], K['base']; aT = 3.19
RED, BLUE = "#b3122b", "#1c5aa8"

def col(S, n, j=1): return np.array(S[n]['x']), np.array(S[n]['chi'])[:, j]
fig, ax = plt.subplots(1, 4, figsize=(14, 3.5))
specs = [('kappa', r'$\kappa/\kappa_0$', lambda x, b: x/b['kappa']),
         ('v', r'$v/v_0$', lambda x, b: x/b['v']),
         ('mD', r'$2m_D$ (eV)', lambda x, b: 2*x),
         ('lam', r'$\lambda$ (eV)', lambda x, b: x)]
for i, (n, lab, fx) in enumerate(specs):
    a = ax[i]
    for S, b, c, ls, name in ((T, bT, RED, '-', 'TB-derived (TNN)'), (K, bK, BLUE, '--', 'first-principles')):
        x, y = col(S, n); o0 = np.argsort(x); y0 = np.interp(b[n], x[o0], y[o0])
        if n == 'v' and S is T:            # below 0.75 v0 the TB-derived edge leaves K (v^2 + 2 w m_D < 0)
            keep = x >= 0.75*b['v']; x, y = x[keep], y[keep]
        o = np.argsort(fx(x, b)); a.plot(fx(x, b)[o], (y/y0)[o], ls, color=c, lw=1.5, label=name)
        a.plot([fx(b[n], b)], [1.0], 'o', color=c, ms=4)
    a.axhline(0, color="0.85", lw=0.6); a.set_xlabel(lab); a.tick_params(top=True, right=True)
    a.text(-0.01, 1.04, f"({'abcd'[i]})", transform=a.transAxes, fontweight="bold", fontsize=12)
ax[0].set_ylabel(r"$\chi/\chi_0$"); ax[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout(w_pad=1.2); fig.savefig(f"{OUT}/lpge_scaling_grid.pdf"); fig.savefig(f"{OUT}/lpge_scaling_grid.png", dpi=160); plt.close(fig)

# Fig 7 spectra (TB-derived MoS2)
S = json.load(open(f"{D}/kp_spectra.json")); om = np.array(S['omega']); si = -2.0*E**3*(aT*1e-10)/HB**2   # physical: eta_phys = -2 eta_code (check A20)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.7))
a = ax[0]; W = om <= S['Eg']+0.45          # k.p disk (R = 0.6/a) is complete only below ~Eg+0.5 eV
a.plot(om[W], np.array(S['K_B0']['lpge'])[W]*si, '--', color=BLUE, lw=1.3, label=r"single valley $K$, $B=0$ (left)")
a.plot(om[W], np.array(S['KKp_B0']['lpge'])[W]*si, color="0.55", lw=1.3, label=r"$K+K'$, $B=0$ (left)")
a.set_ylabel(r"$\eta^{x;xx}$  ($\mathrm{A\,m\,V^{-2}s^{-1}}$)")
a2 = a.twinx(); a2.plot(om[W], np.array(S['KKp_B30']['lpge'])[W]*si, color=RED, lw=1.5, label=r"$K+K'$, $B=30$ T (right)")
a2.set_ylabel(r"$\eta^{x;xx}_{K+K'}(30\,\mathrm{T})$", color=RED); a2.tick_params(axis='y', colors=RED, direction='in')
for ax_, key in ((a, 'K_B0'), (a2, 'KKp_B30')):
    y_ = np.array(S[key]['lpge'])[W]*si; top = 1.08*np.abs(y_).max(); ax_.set_ylim(*((-0.08*top, top) if y_.max() > -y_.min() else (-top, 0.08*top)))
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1+h2, l1+l2, fontsize=8, frameon=False, loc="upper right")
b_ = ax[1]
b_.plot(om[W], np.array(S['K_B0']['shift'])[W], '--', color=BLUE, lw=1.3, label=r"single valley $K$, $B=0$")
b_.plot(om[W], np.array(S['KKp_B0']['shift'])[W], color="0.55", lw=2.4, label=r"$K+K'$, $B=0$")
b_.plot(om[W], np.array(S['KKp_B30']['shift'])[W], color=RED, lw=1.0, label=r"$K+K'$, $B=30$ T")
b_.set_ylabel(r"$\sigma^{y;yy}$  (model units)")
for i, a in enumerate(ax):
    a.axhline(0, color="0.85", lw=0.6); a.set_xlabel(r"$\hbar\omega$ (eV)"); a.tick_params(top=True)
    a.text(-0.01, 1.04, f"({'ab'[i]})", transform=a.transAxes, fontweight="bold", fontsize=12)
b_.legend(fontsize=8, frameon=False)
fig.tight_layout(w_pad=1.5); fig.savefig(f"{OUT}/kp_spectra.pdf"); fig.savefig(f"{OUT}/kp_spectra.png", dpi=160); plt.close(fig)

tab = json.load(open(f"{D}/kp_table.json"))
print(json.dumps(tab, indent=1))
for n in ('kappa', 'v', 'mD', 'lam'):
    for S, b, nm in ((T, bT, 'TB'), (K, bK, 'FP')):
        x, y = col(S, n); o0 = np.argsort(x); y0 = np.interp(b[n], x[o0], y[o0]); print(n, nm, 'chi/chi0 range', np.round((y/y0).min(), 3), np.round((y/y0).max(), 3))
