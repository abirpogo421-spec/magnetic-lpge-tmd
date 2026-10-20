"""Figs. 8 (maps) and 9 (band edge) from the v7 runs, 1x4 each, in the paper style. Run after run_all_v7.sh."""
import json, math, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.labelsize": 8, "ytick.labelsize": 8})
MATS = [('MoS2', r'MoS$_2$'), ('MoSe2', r'MoSe$_2$'), ('WS2', r'WS$_2$'), ('WSe2', r'WSe$_2$')]
UNIT = r"\mathrm{A\,m\,V^{-2}s^{-1}}"; RED, BLUE = "#b3122b", "#1c5aa8"; TW = 7.0
# Fig. 8: maps
fig, ax = plt.subplots(1, 4, figsize=(TW, 2.1), gridspec_kw=dict(wspace=0.55))
for i, (m, lab) in enumerate(MATS):
    d = np.load(f"prod7_{m}_F.npz", allow_pickle=True); n = json.loads(str(d['meta']))['done']
    B, om, Y = d['B_tesla'][:n], d['omega'], d['eta'][:n]
    Bf = np.concatenate([-B[:0:-1], B]); Yf = np.concatenate([-Y[:0:-1], Y]); vm = np.abs(Yf).max()
    im = ax[i].pcolormesh(om, Bf, Yf, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='gouraud', rasterized=True)
    cb = fig.colorbar(im, ax=ax[i], pad=0.03, fraction=0.06); cb.ax.tick_params(labelsize=6, length=2)
    if i == 3: cb.set_label(r"$\eta^{x;xx}$ ($%s$)" % UNIT, fontsize=7)
    ax[i].set_xlabel(r"$\hbar\omega$ (eV)"); ax[i].set_title(lab, loc="right", pad=3); ax[i].xaxis.set_major_locator(MaxNLocator(4))
    ax[i].text(0.0, 1.04, f"({'abcd'[i]})", transform=ax[i].transAxes, fontweight="bold")
ax[0].set_ylabel(r"$B$ (T)"); fig.savefig("lpge_map_grid.pdf", bbox_inches="tight"); fig.savefig("lpge_map_grid.png", dpi=200, bbox_inches="tight")
# Fig. 9: band edge, TNN
D = {m: json.load(open(f"edge7_{m}_TNN.json")) for m, _ in MATS}
def series(m):
    e = D[m]['eta']; B = np.array(sorted(map(float, e))); Y = np.array([np.mean(e[f"{b:g}"]) for b in B])
    return np.concatenate([-B[:0:-1], B]), np.concatenate([-Y[:0:-1], Y])
vmax = max(np.abs(series(m)[1]).max() for m, _ in MATS); p = int(math.floor(math.log10(vmax))); s = 10.0**p
fig, ax = plt.subplots(1, 4, figsize=(TW, 1.95), gridspec_kw=dict(wspace=0.45))
for i, (m, lab) in enumerate(MATS):
    Bf, yf = series(m); a = ax[i]
    a.plot(Bf, yf/s, '-', color=RED if D[m]['chi_SI_per_T'] > 0 else BLUE, lw=1.3)
    a.axhline(0, color="0.85", lw=0.6); a.axvline(0, color="0.85", lw=0.6); a.set_xlabel(r"$B$ (T)"); a.set_title(lab, loc="right", pad=3)
    a.text(0.0, 1.04, f"({'abcd'[i]})", transform=a.transAxes, fontweight="bold"); a.tick_params(top=True, right=True)
ax[0].set_ylabel(r"$\eta^{x;xx}$ ($10^{%d}\,%s$)" % (p, UNIT))
fig.savefig("lpge_edge_sign_grid.pdf", bbox_inches="tight"); fig.savefig("lpge_edge_sign_grid.png", dpi=200, bbox_inches="tight")
json.dump({m: D[m]['chi_SI_per_T'] for m, _ in MATS}, open("threshold_chi_TNN_v7.json", "w"), indent=1)
print("figures written")
