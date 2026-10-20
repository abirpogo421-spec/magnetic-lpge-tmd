#!/usr/bin/env python3
"""Figs. 7-12 of the paper from the v7 reruns (fixed engine lpge.py, sha256 ff4a692d...), in the shared paper style.

Run from the v7jobs folder (all inputs are relative):
    python3 fig_v7_all.py                       # all figures that have inputs; outputs in ./figs_v7/
    python3 fig_v7_all.py --only 9 11           # selected figures
    python3 fig_v7_all.py --data . --kp ../kp/kp_fp_all_v7.json --out figs_v7

Inputs (physical sign and SI units unless noted):
  Fig. 7  prod7_{m}_F.npz (eta(B) at the van Hove energy = omega of max |eta| at 60 T); optional decomp7_{m}.json (decomp_v7.py)
  Fig. 8  prod7_{m}_F.npz                 maps_v7.py   B_tesla (61), omega (900), eta[B, omega], meta JSON
  Fig. 9  edge7_{m}_TNN.json              edge_v7.py   eta["B"] = 5 omegas in Eg+0.0075..0.0125 eV; chi_SI_per_T; dev_60T_pct
          edge7_{m}_NN.json               (residual check and Fig. 11 only)
  Fig. 10 chiwR9v7_{m}_{TNN,NN}_F_0.001.json  chiw9_v7.py  off, chi
  Fig. 11 edge7_{m}_{TNN,NN}.json (GGA), edge7_{m}_{TNN,NN}_LDA.json (LDA, 10 T steps), ../kp/kp_fp_all_v7.json
  Fig. 12 lam7_{m}_F.npz                  lam_v7.py    lam, Eg, chi_SI_per_T [CODE sign: x -2 for physical], meta.lam_liu
Outputs (in --out):
  lpge_field_merged.pdf (Fig. 7), lpge_map_grid.pdf (Fig. 8), lpge_edge_sign_grid.pdf (Fig. 9), fig9_residuals.pdf (acceptance check),
  chi_vs_offset.pdf (Fig. 10), threshold_models.pdf (Fig. 11), lpge_lambda_grid.pdf (Fig. 12),
  a .png preview and a <name>.prov.json provenance sidecar for every PDF (same record in the PDF Subject field),
  numbers_v7.json (every number the text quotes from these figures).
"""
import os, sys, json, math, hashlib, datetime, platform, argparse, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ shared paper style (= v6_work/figure_data/scripts/style.py)
TW, CW = 7.0, 3.4
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.4,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "image.interpolation": "nearest",
})
MAT = [("MoS2", r"MoS$_2$"), ("MoSe2", r"MoSe$_2$"), ("WS2", r"WS$_2$"), ("WSe2", r"WSe$_2$")]
MATS = [m for m, _ in MAT]
COL = {"MoS2": "#0072B2", "MoSe2": "#D55E00", "WS2": "#009E73", "WSe2": "#CC79A7"}
BLUE, RED, GREY, ORANGE = "#0072B2", "#C0392B", "0.55", "#E69F00"
UNIT = r"\mathrm{A\,m\,V^{-2}\,s^{-1}}"
LET = "abcd"
PHYS = -2.0                      # code sign -> physical sign (only lam7 files are stored in code sign)
EXPECT_ENGINE = "ff4a692d"       # prefix of the fixed engine lpge.py sha256

# Table II trigonal warping kappa (eV A^2), active (band-edge) spin pair
KAPPA_TB = {"TNN": {"MoS2": -0.2365, "MoSe2": -0.0215, "WS2": -0.0223, "WSe2": +0.0848},
            "NN":  {"MoS2": -0.043,  "MoSe2": +0.0265, "WS2": -0.512,  "WSe2": -0.263}}
# LDA parameter sets, from v6_work/results_numbers/lda_kappa_thr.json "k_l" (identical in lda_kappa.json); order TNN x4, NN x4
KAPPA_LDA = {"TNN": dict(zip(MATS, [0.008, 0.456, 0.269, 0.618])), "NN": dict(zip(MATS, [-0.071, 0.07, -0.677, -0.314]))}
# Liu et al. GGA Ising couplings (eV); B transition onset = 2 lambda above E_g (Bonset.json: 0.146, 0.182, 0.422, 0.456)
def _liu_lambda():
    """Liu SOC lambda (eV) from the engine's own parameter file (single source of literature inputs)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pub_pipeline"))
    import tmd_params as _tp
    return {m: float(_tp.LAMBDA[m]) for m in ("MoS2", "MoSe2", "WS2", "WSe2")}
LAM_LIU = _liu_lambda()


def label(ax, s, x=0.0, y=1.02, **kw):
    kw.setdefault("va", "bottom")
    ax.text(x, y, r"$\mathbf{%s}$" % s, transform=ax.transAxes, fontsize=9, ha="left", **kw)


def zero(ax, v=False):
    ax.axhline(0, color="0.8", lw=0.5, zorder=0)
    if v: ax.axvline(0, color="0.8", lw=0.5, zorder=0)


def mirror(B, Y):
    """B >= 0 data -> full field range using eta(-B) = -eta(B)."""
    B = np.asarray(B, float); Y = np.asarray(Y, float)
    return np.concatenate([-B[:0:-1], B]), np.concatenate([-Y[:0:-1], Y])


def pow10(v):
    return int(math.floor(math.log10(v))) if v > 0 else 0


def rjson(path):
    with open(path) as f: return json.load(f)


def wjson(obj, path, **kw):
    with open(path, "w") as f: json.dump(obj, f, **kw)


# ------------------------------------------------------------------ provenance
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""): h.update(blk)
    return h.hexdigest()


class Prov:
    """Collects inputs (path, sha256, engine hash from the file's own metadata) for each figure."""
    def __init__(self, args):
        self.script = os.path.abspath(__file__); self.script_sha = sha256(self.script); self.args = args
        self.inputs = {}       # path -> dict

    def add(self, path, engine=None, kind="lpge_sha256"):
        p = os.path.normpath(path)
        if p not in self.inputs:
            self.inputs[p] = dict(path=p, sha256=sha256(p), engine_key=kind, engine_sha256=engine)
        return p

    def record(self, name, paths, extra=None):
        ins = [self.inputs[os.path.normpath(p)] for p in paths]
        eng = sorted({i["engine_sha256"] for i in ins if i["engine_key"] == "lpge_sha256" and i["engine_sha256"]})
        aux = sorted({f'{i["engine_key"]}={i["engine_sha256"]}' for i in ins if i["engine_key"] != "lpge_sha256" and i["engine_sha256"]})
        missing = [i["path"] for i in ins if i["engine_key"] == "lpge_sha256" and not i["engine_sha256"]]
        rec = dict(figure=name, date=datetime.datetime.now().isoformat(timespec="seconds"),
                   script=os.path.basename(self.script), script_sha256=self.script_sha,
                   engine_sha256=eng[0] if len(eng) == 1 else eng, engine_consistent=len(eng) <= 1,
                   engine_expected_prefix=EXPECT_ENGINE,
                   engine_matches_expected=bool(eng) and all(e.startswith(EXPECT_ENGINE) for e in eng),
                   other_code_hashes=aux, inputs_without_engine_hash=missing,
                   inputs=[dict(path=i["path"], sha256=i["sha256"], engine_sha256=i["engine_sha256"]) for i in ins],
                   python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                   cwd=os.getcwd())
        if extra: rec.update(extra)
        if not rec["engine_consistent"]: warnings.warn(f"{name}: inputs come from different engine versions: {eng}")
        if eng and not rec["engine_matches_expected"]: warnings.warn(f"{name}: engine hash {eng} does not start with {EXPECT_ENGINE}")
        if missing: warnings.warn(f"{name}: no engine hash in {missing}")
        return rec


def save(fig, name, rec, out, png=True):
    js = json.dumps(rec, separators=(",", ":"))
    pdf = os.path.join(out, name + ".pdf")
    fig.savefig(pdf, metadata={"Title": name, "Subject": js, "Creator": rec["script"] + " sha256=" + rec["script_sha256"][:16],
                               "Keywords": "engine=" + str(rec["engine_sha256"])})
    if png: fig.savefig(os.path.join(out, name + ".png"), dpi=200, metadata={"Description": js})
    rec = dict(rec, output=os.path.basename(pdf), output_sha256=sha256(pdf))
    wjson(rec, os.path.join(out, name + ".prov.json"), indent=1)
    plt.close(fig); print(f"  wrote {pdf}  (+ .prov.json{', .png' if png else ''})")


# ------------------------------------------------------------------ loaders
def load_npz(path, prov):
    d = np.load(path, allow_pickle=True); meta = json.loads(str(d["meta"]))
    prov.add(path, meta.get("lpge_sha256")); return d, meta


def load_json(path, prov, key="lpge_sha256"):
    d = rjson(path)
    eng = d.get(key) if key in d else d.get("meta", {}).get(key)
    prov.add(path, eng, key); return d


def edge_series(d):
    """edge7 json -> (B >= 0, eta mean over the 5 omegas, all 5 omegas)."""
    e = d["eta"]; B = np.array(sorted(map(float, e)))
    Y5 = np.array([e[f"{b:g}"] for b in B], float)
    return B, Y5.mean(axis=1), Y5


def odd_fit(B, Y):
    """eta = s B + c B^3, least squares over the mirrored data (same as edge_v7.py)."""
    Bf, Yf = mirror(B, Y)
    s, c = np.linalg.lstsq(np.vstack([Bf, Bf**3]).T, Yf, rcond=None)[0]
    return float(s), float(c)


def crossings(x, y, xmin=None):
    """Zero crossings of y(x) by linear interpolation (excluding exact zeros at the end points)."""
    s = np.sign(y); j = np.where((s[1:] != s[:-1]) & (s[1:] != 0) & (s[:-1] != 0))[0]
    xs = [float(x[i] - y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])) for i in j]
    return [v for v in xs if xmin is None or v > xmin]


def r2(y, yf):
    return float(1 - np.sum((y-yf)**2)/np.sum((y-np.mean(y))**2))


def need(paths):
    miss = [p for p in paths if not os.path.exists(p)]
    if miss: print("  SKIP: missing " + ", ".join(miss))
    return not miss


# ------------------------------------------------------------------ Fig. 7: eta(B) at the van Hove peak (single column) + Sec. III B numbers
def fig7(A, prov, NUM):
    files = [os.path.join(A.data, f"prod7_{m}_F.npz") for m in MATS]
    if not need(files): return
    fig, a = plt.subplots(figsize=(CW, 2.45)); info = {}
    for i, (m, lab) in enumerate(MAT):
        d, meta = load_npz(files[i], prov); n = int(meta.get("done", len(d["B_tesla"])))
        B, om, Y = d["B_tesla"][:n].astype(float), d["omega"], np.asarray(d["eta"])[:n]
        if n < len(d["B_tesla"]) or B.max() < 60: warnings.warn(f"{files[i]}: partial run, fields to {B.max():g} T; vH taken at the largest field")
        jv = int(np.argmax(np.abs(Y[-1]))); vh = float(om[jv]); y = Y[:, jv]            # vH = omega of max |eta| at the largest field
        Bf, yf = mirror(B, y); a.plot(Bf, yf, color=COL[m], label=lab + r" ($%.2f$ eV)" % vh)
        k20 = np.abs(Bf) <= 20 + 1e-9
        s20 = float(np.linalg.lstsq(Bf[k20, None], yf[k20], rcond=None)[0][0])             # paper definition: eta = s B over |B| <= 20 T
        s3, c3 = odd_fit(B, y)                                                              # eta = s B + c B^3 over all B
        b1 = B[B > 0].min()                                                                 # initial slope: eta(B1)/B1 at the smallest field (1 T)
        s0 = float(np.interp(b1, B, y)/b1) if b1 <= 2.0 + 1e-9 else odd_fit(B[B <= 20 + 1e-9], y[B <= 20 + 1e-9])[0]
        e = lambda b: float(np.interp(b, B, y))
        dev = 100*(y[B > 0]/(s0*B[B > 0]) - 1)                                                    # deviation from the initial linear law
        first = lambda thr: next((float(b) for b, v in zip(B[B > 0], dev) if abs(v) > thr), None)
        info[m] = dict(vH_eV=vh, omega_index=jv, Eg=meta.get("Eg"), chi_vH_s20_SI_per_T=s20, chi_vH_cubic_all_B=dict(s=s3, c=c3, dev_60T_pct=float(100*c3*3600/s3)),
                       chi_vH_initial_s0=s0, eta_60T=e(60.0) if B.max() >= 60 else None, abs_eta_60T=abs(e(60.0)) if B.max() >= 60 else None,
                       pct_below_linear_extrap_60T_vs_s20=float(100*(1 - e(60.0)/(s20*60))) if B.max() >= 60 else None,
                       pct_below_linear_extrap_60T_vs_s0=float(100*(1 - e(60.0)/(s0*60))) if B.max() >= 60 else None,
                       dev_from_linear_20T_pct_vs_s0=float(100*(e(20.0)/(s0*20) - 1)), dev_from_linear_20T_pct_vs_s20=float(100*(e(20.0)/(s20*20) - 1)),
                       field_where_dev_exceeds_1pct_T=first(1.0), field_where_dev_exceeds_2pct_T=first(2.0), field_where_dev_exceeds_5pct_T=first(5.0))
    zero(a, v=True); a.set_xlabel(r"$B$ (T)"); a.set_ylabel(r"$\eta^{x;xx}$ ($%s$)" % UNIT); a.set_xlim(-62, 62); a.legend(loc="upper left")
    a.xaxis.set_major_locator(MaxNLocator(7, steps=[1, 2, 2.5, 5, 10]))
    # decomposition (decomp_v7.py) if available
    dec = {}
    for m in MATS:
        f = os.path.join(A.data, f"decomp7_{m}.json")
        if not os.path.exists(f): dec[m] = "pending (decomp7_%s.json not found)" % m; continue
        dd = load_json(f, prov); g = lambda k: dd.get(f"{m}|{k}")
        r = dict(vH_eV_used_in_decomp=dd.get("meta", {}).get("vH_eV"), vH_eV_fig7=info[m]["vH_eV"])
        if g("vH|F") is not None and g("vH|Z") is not None: r["vH_F_over_Z"] = g("vH|F")/g("vH|Z")
        if g("edge|D") is not None and g("edge|Z") is not None: r["edge_D_vs_Z_pct"] = 100*(g("edge|D") - g("edge|Z"))/g("edge|Z")
        r["all"] = {k.split("|", 1)[1]: v for k, v in dd.items() if k.startswith(m + "|")}
        if r["vH_eV_used_in_decomp"] is not None and abs(r["vH_eV_used_in_decomp"] - info[m]["vH_eV"]) > 0.01:
            r["note"] = "decomp_v7.py vH energy differs from the Fig. 7 vH by more than 10 meV"
            warnings.warn(f"{m}: decomp vH {r['vH_eV_used_in_decomp']} eV vs Fig. 7 vH {info[m]['vH_eV']:.3f} eV")
        dec[m] = r
    print("  Fig. 7 / Sec. III B (chi in 1e-2 A m V^-2 s^-1 T^-1):")
    for m in MATS:
        v = info[m]
        print(f"    {m:6s} vH {v['vH_eV']:.3f} eV  s20 {v['chi_vH_s20_SI_per_T']*1e2:+.3f}  s(cubic, all B) {v['chi_vH_cubic_all_B']['s']*1e2:+.3f}"
              f"  |eta(60)| {v['abs_eta_60T']:.3f}  below lin. extrap. at 60 T {v['pct_below_linear_extrap_60T_vs_s20']:+.1f}% (vs s20) {v['pct_below_linear_extrap_60T_vs_s0']:+.1f}% (vs s0)"
              f"  dev at 20 T {v['dev_from_linear_20T_pct_vs_s0']:+.2f}% (vs s0)  >1% from {v['field_where_dev_exceeds_1pct_T']} T, >2% from {v['field_where_dev_exceeds_2pct_T']} T")
        print(f"           decomp: {dec[m] if isinstance(dec[m], str) else {k: v_ for k, v_ in dec[m].items() if k != 'all'}}")
    q = [info[m] for m in MATS]
    NUM["fig7_secIIIB"] = dict(
        definitions=dict(vH="omega of max |eta| at the largest field (60 T), same as fig_main.py / decomp_v7.py",
                         chi_vH_s20="least squares eta = s B over |B| <= 20 T (paper definition)",
                         chi_vH_cubic_all_B="eta = s B + c B^3 over all |B| <= 60 T", s0="initial slope eta(1 T)/1 T (smallest field; cubic fit over |B| <= 20 T if the smallest field is > 2 T)",
                         pct_below_linear_extrap_60T="100 (1 - eta(60 T)/(s 60 T))", dev_from_linear="100 (eta(B)/(s0 B) - 1)"),
        per_material=info,
        ranges=dict(vH_eV=[min(v["vH_eV"] for v in q), max(v["vH_eV"] for v in q)],
                    chi_vH_s20_SI_per_T=[min(v["chi_vH_s20_SI_per_T"] for v in q), max(v["chi_vH_s20_SI_per_T"] for v in q)],
                    abs_eta_60T=[min(v["abs_eta_60T"] for v in q if v["abs_eta_60T"] is not None), max(v["abs_eta_60T"] for v in q if v["abs_eta_60T"] is not None)] if all(v["abs_eta_60T"] is not None for v in q) else None,
                    pct_below_60T_vs_s20=[min(v["pct_below_linear_extrap_60T_vs_s20"] for v in q), max(v["pct_below_linear_extrap_60T_vs_s20"] for v in q)] if all(v["pct_below_linear_extrap_60T_vs_s20"] is not None for v in q) else None,
                    max_abs_dev_20T_pct_vs_s0=max(abs(v["dev_from_linear_20T_pct_vs_s0"]) for v in q)),
        decomposition=dec,
        decomposition_summary=dict(
            vH_F_over_Z=[dec[m]["vH_F_over_Z"] for m in MATS] if all(isinstance(dec[m], dict) and "vH_F_over_Z" in dec[m] for m in MATS) else "pending",
            edge_D_vs_Z_pct=[dec[m]["edge_D_vs_Z_pct"] for m in MATS] if all(isinstance(dec[m], dict) and "edge_D_vs_Z_pct" in dec[m] for m in MATS) else "pending"))
    used = files + [os.path.join(A.data, f"decomp7_{m}.json") for m in MATS if isinstance(dec[m], dict)]
    save(fig, "lpge_field_merged", prov.record("Fig. 7 lpge_field_merged", used), A.out, A.png)


# ------------------------------------------------------------------ Fig. 8: omega-B maps (1x4, per-panel colour bars)
def fig8(A, prov, NUM):
    files = [os.path.join(A.data, f"prod7_{m}_F.npz") for m in MATS]
    if not need(files): return
    fig, ax = plt.subplots(1, 4, figsize=(TW, 1.95), gridspec_kw=dict(wspace=0.62))
    info = {}
    for i, (m, lab) in enumerate(MAT):
        d, meta = load_npz(files[i], prov); n = int(meta.get("done", len(d["B_tesla"])))
        if n < len(d["B_tesla"]): warnings.warn(f"{files[i]}: partial run, {n}/{len(d['B_tesla'])} fields")
        B, om, Y = d["B_tesla"][:n], d["omega"], np.asarray(d["eta"])[:n]
        Bf, Yf = mirror(B, Y); vm = float(np.abs(Yf).max()); a = ax[i]
        im = a.pcolormesh(om, Bf, Yf, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="nearest", rasterized=True)
        cb = fig.colorbar(im, ax=a, pad=0.03, fraction=0.07, aspect=22)
        cb.ax.tick_params(labelsize=6, width=0.5, length=2, direction="out"); cb.outline.set_linewidth(0.5); cb.ax.minorticks_off()
        cb.locator = MaxNLocator(6, symmetric=True); cb.update_ticks()
        if i == 3: cb.set_label(r"$\eta^{x;xx}$ ($%s$)" % UNIT, fontsize=7, labelpad=2)
        a.set_xlabel(r"$\hbar\omega$ (eV)"); a.set_title(lab, pad=3, loc="right"); label(a, f"({LET[i]})")
        a.tick_params(top=False, right=False, which="both"); a.xaxis.set_major_locator(MaxNLocator(4))
        a.set_ylim(Bf.min(), Bf.max()); a.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 3, 5, 6, 10]))
        info[m] = dict(Eg=meta.get("Eg"), Emax_interband=meta.get("Emax_interband"), n_fields=int(n), n_omega=int(len(om)),
                       omega_range_eV=[float(om[0]), float(om[-1])], max_abs_eta=vm, N=meta.get("N"), nu_eV=meta.get("nu_eV"),
                       odd_check_B0_max=float(np.abs(Y[0]).max()) if B[0] == 0 else None,
                       banding_max_d2B_over_max_eta=float(np.abs(np.diff(Y, 2, axis=0)).max()/vm) if n > 2 else None,
                       banding_max_d1B_over_max_eta=float(np.abs(np.diff(Y, 1, axis=0)).max()/vm) if n > 1 else None)
        print(f"  Fig. 8 {m:6s} omega {om[0]:.3f}-{om[-1]:.3f} eV ({len(om)} pts, Eg {meta.get('Eg', float('nan')):.3f}, 1.05 Emax {1.05*meta.get('Emax_interband', float('nan')):.3f})"
              f"  max|eta| {vm:.3f}  max|d2eta/dB2| step {info[m]['banding_max_d2B_over_max_eta']:.2e} of max  eta(B=0) max {info[m]['odd_check_B0_max']:.1e}")
    ax[0].set_ylabel(r"$B$ (T)")
    NUM["fig8"] = info
    save(fig, "lpge_map_grid", prov.record("Fig. 8 lpge_map_grid", files), A.out, A.png)


# ------------------------------------------------------------------ Fig. 9: band edge eta(B), TNN, plus residual acceptance figure
def fig9(A, prov, NUM):
    fT = [os.path.join(A.data, f"edge7_{m}_TNN.json") for m in MATS]
    if not need(fT): return
    D = {m: load_json(f, prov) for m, f in zip(MATS, fT)}
    S = {m: edge_series(D[m]) for m in MATS}
    vmax = max(np.abs(S[m][1]).max() for m in MATS); p = pow10(vmax); sc = 10.0**p
    fig, ax = plt.subplots(1, 4, figsize=(TW, 1.85), gridspec_kw=dict(wspace=0.42))
    for i, (m, lab) in enumerate(MAT):
        B, Y, _ = S[m]; s, c = odd_fit(B, Y); Bf, Yf = mirror(B, Y); a = ax[i]
        a.plot(Bf, Yf/sc, color=RED if s > 0 else BLUE)
        zero(a, v=True); a.set_xlabel(r"$B$ (T)"); a.set_xlim(-62, 62); a.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 3, 5, 6, 10]))
        a.set_title(lab, pad=3, loc="right"); label(a, f"({LET[i]})"); a.yaxis.set_major_locator(MaxNLocator(5))
    ax[0].set_ylabel(r"$\eta^{x;xx}$ ($10^{%d}\,%s$)" % (p, UNIT) if p else r"$\eta^{x;xx}$ ($%s$)" % UNIT)
    save(fig, "lpge_edge_sign_grid", prov.record("Fig. 9 lpge_edge_sign_grid", fT, dict(y_scale_power=p)), A.out, A.png)

    # ---- acceptance artifact: residual of the cubic odd fit, % of |eta(60 T)|, TNN and NN
    fN = [os.path.join(A.data, f"edge7_{m}_NN.json") for m in MATS]
    have_nn = all(os.path.exists(f) for f in fN)
    if not have_nn: print("  NOTE: NN edge files missing; residual figure uses TNN only")
    fig, ax = plt.subplots(1, 4, figsize=(TW, 1.85), gridspec_kw=dict(wspace=0.42))
    res = {}; worst = 0.0; used = list(fT)
    for i, (m, lab) in enumerate(MAT):
        a = ax[i]
        for model, c_, mk in (("TNN", RED, "o"), ("NN", ORANGE, "^")):
            f = os.path.join(A.data, f"edge7_{m}_{model}.json")
            if not os.path.exists(f): continue
            if model == "TNN": d = D[m]
            else: d = load_json(f, prov); used.append(f)
            B, Y, Y5 = edge_series(d); s, c = odd_fit(B, Y)
            ref = abs(np.interp(60.0, B, Y)) if B.max() >= 60 else abs(Y[-1])
            if B.max() < 60: warnings.warn(f"{f}: fields only up to {B.max():g} T; residual normalised to |eta({B.max():g} T)|")
            rr = 100*(Y - (s*B + c*B**3))/ref; mx = float(np.abs(rr).max()); worst = max(worst, mx)
            a.plot(B, rr, "-", marker=mk, ms=2.2, mew=0, color=c_, lw=0.8, label=model)
            a.text(0.97, 0.04 + 0.09*(model == "NN"), r"%s: $\max|r|=%.2f\%%$" % (model, mx), transform=a.transAxes, ha="right", va="bottom", fontsize=6, color=c_)
            stored = d.get("chi_SI_per_T")
            res[f"{m}|{model}"] = dict(max_abs_residual_pct=mx, s_refit=s, c_refit=c, chi_SI_per_T_stored=stored,
                                       chi_refit_vs_stored_rel=(abs(s-stored)/abs(stored) if stored else None),
                                       dev_60T_pct_refit=float(100*c*60**2/s), dev_60T_pct_stored=d.get("dev_60T_pct"),
                                       spread_5omega_pct_of_eta60=float(100*np.abs(Y5 - Y[:, None]).max()/ref),
                                       n_fields=int(len(B)), Bmax=float(B.max()))
        a.axhspan(-0.5, 0.5, color="0.92", lw=0, zorder=0); zero(a); yl = max(0.6, 1.1*max(abs(np.array(a.get_ylim())))); a.set_ylim(-yl, yl)
        a.set_xlabel(r"$B$ (T)"); a.set_xlim(-2, 62); a.set_title(lab, pad=3, loc="right"); label(a, f"({LET[i]})")
        a.yaxis.set_major_locator(MaxNLocator(5))
    ax[0].set_ylabel(r"$(\eta-sB-cB^3)/|\eta(60\,\mathrm{T})|$ (%)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), frameon=False)
    ok = worst < 0.5
    print(f"  Fig. 9 residual check: max |eta - (sB + cB^3)| = {worst:.3f}% of |eta(60 T)|  ->  {'PASS' if ok else 'FAIL'} (threshold 0.5%)")
    for k, v in res.items():
        print(f"    {k:11s} max|res| {v['max_abs_residual_pct']:.3f}%  chi {v['s_refit']:+.4e} (stored {v['chi_SI_per_T_stored']})"
              f"  dev60 {v['dev_60T_pct_refit']:+.3f}%  5-omega spread {v['spread_5omega_pct_of_eta60']:.2f}%")
    NUM["fig9"] = dict(y_scale_power=p, residual_check=dict(threshold_pct=0.5, max_abs_residual_pct=worst, verdict="PASS" if ok else "FAIL",
                                                           per_series=res))
    save(fig, "fig9_residuals", prov.record("Fig. 9 residual acceptance check", used, dict(verdict="PASS" if ok else "FAIL",
                                                                                           max_abs_residual_pct=worst)), A.out, A.png)


# ------------------------------------------------------------------ Fig. 10: chi vs photon energy above the gap
def fig10(A, prov, NUM):
    files = {(m, md): os.path.join(A.data, f"chiwR9v7_{m}_{md}_F_0.001.json") for m in MATS for md in ("TNN", "NN")}
    if not need(list(files.values())): return
    XMAX = A.xmax10; info = {}
    fig, ax = plt.subplots(1, 4, figsize=(TW, 2.25), gridspec_kw=dict(wspace=0.36))
    for i, (m, lab) in enumerate(MAT):
        a = ax[i]; ys = []
        for md, c, nm in (("TNN", RED, "TNN"), ("NN", ORANGE, "NN")):
            d = load_json(files[(m, md)], prov); o = np.array(d["off"]); raw = np.array(d["chi"])*1e4
            y = np.convolve(raw, np.ones(5)/5, "same")             # 5 meV running average (caption)
            k = (o >= -0.02) & (o <= XMAX - 0.003); a.plot(o[k], y[k], color=c, lw=1.0, label=nm); ys.append(y[k])
            xs = crossings(o[k], y[k], xmin=0.004)
            for x0 in xs: a.plot(x0, 0, "o", ms=4, mfc="white", mec=c, mew=1.0, zorder=5)
            thr = float(np.interp(0.01, o, y)); first = xs[0] if xs else None
            raw_x = crossings(o[(o > 0.004) & (o <= XMAX)], raw[(o > 0.004) & (o <= XMAX)])
            info[f"{m}|{md}"] = dict(Eg=d.get("Eg"), chi_Eg_plus_0p01_smoothed_SI_per_T=thr*1e-4,
                                     chi_Eg_plus_0p10_smoothed_SI_per_T=float(np.interp(0.10, o, y))*1e-4,
                                     threshold_sign=int(np.sign(thr)), sign_changes_eV_above_Eg=xs,
                                     threshold_sign_persists_to_eV=first if first is not None else f">={o[k].max():.3f} (end of window)",
                                     sign_changes_raw_unsmoothed=raw_x)
        yb = np.concatenate(ys); lo, hi = min(yb.min(), 0), max(yb.max(), 0); pad = 0.08*(hi - lo); a.set_ylim(lo - pad, hi + pad)
        a.axvline(0, color="0.6", lw=0.6, ls="-")
        eb = 2*LAM_LIU[m]; info[f"{m}|B_onset_eV_above_Eg"] = eb
        if eb < XMAX:
            a.axvline(eb, color="0.5", lw=0.6, ls=":"); a.text(eb - 0.008, 0.93, "B", transform=a.get_xaxis_transform(), fontsize=7, color="0.35", ha="right")
        else:
            info[f"{m}|B_onset_note"] = f"B onset at Eg+{eb:.3f} eV is outside the plotted window (<= {XMAX} eV); no dotted line drawn"
            print(f"  NOTE {m}: B onset Eg+{eb:.3f} eV outside the window (xmax {XMAX} eV)")
        zero(a); a.set_xlim(-0.02, XMAX); a.set_xlabel(r"$\hbar\omega-E_g$ (eV)"); a.yaxis.set_major_locator(MaxNLocator(6))
        a.set_title(lab, pad=3, loc="right"); label(a, f"({LET[i]})")
    ax[0].set_ylabel(r"$\chi$ ($10^{-4}\,%s\,\mathrm{T^{-1}}$)" % UNIT)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), frameon=False)
    print("  Fig. 10 sign changes (eV above Eg, 5 meV smoothed) and persistence of the threshold sign:")
    for m in MATS:
        for md in ("TNN", "NN"):
            v = info[f"{m}|{md}"]
            print(f"    {m:6s} {md:3s} chi(Eg+0.01) {v['chi_Eg_plus_0p01_smoothed_SI_per_T']:+.3e}  crossings {[round(x, 3) for x in v['sign_changes_eV_above_Eg']]}"
                  f"  threshold sign persists to {v['threshold_sign_persists_to_eV'] if isinstance(v['threshold_sign_persists_to_eV'], str) else round(v['threshold_sign_persists_to_eV'], 3)}")
    NUM["fig10"] = dict(window_eV=[-0.02, XMAX], smoothing="5-point (5 meV) running mean", crossing_min_eV=0.004, per_series=info)
    save(fig, "chi_vs_offset", prov.record("Fig. 10 chi_vs_offset", list(files.values())), A.out, A.png)


# ------------------------------------------------------------------ Fig. 11: threshold chi across models
def fig11(A, prov, NUM):
    fg = {(m, md): os.path.join(A.data, f"edge7_{m}_{md}.json") for m in MATS for md in ("TNN", "NN")}
    if not need(list(fg.values()) + [A.kp]): return
    fl = {(m, md): os.path.join(A.data, f"edge7_{m}_{md}_LDA.json") for m in MATS for md in ("TNN", "NN")}
    have_lda = {k: os.path.exists(v) for k, v in fl.items()}
    if not all(have_lda.values()): print("  NOTE: missing LDA files: " + ", ".join(v for k, v in fl.items() if not have_lda[k]))

    def chi_of(path):
        d = load_json(path, prov); s = d.get("chi_SI_per_T")
        if s is None:
            B, Y, _ = edge_series(d); s = odd_fit(B, Y)[0]; warnings.warn(f"{path}: no chi_SI_per_T (unfinished?), refitted from {len(B)} fields")
        return float(s), d.get("dev_60T_pct")
    G = {k: chi_of(v) for k, v in fg.items()}
    L = {k: chi_of(v) for k, v in fl.items() if have_lda[k]}
    kp = load_json(A.kp, prov, key="kp2_sha256")
    KP = {}
    for m in MATS:
        off = np.array(kp[m]["offsets_eV"]); j = int(np.argmin(np.abs(off - 0.01))); assert abs(off[j] - 0.01) < 1e-9
        KP[m] = dict(chi=float(kp[m]["chi_phys_SI_per_T"][j]), kappa=float(kp[m]["params"]["kappa"]))

    # linear fit to the 8 GGA TB points (chi in 1e-4 SI, kappa in eV A^2)
    kx = np.array([KAPPA_TB[md][m] for md in ("TNN", "NN") for m in MATS]); cy = np.array([G[(m, md)][0]*1e4 for md in ("TNN", "NN") for m in MATS])
    sl, ic = np.polyfit(kx, cy, 1); r = float(np.corrcoef(kx, cy)[0, 1])
    atFP = {m: float(sl*KP[m]["kappa"] + ic)*1e-4 for m in MATS}
    opp = lambda chi, kap: bool(np.sign(chi) == -np.sign(kap))

    fig, ax = plt.subplots(1, 2, figsize=(TW, 2.35), gridspec_kw=dict(wspace=0.28))
    ST = [("TNN", "o", RED, "TB, TNN"), ("NN", "^", ORANGE, "TB, NN")]
    kw = lambda c, fill, lab: dict(color=c, mfc=c if fill else "none", mew=0.9, ms=4.5, ls="none", label=lab)
    for k, (md, mk, c, lab) in enumerate(ST):
        ax[0].plot([i + (k - 1)*0.16 for i in range(4)], [G[(m, md)][0]*1e4 for m in MATS], mk, **kw(c, True, lab))
        ax[1].plot([KAPPA_TB[md][m] for m in MATS], [G[(m, md)][0]*1e4 for m in MATS], mk, **kw(c, True, lab))
    kpl = r"$\mathbf{k}\cdot\mathbf{p}$, first-principles"
    ax[0].plot([i + 0.16 for i in range(4)], [KP[m]["chi"]*1e4 for m in MATS], "s", **kw(BLUE, False, kpl))
    ax[1].plot([KP[m]["kappa"] for m in MATS], [KP[m]["chi"]*1e4 for m in MATS], "s", **kw(BLUE, False, kpl))
    for md, mk, c, lab in ST:
        ms_ = [m for m in MATS if (m, md) in L]
        if ms_: ax[1].plot([KAPPA_LDA[md][m] for m in ms_], [L[(m, md)][0]*1e4 for m in ms_], mk, **kw(c, False, lab + " (LDA)"))
    allk = list(kx) + [KP[m]["kappa"] for m in MATS] + [KAPPA_LDA[md][m] for (m, md) in L]
    xs = np.linspace(min(allk) - 0.12, max(allk) + 0.08, 50); ax[1].plot(xs, sl*xs + ic, ":", color="0.4", lw=0.9, label="fit to TB (GGA)")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels([l for _, l in MAT]); ax[0].tick_params(axis="x", which="minor", bottom=False, top=False)
    ax[0].set_xlim(-0.5, 3.5)
    ya = [G[(m, md)][0]*1e4 for m in MATS for md in ("TNN", "NN")] + [KP[m]["chi"]*1e4 for m in MATS]
    lo, hi = min(min(ya), 0), max(max(ya), 0); ax[0].set_ylim(lo - 0.06*(hi - lo), hi + 0.45*(hi - lo))   # room for the legend
    ax[0].set_ylabel(r"$\chi(E_g+0.01\,\mathrm{eV})$ ($10^{-4}\,%s\,\mathrm{T^{-1}}$)" % UNIT); ax[1].set_xlabel(r"$\kappa$ (eV$\,\mathrm{\AA}^2$)")
    for i, a in enumerate(ax): zero(a, v=(i == 1)); label(a, f"({LET[i]})")
    ax[0].legend(loc="upper left"); ax[1].legend(loc="upper right")

    print(f"  Fig. 11 fit to 8 GGA TB points: chi = ({sl:+.3f} kappa {ic:+.3f}) x 1e-4 A m V^-2 s^-1 T^-1 (kappa in eV A^2), r = {r:+.4f}")
    for m in MATS:
        print(f"    {m:6s} TNN {G[(m,'TNN')][0]:+.3e} NN {G[(m,'NN')][0]:+.3e}  LDA TNN {L.get((m,'TNN'), (float('nan'),))[0]:+.3e} NN {L.get((m,'NN'), (float('nan'),))[0]:+.3e}"
              f"  kp(FP) {KP[m]['chi']:+.3e} at kappa {KP[m]['kappa']:+.2f}  fit@FP kappa {atFP[m]:+.3e}")
    nopp_g = sum(opp(G[(m, md)][0], KAPPA_TB[md][m]) for m in MATS for md in ("TNN", "NN"))
    nopp_l = sum(opp(L[k][0], KAPPA_LDA[k[1]][k[0]]) for k in L)
    print(f"    sign opposite to kappa: GGA TB {nopp_g}/8, LDA TB {nopp_l}/{len(L)}")
    q = lambda v: float(f"{v:.2g}")
    NUM["fig11"] = dict(
        units="chi in A m V^-2 s^-1 T^-1 (physical sign); kappa in eV A^2",
        threshold_chi={**{f"{m}|{md}|GGA": dict(chi_SI_per_T=G[(m, md)][0], dev_60T_pct=G[(m, md)][1], kappa=KAPPA_TB[md][m]) for m in MATS for md in ("TNN", "NN")},
                       **{f"{m}|{md}|LDA": dict(chi_SI_per_T=L[(m, md)][0], dev_60T_pct=L[(m, md)][1], kappa=KAPPA_LDA[md][m]) for (m, md) in L}},
        kp_first_principles={m: dict(chi_SI_per_T=KP[m]["chi"], kappa_down=KP[m]["kappa"]) for m in MATS},
        fit_GGA_TB=dict(slope_1em4_per_eVA2=float(sl), intercept_1em4=float(ic), r=r, n_points=8,
                        fit_at_FP_kappa_SI_per_T=atFP, fit_at_FP_kappa_range=[min(atFP.values()), max(atFP.values())]),
        kp_FP_chi_range=[min(KP[m]["chi"] for m in MATS), max(KP[m]["chi"] for m in MATS)],
        sign_opposite_to_kappa=dict(GGA=f"{nopp_g}/8", LDA=f"{nopp_l}/{len(L)}"),
        kappa_FP_over_kappa_TNN={m: KP[m]["kappa"]/KAPPA_TB["TNN"][m] for m in MATS},
        dev_60T_pct_TNN_GGA_max_abs=max(abs(G[(m, "TNN")][1]) for m in MATS if G[(m, "TNN")][1] is not None) if any(G[(m, "TNN")][1] is not None for m in MATS) else None,
        quoted_1em4={f"{m}|{md}": q(G[(m, md)][0]*1e4) for m in MATS for md in ("TNN", "NN")})
    used = list(fg.values()) + [v for k, v in fl.items() if have_lda[k]] + [A.kp]
    save(fig, "threshold_models", prov.record("Fig. 11 threshold_models", used, dict(fit_slope=float(sl), fit_intercept=float(ic), fit_r=r)), A.out, A.png)


# ------------------------------------------------------------------ Fig. 12: chi(Eg + 0.1 eV) vs lambda
def fig12(A, prov, NUM):
    files = [os.path.join(A.data, f"lam7_{m}_F.npz") for m in MATS]
    if not need(files): return
    fig, ax = plt.subplots(1, 4, figsize=(TW, 1.85), gridspec_kw=dict(wspace=0.42)); info = {}
    for i, (m, lab) in enumerate(MAT):
        d, meta = load_npz(files[i], prov); n = int(meta.get("done", len(d["lam"]))); l0 = float(meta["lam_liu"])
        if n < len(d["lam"]): warnings.warn(f"{files[i]}: partial run, {n}/{len(d['lam'])} lambda values")
        off = np.array(meta.get("offsets_eV", [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30])); j = int(np.argmin(np.abs(off - 0.10)))
        assert abs(off[j] - 0.10) < 1e-9, "offset 0.10 eV not found"
        lam = d["lam"][:n]; chi = PHYS*d["chi_SI_per_T"][:n, j]                  # code sign -> physical
        w = (lam >= 0.5*l0 - 1e-9) & (lam <= 1.5*l0 + 1e-9); o = np.argsort(lam[w]); x, y = lam[w][o], chi[w][o]; a = ax[i]
        a.plot(x, y*1e4, "o", color=COL[m], ms=2.6, mew=0)
        fits = {}
        for deg, ls in ((1, "-"), (2, "--")):
            cf = np.polyfit(x, y, deg); yf = np.polyval(cf, x); a.plot(x, yf*1e4, ls, color=COL[m], lw=0.8)
            fits[deg] = dict(coef_SI=cf.tolist(), R2=r2(y, yf))
        root = -fits[1]["coef_SI"][1]/fits[1]["coef_SI"][0]
        a.axvline(l0, color="0.6", ls=":", lw=0.7); zero(a); a.set_xlabel(r"$\lambda$ (eV)"); a.set_title(lab, pad=3, loc="right")
        label(a, f"({LET[i]})"); a.xaxis.set_major_locator(MaxNLocator(4))
        k0 = int(np.argmin(np.abs(lam - l0)))
        info[m] = dict(lam0=l0, n_points_in_window=int(w.sum()), R2_linear=fits[1]["R2"], R2_quadratic=fits[2]["R2"],
                       linear_slope_SI_per_T_per_eV=fits[1]["coef_SI"][0], quad_coef_SI=fits[2]["coef_SI"],
                       chi_at_lam0_SI_per_T=float(chi[k0]), lam_at_lam0_point=float(lam[k0]),
                       sign_changes_data_lam_eV=crossings(x, y), sign_change_linear_fit_lam_eV=float(root) if x.min() <= root <= x.max() else None,
                       sign_change_in_window=bool(crossings(x, y)) or bool(x.min() <= root <= x.max()),
                       monotonic=bool(np.all(np.diff(y) > 0) or np.all(np.diff(y) < 0)),
                       Eg_at_lam0=float(d["Eg"][k0]))
        print(f"  Fig. 12 {m:6s} lam0 {l0:.3f}  R2 lin {fits[1]['R2']:.3f} quad {fits[2]['R2']:.3f}  chi(lam0) {chi[k0]:+.3e}"
              f"  sign changes (data) {[round(v, 4) for v in crossings(x, y)]}  lin-fit root {root:.4f}{'' if x.min() <= root <= x.max() else ' (outside window)'}")
    ax[0].set_ylabel(r"$\chi$ ($10^{-4}\,%s\,\mathrm{T^{-1}}$)" % UNIT)
    NUM["fig12"] = dict(offset_eV=0.10, window="[0.5, 1.5] lambda0", sign_note="lam7 chi_SI_per_T is code sign; multiplied by -2", per_material=info)
    save(fig, "lpge_lambda_grid", prov.record("Fig. 12 lpge_lambda_grid", files), A.out, A.png)


# ------------------------------------------------------------------ cross-checks between figures (cheap, printed and stored)
def cross_checks(NUM):
    cc = {}
    f10, f11, f12 = NUM.get("fig10"), NUM.get("fig11"), NUM.get("fig12")
    if f10 and f11:      # chi(Eg+0.01) from chi(omega) (h=0.001/a, B=30 T) vs edge fit (h=0.0007/a, cubic fit to 60 T)
        for m in MATS:
            for md in ("TNN", "NN"):
                a_ = f10["per_series"][f"{m}|{md}"]["chi_Eg_plus_0p01_smoothed_SI_per_T"]; b_ = f11["threshold_chi"][f"{m}|{md}|GGA"]["chi_SI_per_T"]
                cc[f"chi(Eg+0.01) {m}|{md}: fig10 vs fig11"] = dict(fig10=a_, fig11=b_, same_sign=bool(np.sign(a_) == np.sign(b_)))
    if f10 and f12:      # chi(Eg+0.1) TNN: fig10 grid vs lambda sweep at lambda0 (h=0.002/a, B=+-10 T)
        for m in MATS:
            a_ = f10["per_series"][f"{m}|TNN"]["chi_Eg_plus_0p10_smoothed_SI_per_T"]; b_ = f12["per_material"][m]["chi_at_lam0_SI_per_T"]
            cc[f"chi(Eg+0.10) {m}|TNN: fig10 vs fig12(lam0)"] = dict(fig10=a_, fig12=b_, same_sign=bool(np.sign(a_) == np.sign(b_)))
    if cc:
        print("  cross-checks:")
        for k, v in cc.items():
            vals = [x for x in v.values() if isinstance(x, float)]
            print(f"    {k:42s} {vals[0]:+.3e} vs {vals[1]:+.3e}  {'same sign' if v['same_sign'] else 'SIGN MISMATCH'}")
    NUM["cross_checks"] = cc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=".", help="folder with prod7/edge7/chiwR9v7/lam7 files (default: .)")
    ap.add_argument("--kp", default=os.path.join("..", "kp", "kp_fp_all_v7.json"), help="first-principles k.p json (default ../kp/kp_fp_all_v7.json)")
    ap.add_argument("--out", default="figs_v7", help="output folder (default figs_v7)")
    ap.add_argument("--only", nargs="*", type=int, default=[7, 8, 9, 10, 11, 12])
    ap.add_argument("--xmax10", type=float, default=0.45, help="Fig. 10 right edge, eV above Eg")
    ap.add_argument("--no-png", dest="png", action="store_false")
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
    import logging; logging.getLogger("fontTools").setLevel(logging.ERROR)   # cmr10 subsetting chatter
    warnings.simplefilter("always"); warnings.simplefilter("ignore", ResourceWarning); warnings.formatwarning = lambda msg, cat, *a, **k: f"  WARNING: {msg}\n"
    prov = Prov(A); NUM = {}
    for k, fn in ((7, fig7), (8, fig8), (9, fig9), (10, fig10), (11, fig11), (12, fig12)):
        if k in A.only: print(f"Fig. {k}"); fn(A, prov, NUM)
    cross_checks(NUM)
    eng = sorted({i["engine_sha256"] for i in prov.inputs.values() if i["engine_key"] == "lpge_sha256" and i["engine_sha256"]})
    NUM["provenance"] = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), script=os.path.basename(prov.script),
                             script_sha256=prov.script_sha, engine_sha256=eng, engine_consistent=len(eng) <= 1,
                             engine_matches_expected=bool(eng) and all(e.startswith(EXPECT_ENGINE) for e in eng),
                             inputs={p: i["sha256"] for p, i in sorted(prov.inputs.items())})
    path = os.path.join(A.out, "numbers_v7.json"); wjson(NUM, path, indent=1, default=float)
    print(f"wrote {path}; engine hash(es) {eng}{'' if NUM['provenance']['engine_matches_expected'] else '  <-- NOT the expected ' + EXPECT_ENGINE}")


if __name__ == "__main__":
    main()
