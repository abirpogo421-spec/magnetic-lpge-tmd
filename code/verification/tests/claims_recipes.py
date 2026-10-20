"""Recipes that (re)compute claim values for claims.json (G4 fast recompute, G5 full recompute, and the builder).

Every recipe is a plain function recipe(ctx, **args) -> float. Recipes use the independent reference
(ref/refimpl.py) wherever possible, the literature tables in literature/, or a declared data file.
"""
import os, json
import numpy as np
from vcore import SUITE_DIR

MAT4 = ["MoS2", "MoSe2", "WS2", "WSe2"]
E = 1.602176634e-19; HBAR = 1.054571817e-34; ME = 9.1093837015e-31; MUB = 5.7883818060e-5
C0 = 299792458.0; EPS0 = 8.8541878128e-12


def _agg(vals, agg):
    vals = np.asarray(vals, float)
    return float({"min": np.min, "max": np.max, "absmin": lambda v: np.min(np.abs(v)), "absmax": lambda v: np.max(np.abs(v))}[agg](vals))


def _lit(name):
    return json.load(open(os.path.join(SUITE_DIR, "literature", name)))


# ------------------------------------------------------------------------------------------------ Lowdin (Table II)
def lowdin(ctx, material, model="TNN", functional="GGA", q="kappa", lam=None, lam_scale=None, with_soc=True):
    ref = ctx.ref
    if lam_scale is not None:
        lam = lam_scale * ref.params(material)["lam"]
    return float(ref.kappa_lowdin(material, model, functional, lam=lam, with_soc=with_soc)[q])


def lowdin_range(ctx, model="TNN", functional="GGA", q="kappa", agg="min"):
    return _agg([lowdin(ctx, m, model, functional, q) for m in MAT4], agg)


def kfp_over_tnn_min(ctx):
    K = _lit("kormanyos_2015.json")
    kap = K["table6"]["rows"][K["interpretation"]["kappa_column"]]["DFT"]
    return float(min(abs(kap[i]) / abs(lowdin(ctx, m)) for i, m in enumerate(K["materials"])))


# ------------------------------------------------------------------------------------------------ literature
def lit(ctx, file, path):
    d = _lit(file)
    for p in path:
        d = d[p]
    return float(d)


def lit_range(ctx, file, path, agg):
    d = _lit(file)
    for p in path:
        d = d[p]
    return _agg(d, agg)


# ------------------------------------------------------------------------------------------------ constants
def const(ctx, name):
    return {"mu_B": MUB, "g_s": 2.0}[name]


def B_for_zeeman(ctx, muBB_eV):
    return muBB_eV / MUB


def ka_range(ctx, agg):
    vals = [ME * (ctx.ref.params(m)["a"] * 1e-10) ** 2 / HBAR ** 2 * E for m in MAT4]
    return _agg(vals, agg)


def half(ctx, x):
    return x / 2


# ------------------------------------------------------------------------------------------------ band geometry near K
def moment_K_range(ctx, agg, model="TNN"):
    ref = ctx.ref
    vals = []
    for m in MAT4:
        p = ref.params(m, model)
        for tau, s in ((1, 1), (-1, -1)):
            b = ref.band_quantities([tau * ref.K_POINT[0]], [0.0], s, 0.0, p=p, need_dM=False)
            vals += list(np.abs(b["M"][0, :2]))
    return _agg(vals, agg)


def D_max(ctx, B=60.0, R=0.3, h=0.01):
    ref = ctx.ref
    k, dA = ref.disk_grid(R, h)
    worst = 0.0
    for m in MAT4:
        p = ref.params(m, "TNN")
        for s in (1, -1):
            b = ref.band_quantities(k[:, 0], k[:, 1], s, B, p=p, need_dM=False)
            worst = max(worst, float(np.max(np.abs(b["D"][:, :2] - 1))))
    return worst


def valley_g(ctx, material, model="TNN", B=20.0):
    ref = ctx.ref
    p = ref.params(material, model)
    K = ref.K_POINT

    def G(b):
        a = ref.band_quantities([K[0]], [K[1]], 1, b, p=p, need_dM=False)
        c = ref.band_quantities([-K[0]], [-K[1]], -1, b, p=p, need_dM=False)
        return (a["E_shift"][0, 1] - a["E_shift"][0, 0]) - (c["E_shift"][0, 1] - c["E_shift"][0, 0])
    return float((G(B) - G(-B)) / (2 * MUB * B))


def valley_g_range(ctx, agg, model="TNN"):
    return _agg([valley_g(ctx, m, model) for m in MAT4], agg)


# ------------------------------------------------------------------------------------------------ ring averages (Table IV)
def ring(ctx, k, channel, v=1.0, m=0.83, kappa=0.25):
    ref = ctx.ref
    if channel == "inj_exact":
        return float(ref.kp_ring(k, v, kappa, m)["inj"]["xxx"])
    if channel == "inj_closed":
        return float(ref.kp_inj_closed_lead(k, v, kappa, m))
    if channel == "shift_exact":
        return float(ref.kp_ring(k, v, kappa, m)["shift"]["yyy"])
    if channel == "shift_closed":
        return float(ref.kp_shift_closed(0.0, v, kappa, m))
    raise KeyError(channel)


def ring_shift_sigfigs(ctx, v=1.0, m=0.83, kappa=0.25, k=1e-4):
    """Number of significant figures to which the exact k->0 ring average equals kappa v/2m^2."""
    a, b = ring(ctx, k, "shift_exact", v, m, kappa), ring(ctx, 0.0, "shift_closed", v, m, kappa)
    return float(np.floor(-np.log10(abs(a - b) / abs(b) * 2)))


# ------------------------------------------------------------------------------------------------ k.p numbers (Table V, Fig. 6, App. D)
def _kp_base(ctx):
    return json.load(open(os.path.join(ctx.path("kp_out"), "kp_params_TNN.json")))["MoS2"]


def kp_table(ctx, entry, Nk=401):
    """Table V entries recomputed with kp2 (k in 1/a, parameters kp_params_TNN.json MoS2), physical SI."""
    kp2 = ctx.eng.kp2
    base = _kp_base(ctx)
    a = base["a_A"] * 1e-10
    si = E ** 3 * a / HBAR ** 2; ssi = E ** 2 * a / HBAR * 1e15
    Eg = 2 * base["mD"] - base["lam"]; o = np.array([Eg + 0.10])
    tag, comp = entry.split(":")
    b, pairs = {"KKp_B0": (0.0, ((1, 1), (-1, -1))), "KKp_B30": (MUB * 30, ((1, 1), (-1, -1))), "K_B0": (0.0, ((1, 1),))}[tag]
    Lt, St = kp2.response("x", b, o, p=base, R=0.6, Nk=Nk, nu=0.01, want_shift=comp.startswith("shift"), pairs=pairs, full=True)
    return float({"lpge_xxx": -2 * si * Lt[0, 0, 0, 0], "lpge_yyy": -2 * si * Lt[0, 1, 1, 1],
                  "shift_yyy": ssi * St[0, 1, 1, 1], "shift_xxx": ssi * St[0, 0, 0, 0]}[comp])


def kp_table_data(ctx, entry):
    d = json.load(open(os.path.join(ctx.path("kp_out"), "kp_table.json")))
    base = _kp_base(ctx)
    a = base["a_A"] * 1e-10
    tag, comp = entry.split(":")
    v = d[tag][comp]
    return float(-2 * E ** 3 * a / HBAR ** 2 * v if comp.startswith("lpge") else E ** 2 * a / HBAR * 1e15 * v)


def kp_spectra(ctx, what, onset_window=0.01):
    """Fig. 6 statements from kp_spectra.json; 'shift_change_30T_pointwise' = max over the plotted range of
    |sigma(30 T) - sigma(0)|/|sigma(0)| (points with |sigma(0)| > 1e-3 of its maximum), '..._argmax_offset' = where (eV from E_g)."""
    S = json.load(open(os.path.join(ctx.path("kp_out"), "kp_spectra.json")))
    om = np.array(S["omega"]); W = om <= S["Eg"] + 0.45
    if what in ("shift_change_30T_pointwise", "shift_change_30T_argmax_offset"):
        S0, S30 = np.array(S["KKp_B0"]["shift"]), np.array(S["KKp_B30"]["shift"])
        k = W & (np.abs(S0) > 1e-3 * np.abs(S0[W]).max())
        r = np.abs(S30[k] - S0[k]) / np.abs(S0[k])
        return float(r.max()) if what.endswith("pointwise") else float(abs(om[k][np.argmax(r)] - S["Eg"]))
    L30, LK = np.array(S["KKp_B30"]["lpge"]), np.array(S["K_B0"]["lpge"])
    S0, S30 = np.array(S["KKp_B0"]["shift"]), np.array(S["KKp_B30"]["shift"])
    if what == "B30_over_single":
        return float(np.abs(L30[W]).max() / np.abs(LK[W]).max())
    if what == "shift_change_30T":
        return float((np.abs(S30 - S0)[W]).max() / np.abs(S0[W]).max())
    raise KeyError(what)


def kp_sweep_lam_change(ctx):
    sw = json.load(open(os.path.join(ctx.path("kp_out"), "kp_sweeps.json")))
    c = np.array(sw["lam"]["chi"])
    return float(np.max(np.abs(c / c[0] - 1)))


def kp_fp_chi(ctx, agg, offset_index=0):
    d = json.load(open(os.path.join(ctx.path("kp_out"), "kp_fp_all_v7.json")))
    return _agg([d[m]["chi_phys_SI_per_T"][offset_index] for m in MAT4], agg)


def kp_fp_sign(ctx, material):
    d = json.load(open(os.path.join(ctx.path("kp_out"), "kp_fp_all_v7.json")))
    return float(np.sign(d[material]["chi_phys_SI_per_T"][0]))


def kp_fp_order(ctx):
    """Common decimal order of the four first-principles k.p threshold chi (None if they differ)."""
    d = json.load(open(os.path.join(ctx.path("kp_out"), "kp_fp_all_v7.json")))
    o = {int(np.round(np.log10(abs(d[m]["chi_phys_SI_per_T"][0])))) for m in MAT4}
    return float(o.pop()) if len(o) == 1 else float("nan")


# ------------------------------------------------------------------------------------------------ v6 data (provisional)
def _prod_v6(ctx, m):
    d = np.load(os.path.join(ctx.path("data_v6"), f"prod_{m}_F.npz"), allow_pickle=True)
    meta = json.loads(str(d["meta"])); n = meta["done"]
    return d["B_tesla"][:n], d["omega"], d["eta"][:n, :, 0, 0, 0] * meta["si_factor"] * -2.0


def vh_energy(ctx, material):
    B, om, Y = _prod_v6(ctx, material)
    return float(om[np.argmax(np.abs(Y[-1]))])


def vh_slope(ctx, material, Bmax=20.0):
    """Small-field slope at the van Hove peak as in fig_main.py: linear fit of the odd-extended eta(B) over |B| <= 20 T."""
    B, om, Y = _prod_v6(ctx, material)
    y = Y[:, np.argmax(np.abs(Y[-1]))]
    Bf = np.concatenate([-B[:0:-1], B]); yf = np.concatenate([-y[:0:-1], y]); k = np.abs(Bf) <= Bmax
    return float(np.polyfit(Bf[k], yf[k], 1)[0])


def vh_slope_range(ctx, agg):
    return _agg([vh_slope(ctx, m) for m in MAT4], agg)


def lin_v6_dev(ctx, B="20", agg="max"):
    d = json.load(open(os.path.join(ctx.path("results_v6"), "lin_tnn.json")))
    return _agg([abs(d[f"{m}|TNNvH|F"]["dev"][B]) for m in MAT4], agg)


def vh_eta60(ctx, agg):
    vals = []
    for m in MAT4:
        B, om, Y = _prod_v6(ctx, m)
        vals.append(abs(Y[-1][np.argmax(np.abs(Y[-1]))]))
    return _agg(vals, agg)


def decomp_v6(ctx, key):
    return float(json.load(open(os.path.join(ctx.path("results_v6"), "decomp.json")))[key])


def decomp_v6_ratio_range(ctx, where, num, den, agg, minus_one=False):
    d = json.load(open(os.path.join(ctx.path("results_v6"), "decomp.json")))
    vals = [d[f"{m}|{where}|{num}"] / d[f"{m}|{where}|{den}"] - (1 if minus_one else 0) for m in MAT4]
    return _agg(vals, agg)


def decomp_v6_chi_range(ctx, agg):
    d = json.load(open(os.path.join(ctx.path("results_v6"), "decomp.json")))
    return _agg([d[f"{m}|vH|F"] for m in MAT4], agg)


def lin_v6_dev60(ctx, agg):
    d = json.load(open(os.path.join(ctx.path("results_v6"), "lin_tnn.json")))
    return _agg([abs(d[f"{m}|TNNvH|F"]["dev"]["60"]) for m in MAT4], agg)


def lin_v6_linear_to(ctx, tol_pct=1.0):
    """Largest field (from 10, 20, 30, 40, 60 T) at which every van Hove deviation from linear is below tol_pct."""
    d = json.load(open(os.path.join(ctx.path("results_v6"), "lin_tnn.json")))
    ok = [b for b in ("10", "20", "30", "40", "60") if all(abs(d[f"{m}|TNNvH|F"]["dev"][b]) < tol_pct * 2 for m in MAT4)]
    return float(max(map(float, ok))) if ok else 0.0


# ------------------------------------------------------------------------------------------------ v7 data (final text)
# All v7 production files store the PHYSICAL sign (maps_v7/edge_v7/chiw9_v7 multiply by -2 si_factor) except
# lam7_*.npz (code sign; x -2 here). Definitions follow the final text (2026-10-14):
#   van Hove energy = omega of max |eta| at 60 T in prod7; van Hove chi = eta(1 T)/(1 T) (initial slope s0);
#   "within 5% of linear up to 20 T" and "x% below the linear extrapolation at 60 T" are relative to s0;
#   threshold chi = edge7 chi_SI_per_T (fit s B + c B^3 of the 5-omega mean over |B| <= 60 T).
class PendingData(Exception):
    """The v7 output a recipe needs is missing or unfinished (cloud copy partial / queue running)."""


def _v7(ctx, name, key="v7_out"):
    p = os.path.join(ctx.path(key), name)
    ctx.memo.setdefault("_access", set()).add(f"{key}:{name}")     # the builder records the sha of every file read
    if not os.path.exists(p):
        raise PendingData(f"{name} missing in {ctx.path(key)}")
    return p


def _prod7(ctx, m):
    d = np.load(_v7(ctx, f"prod7_{m}_F.npz"), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    if meta["done"] != len(d["B_tesla"]):
        raise PendingData(f"prod7_{m}_F.npz unfinished ({meta['done']}/{len(d['B_tesla'])} fields)")
    return np.asarray(d["B_tesla"], float), np.asarray(d["omega"], float), np.asarray(d["eta"], float)


def _vh7(ctx, m):
    B, om, Y = _prod7(ctx, m)
    i = int(np.argmax(np.abs(Y[-1])))
    assert abs(B[-1] - 60.0) < 1e-9 and abs(B[1] - 1.0) < 1e-9, "prod7 field grid is not 0, 1, ..., 60 T"
    return B, om[i], Y[:, i]


def vh7_energy(ctx, material):
    return float(_vh7(ctx, material)[1])


def vh7_slope(ctx, material):
    """Initial slope s0 = eta(1 T)/(1 T) at the van Hove energy."""
    B, _, y = _vh7(ctx, material)
    return float(y[1] / B[1])


def vh7_slope_range(ctx, agg):
    return _agg([vh7_slope(ctx, m) for m in MAT4], agg)


def vh7_eta60(ctx, agg):
    return _agg([abs(_vh7(ctx, m)[2][-1]) for m in MAT4], agg)


def vh7_dev20(ctx, agg="max", Bmax=20.0):
    """max over 0 < B <= Bmax of |eta(B)/(s0 B) - 1| (fraction), per material, aggregated."""
    vals = []
    for m in MAT4:
        B, _, y = _vh7(ctx, m); s0 = y[1] / B[1]; k = (B > 0) & (B <= Bmax + 1e-9)
        vals.append(float(np.max(np.abs(y[k] / (s0 * B[k]) - 1))))
    return _agg(vals, agg)


def vh7_below60(ctx, agg):
    """1 - eta(60 T)/(s0 60 T) (fraction below the linear extrapolation of the initial slope)."""
    vals = []
    for m in MAT4:
        B, _, y = _vh7(ctx, m); s0 = y[1] / B[1]
        vals.append(float(1 - y[-1] / (s0 * B[-1])))
    return _agg(vals, agg)


def maps7_weight_fraction(ctx, lo, hi, agg="min"):
    """Fraction of sum_B sum_omega |eta| of each prod7 map inside [lo, hi] eV (assertion 'most of the weight')."""
    vals = []
    for m in MAT4:
        B, om, Y = _prod7(ctx, m); w = np.abs(Y).sum(0); k = (om >= lo) & (om <= hi)
        vals.append(float(w[k].sum() / w.sum()))
    return _agg(vals, agg)


def maps7_banding(ctx, material="WSe2", level=0.005, what="Emin"):
    """Field-banding indicator |d2 eta/dB2|/max|eta| > level: lowest/highest energy and lowest field where it occurs."""
    B, om, Y = _prod7(ctx, material)
    d2 = np.abs(Y[2:] - 2 * Y[1:-1] + Y[:-2]) / np.abs(Y).max()
    ii, jj = np.where(d2 > level)
    if not len(ii):
        return float("nan")
    return float({"Emin": om[jj].min(), "Emax": om[jj].max(), "Bmin": B[1:-1][ii].min()}[what])


def maps7_banding_others(ctx, level=0.005):
    """Largest banding indicator in the three maps other than WSe2 (must stay below 'level' for the WSe2-only remark)."""
    out = []
    for m in ("MoS2", "MoSe2", "WS2"):
        B, om, Y = _prod7(ctx, m)
        out.append(float((np.abs(Y[2:] - 2 * Y[1:-1] + Y[:-2]) / np.abs(Y).max()).max()))
    return max(out)


def decomp7_ratio_range(ctx, where, num, den, agg, minus_one=False):
    vals = []
    for m in MAT4:
        d = json.load(open(_v7(ctx, f"decomp7_{m}.json")))
        vals.append(d[f"{m}|{where}|{num}"] / d[f"{m}|{where}|{den}"] - (1 if minus_one else 0))
    return _agg(vals, agg)


def _edge7(ctx, m, model, functional="GGA"):
    fn = f"edge7_{m}_{model}.json" if functional == "GGA" else f"edge7_{m}_{model}_{functional}.json"
    d = json.load(open(_v7(ctx, fn)))
    if "chi_SI_per_T" not in d or "finished" not in d:
        raise PendingData(f"{fn} unfinished ({len(d.get('eta', {}))} fields)")
    return d


def edge7_chi(ctx, material, model="TNN", functional="GGA"):
    return float(_edge7(ctx, material, model, functional)["chi_SI_per_T"])


def edge7_sign(ctx, material, model="TNN", functional="GGA"):
    return float(np.sign(edge7_chi(ctx, material, model, functional)))


def edge7_dev60(ctx, model="TNN", functional="GGA", agg="absmax"):
    """|c 60^2/s| of the edge fit (fraction)."""
    return _agg([_edge7(ctx, m, model, functional)["dev_60T_pct"] / 100 for m in MAT4], agg)


def edge7_lin30(ctx, model="TNN", B=30.0):
    """max over materials of |eta(B)/B - s|/|s|, eta = 5-omega mean of the stored edge run, s = chi_SI_per_T (B = 0 slope)."""
    vals = []
    for m in MAT4:
        d = _edge7(ctx, m, model); s = d["chi_SI_per_T"]
        vals.append(abs(np.mean(d["eta"][f"{B:g}"]) / B - s) / abs(s))
    return float(max(vals))


def edge7_lin30_local(ctx, model="TNN", B=30.0):
    """Same with the local derivative d eta/dB at B (from the fit s + 3 c B^2) instead of the B = 0 slope."""
    vals = []
    for m in MAT4:
        d = _edge7(ctx, m, model); s = d["chi_SI_per_T"]; c = d["dev_60T_pct"] / 100 * s / 60 ** 2
        vals.append(abs(np.mean(d["eta"][f"{B:g}"]) / B - (s + 3 * c * B * B)) / abs(s))
    return float(max(vals))


def edge7_omega_window(ctx, model="TNN"):
    """Photon-energy span of the 5-point average of the edge runs (eV), max over materials."""
    return float(max(np.ptp(_edge7(ctx, m, model)["omega"]) for m in MAT4))


def _fit11(ctx):
    kap, y = [], []
    for model in ("TNN", "NN"):
        for m in MAT4:
            kap.append(lowdin(ctx, m, model)); y.append(edge7_chi(ctx, m, model) / 1e-4)
    return np.array(kap), np.array(y)


def fit11(ctx, what):
    """Fig. 11 linear fit of the 8 GGA TB threshold chi (1e-4 SI) against the Lowdin kappa (eV A^2)."""
    kap, y = _fit11(ctx)
    A = np.polyfit(kap, y, 1)
    if what == "slope":
        return float(A[0])
    if what == "icpt":
        return float(A[1])
    if what == "r":
        return float(np.corrcoef(kap, y)[0, 1])
    if what == "count_opposite":
        return float(np.sum(np.sign(y) == -np.sign(kap)))
    K = _lit("kormanyos_2015.json")
    kfp = np.array(K["table6"]["rows"][K["interpretation"]["kappa_column"]]["DFT"])
    atfp = (A[0] * kfp + A[1]) * 1e-4
    return float({"atFP_min": atfp.min(), "atFP_max": atfp.max()}[what])


def lda_count_opposite(ctx, models=("TNN", "NN")):
    return float(sum(np.sign(edge7_chi(ctx, m, mo, "LDA")) == -np.sign(lowdin(ctx, m, mo, "LDA")) for mo in models for m in MAT4))


def all16_opposite(ctx):
    n = sum(np.sign(edge7_chi(ctx, m, mo, f)) == -np.sign(lowdin(ctx, m, mo, f)) for f in ("GGA", "LDA") for mo in ("TNN", "NN") for m in MAT4)
    return float(n)


def _chiw7(ctx, m, model, xmax=0.45, smooth=5, offmin=0.004):
    d = json.load(open(_v7(ctx, f"chiwR9v7_{m}_{model}_F_0.001.json")))
    off = np.array(d["off"]); c = np.convolve(np.array(d["chi"]), np.ones(smooth) / smooth, "same")
    k = (off >= offmin) & (off <= xmax + 1e-9)
    s = np.sign(c[k]); idx = np.where(np.diff(s) != 0)[0]
    # linear interpolation of the first zero crossing
    if not len(idx):
        return None, float(off[k][-1])
    i = idx[0]; x0, x1, y0, y1 = off[k][i], off[k][i + 1], c[k][i], c[k][i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0)), float(off[k][-1])


def chiw7_signchange(ctx, material, model):
    x, end = _chiw7(ctx, material, model)
    if x is None:
        raise ValueError(f"no sign change for {material} {model}")
    return x


def chiw7_early_count(ctx, below=0.1):
    return float(sum((lambda x: x is not None and x < below)(_chiw7(ctx, m, mo)[0]) for m in MAT4 for mo in ("TNN", "NN")))


def chiw7_persist(ctx, below=0.1):
    """min over the series without an early sign change of the energy to which the threshold sign persists
    (end of the plotted window if it never changes)."""
    vals = []
    for m in MAT4:
        for mo in ("TNN", "NN"):
            x, end = _chiw7(ctx, m, mo)
            if x is None:
                vals.append(end)
            elif x >= below:
                vals.append(x)
    return float(min(vals))


def lam7_R2(ctx, materials=("WS2", "WSe2"), agg="min", offset_index=4):
    """R^2 of a linear fit of chi(Eg+0.1 eV) vs lambda over [0.5, 1.5] lambda0 (lam7 code sign x -2)."""
    vals = []
    for m in materials:
        d = np.load(_v7(ctx, f"lam7_{m}_F.npz"), allow_pickle=True); meta = json.loads(str(d["meta"])); n = meta["done"]
        if n != len(d["lam"]):
            raise PendingData(f"lam7_{m}_F.npz unfinished")
        l0 = meta["lam_liu"]; lam = d["lam"][:n]; c = -2.0 * d["chi_SI_per_T"][:n, offset_index]
        w = (lam >= 0.5 * l0 - 1e-9) & (lam <= 1.5 * l0 + 1e-9)
        A = np.polyfit(lam[w], c[w], 1); res = c[w] - np.polyval(A, lam[w])
        vals.append(float(1 - res.var() / c[w].var()))
    return _agg(vals, agg)


def lam7_slope_sign(ctx, materials=("WS2", "WSe2"), offset_index=4):
    """min over materials of sign(linear slope) (+1: chi increases with lambda for all)."""
    out = []
    for m in materials:
        d = np.load(_v7(ctx, f"lam7_{m}_F.npz"), allow_pickle=True); meta = json.loads(str(d["meta"])); l0 = meta["lam_liu"]
        lam = d["lam"]; c = -2.0 * d["chi_SI_per_T"][:, offset_index]; w = (lam >= 0.5 * l0 - 1e-9) & (lam <= 1.5 * l0 + 1e-9)
        out.append(np.sign(np.polyfit(lam[w], c[w], 1)[0]))
    return float(min(out))


def conv_c2(ctx):
    """max over materials of |chi(N=900)/chi(N=600) - 1| at the van Hove peak (B = 1 T): checks/c2_N_convergence.json."""
    d = json.load(open(_v7(ctx, "c2_N_convergence.json", "v7_checks")))
    return float(max(abs(v["rel"]) for k, v in d.items() if k != "meta"))


def conv_c3(ctx):
    """max over the 8 GGA sets of |chi(h=0.0007)/chi(h=0.001) - 1| at E_g+0.01 eV, 30 T: checks/c3_h_convergence.json."""
    d = json.load(open(_v7(ctx, "c3_h_convergence.json", "v7_checks")))
    return float(max(abs(v["rel"]) for k, v in d.items() if k != "meta"))


def psf_thr(ctx, agg, num="ED", den="E", h=0.002, B=10.0, model="TNN"):
    """|chi_num/chi_den - 1| at threshold (E_g+0.01 eV, 5-omega mean over +-2.5 meV as edge_v7.py), engine lpge.eta on
    disks R=0.9/a with spacing h, nu = 5 meV, chi = eta(B)/B (eta exactly odd on T-paired disks). Default: the phase-space
    factor added to the moment energy shift (ED vs E). Expensive (~3 min): full mode only."""
    key = ("psf", num, den, h, B, model)
    if key not in ctx.memo:
        EN = ctx.eng; L, tb = EN.L, EN.tb; vals = []
        for m in MAT4:
            tb.set_config(m, "GGA")
            q = np.linspace(-0.05, 0.05, 401)
            Eg = float(min((lambda e: (e[:, 1] - e[:, 0]).min())(np.linalg.eigvalsh(blk)) for blk in L._H6(4 * np.pi / 3 + q, 0 * q, model, 0.0)))
            om = Eg + 0.01 + np.linspace(-0.0025, 0.0025, 5); kl = L.local_kpts(0.9, h)
            c = {v: float(np.mean(L.eta(m, model, None, B, om, 0.005, EN.FLAGS[v], kpts=kl)["sym"][:, 0, 0, 0])) for v in (num, den)}
            vals.append(abs(c[num] / c[den] - 1))
        ctx.memo[key] = vals
    return _agg(ctx.memo[key], agg)


def c4_change(ctx, materials=("MoS2", "WS2"), Na=600, Nb=900):
    """max |eta(Nb)/eta(Na) - 1| over the c4 hotspot points of the given materials (full field model F)."""
    d = json.load(open(_v7(ctx, "c4_hotspots.json", "v7_checks")))
    pts = [p for p in d["points"] if p["m"] in materials and p["flags"].get("vel") and p["flags"].get("mom") and p["flags"].get("dos")]
    out = []
    for p in pts:
        if p["N"] != Na:
            continue
        q = [x for x in pts if x["m"] == p["m"] and x["omega"] == p["omega"] and x["B"] == p["B"] and x["N"] == Nb and x["flags"] == p["flags"]]
        if q:
            out.append(abs(q[0]["eta"] / p["eta"] - 1))
    if not out:
        raise PendingData("no N pairs in c4_hotspots.json")
    return float(max(out))


# ------------------------------------------------------------------------------------------------ k.p lambda sweep (App. D)
def kp_lam_change(ctx, offset, Nk=401, lams=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25), R=0.6, nu=0.005, BT=30.0):
    """max over materials and lambda in [0, 0.25] eV of |chi(lambda)/chi(0) - 1|, two-band model with the TNN-derived
    parameters (kp_params_TNN.json: v, kappa, beta, mD in units of a), chi = [eta(+B) - eta(-B)]/2B, omega = 2 mD - lambda + offset."""
    kp2 = ctx.eng.kp2
    P = json.load(open(os.path.join(ctx.path("kp_out"), "kp_params_TNN.json")))
    worst = 0.0
    for m in MAT4:
        c = []
        for lam in lams:
            p = {k: P[m][k] for k in ("v", "kappa", "beta", "mD")}; p["lam"] = lam
            om = np.array([2 * p["mD"] - lam + offset]); b = MUB * BT
            ep = kp2.response("x", b, om, p=p, R=R, Nk=Nk, nu=nu, want_shift=False)[0]
            em = kp2.response("x", -b, om, p=p, R=R, Nk=Nk, nu=nu, want_shift=False)[0]
            c.append(float(np.ravel((ep - em) / (2 * BT))[0]))
        c = np.array(c); worst = max(worst, float(np.max(np.abs(c / c[0] - 1))))
    return worst


# ------------------------------------------------------------------------------------------------ TB band warping (Sec. III C)
def _tb_warp(ctx, m, model="TNN"):
    """Exact q^3 cos3phi coefficient c3 of d^2 = ((E_c - E_v)/2)^2 of the active spin at K, and the Kormanyos-style fit
    over 16% of |Gamma K| (Gamma-K and K-M lines), both expressed as kappa = c3/(2|v|) in eV A^2 (cf. audit X2 tb_warping.py)."""
    key = ("tbwarp", m, model)
    if key in ctx.memo:
        return ctx.memo[key]
    ref = ctx.ref; p = ref.params(m, model); a = p["a"]; K = ref.K_POINT[0]
    v = abs(lowdin(ctx, m, model, q="v"))                         # eV A
    best = None
    for s in (1, -1):
        e = np.linalg.eigvalsh(ref.H_block(K, 0.0, s, 0.0, p=p))
        if best is None or e[0] > best[1]:
            best = (s, e[0])
    s = best[0]

    def bands(qx, qy):                                           # q in 1/A
        e = np.linalg.eigvalsh(ref.H_block(K + qx * a, qy * a, s, 0.0, p=p))
        return e[..., 1], e[..., 0]
    Ec0, Ev0 = bands(np.array([0.0]), np.array([0.0])); Eg = float(Ec0[0] - Ev0[0])
    qs = np.linspace(0.004, 0.03, 8); ph = 2 * np.pi * np.arange(48) / 48; F = []
    for q in qs:
        Ec, Ev = bands(q * np.cos(ph), q * np.sin(ph)); d2 = ((Ec - Ev) / 2) ** 2
        F.append(2 * np.mean(d2 * np.cos(3 * ph)) / q ** 3)
    c3 = float(np.polyval(np.polyfit(qs ** 2, F, 2), 0))
    # Kormanyos-style simultaneous fit of E_c, E_v along Gamma-K-M over 16% of |Gamma K|
    from scipy.optimize import least_squares
    GK = 4 * np.pi / (3 * a); qq = np.linspace(-0.16 * GK, 0.16 * GK, 81); qq = qq[qq != 0]
    Ec, Ev = bands(qq, 0 * qq); em = (Ec0[0] + Ev0[0]) / 2
    c1 = np.polyval(np.polyfit(qs ** 2, [np.mean(sum(bands(q * np.cos(ph), q * np.sin(ph))) / 2 - em) / q ** 2 for q in qs], 2), 0)
    c2 = np.polyval(np.polyfit(qs ** 2, [np.mean(((lambda c, v_: ((c - v_) / 2) ** 2)(*bands(q * np.cos(ph), q * np.sin(ph)))) - Eg ** 2 / 4) / q ** 2
                                          for q in qs], 2), 0)
    Q, cs = np.abs(qq), np.sign(qq)

    def res(pp):
        root = np.sqrt(np.maximum(Eg ** 2 / 4 + c2 * Q ** 2 + pp[0] * Q ** 3 * cs + pp[1] * Q ** 4, 1e-12))
        return np.concatenate([em + c1 * Q ** 2 + root - Ec, em + c1 * Q ** 2 - root - Ev])
    c3fit = float(least_squares(res, [c3, 0.0]).x[0])
    ctx.memo[key] = out = dict(kappa_disp=c3 / (2 * v), kappa_fit16=c3fit / (2 * v))
    return out


def tb_warp(ctx, q, agg, model="TNN"):
    return _agg([_tb_warp(ctx, m, model)[q] for m in MAT4], agg) if agg != "mean" else float(np.mean([_tb_warp(ctx, m, model)[q] for m in MAT4]))


# ------------------------------------------------------------------------------------------------ discussion estimate
def current_nA(ctx, chi=1e-3, B=30.0, tau=1e-12, P=1e-3, area=1e-12, width=1e-6, convention="sipe"):
    """j = tau chi B |E|^2 across a channel of the given width. I = P/area; Sipe convention E(t) = E(w)e^{-iwt} + c.c.
    gives |E(w)|^2 = I/(2 c eps0); the peak-amplitude convention gives E0^2 = 2I/(c eps0)."""
    I = P / area
    E2 = I / (2 * C0 * EPS0) if convention == "sipe" else 2 * I / (C0 * EPS0)
    return float(tau * chi * B * E2 * width * 1e9)


def count(ctx, n):
    return float(n)


RECIPES = {k: v for k, v in globals().items() if callable(v) and not isinstance(v, type) and not k.startswith("_") and k not in ("os", "json")}
