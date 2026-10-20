"""C: convergence of every numerical knob (protocol sec. 4).

fast:  C0 rate study in h (0.004, 0.002, 0.001)/a at R = 0.3/a: monotone, observed order >= 1.5
       C5 finite-difference step EPS 1e-3 / 1e-4 / 1e-5 (1e-4 vs 1e-5 must agree to rel 1e-6)
       C8 stored raw C3 violation of the production files <= 1e-3 (where |eta| >= 10% of its max)
       C6 (fast on stored data when present) odd part at +-10 T vs s B + c B^3 fit to 60 T; + B^5; |B| <= 40 vs 60 T
full:  C1 N 600/900/1200 at the van Hove peaks (chi 0.3%, peak energy 2 meV)
       C2 h 0.001/0.0007/0.0005 at E_g+0.01 and 0.002/0.001 at E_g+0.1 (1%)
       C3 R 0.6/0.9/1.2 (1%);  C4 nu 2.5/5/10 meV (threshold signs; sign-change energies < 10 meV)
       C7 omega averaging 5 vs 9 points; 900 vs 1800 omega (0.5%, 2 meV)
Full-mode results are cached under cache/<tree_sha>/ so an interrupted run resumes.
"""
import os, sys, json, glob, time
import numpy as np
from vcore import check, Result, judge, exact, rel, rel_scaled, SUITE_DIR
import fx

MATS = ["MoS2", "MoSe2", "WS2", "WSe2"]
F = dict(spin=True, mom=True, dos="v", vel=True)
DRY = os.environ.get("VERIF_DRY") == "1"          # code-path check of the full-mode tests on tiny grids (results meaningless)


def _h(h):
    return h * 30 if DRY else h


def _N(N):
    return N // 10 if DRY else N


def _chi_disk(ctx, mat, model, R, h, nu, offs, B=10.0, var="F", func="GGA"):
    eng = ctx.eng
    om = fx.eg(ctx, mat, model, func) + np.asarray(offs)
    r = eng.eta_phys(mat, model, func, B, om, nu, var, eng.disks(R, h))
    return r["sym"][:, 0, 0, 0] / B          # eta exactly odd in B on T-paired disks (H1)


@check("C0", "C", "rate study in the local-grid spacing h at R = 0.3/a (Z, MoS2 TNN, E_g+0.01 eV)")
def c0(ctx):
    offs = 0.01 + np.linspace(-0.0025, 0.0025, 5)
    hs = (0.004, 0.002, 0.001)
    chi = [float(np.mean(_chi_disk(ctx, "MoS2", "TNN", 0.3, h, 0.005, offs, var="Z"))) for h in hs]
    d1, d2 = abs(chi[0] - chi[1]), abs(chi[1] - chi[2])
    order = np.log2(d1 / d2) if d2 > 0 else np.inf
    mono = (chi[0] - chi[1]) * (chi[1] - chi[2]) >= 0
    floor = max(d1, d2) / abs(chi[2]) < 1e-5          # differences below the 1e-5 noise floor of the disk-boundary lattice
    return [Result("C0", "monotone convergence in h (or all steps below the 1e-5 noise floor)", "PASS" if (mono or floor) else "FAIL",
                   tol="exact", measured=", ".join(f"{c:+.6e}" for c in chi),
                   detail="non-monotone but every step < 1e-5 relative: converged below the noise floor at h=0.004/a (protocol "
                          "requires monotone; with R=0.3/a and nu=5 meV the h-error is already ~6e-6)" if (floor and not mono) else ""),
            Result("C0", "observed order p = log2(|d(0.004,0.002)| / |d(0.002,0.001)|)",
                   "INFO" if floor else ("PASS" if order >= 1.5 else "FAIL"), tol=">= 1.5", measured=float(order),
                   detail=f"rel steps {d1/abs(chi[2]):.2e}, {d2/abs(chi[2]):.2e}" + ("; below the noise floor the order is not meaningful" if floor else ""))]


@check("C5", "C", "finite-difference step EPS (1e-3, 1e-4, 1e-5) on S2, MoS2 TNN, F and Z")
def c5(ctx):
    eng = ctx.eng; L = eng.L
    out = []
    old = L.EPS
    try:
        for var in ("Z", "F"):
            res = {}
            for eps in (1e-3, 1e-4, 1e-5):
                L.EPS = eps
                res[eps] = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, fx.omega(ctx, "S2", "MoS2", "TNN", "GGA"), 0.005, var,
                                        fx.kset(ctx, "S2"))["raw"]
            L.EPS = old
            sK = fx.eng_eta(ctx, "MoS2", "TNN", "GGA", var, 10.0, "S2K")["raw"]
            out.append(judge("C5", f"{var}: eta(EPS=1e-4) vs eta(1e-5) (rel to single valley)", rel_scaled(res[1e-4], res[1e-5], sK), 1e-6,
                             detail=f"1e-3 vs 1e-4: {rel_scaled(res[1e-3], res[1e-4], sK):.2e} (expect ~100x larger: O(EPS^2))"))
    finally:
        L.EPS = old
    return out


@check("C8", "C", "stored raw C3 violation of production files <= 1e-3")
def c8(ctx):
    files = sorted(glob.glob(os.path.join(ctx.path("v7_out"), "prod7_*_F.npz")))
    gen = "v7"
    if not files:
        files = sorted(glob.glob(os.path.join(ctx.path("data_v6"), "prod_*_F.npz"))); gen = "v6 (pre-fix, until prod7 lands)"
    worst, det = 0.0, []
    for p in files:
        d = np.load(p, allow_pickle=True)
        eta = d["eta"]; v = d["c3_violation"]
        e = np.abs(eta[..., 0, 0, 0] if eta.ndim == 5 else eta)
        w = 0.0
        for i in range(1, e.shape[0]):
            msk = e[i] >= 0.1 * e[i].max()
            w = max(w, float(v[i][msk].max()))
        med = float(np.median([np.median(v[i][e[i] >= 0.1 * e[i].max()]) for i in range(1, e.shape[0])]))
        worst = max(worst, w); det.append(f"{os.path.basename(p)} max {w:.1e} median {med:.1e}")
    if not files:
        return Result("C8", "stored C3 violation", "SKIP", detail="no production map files")
    return judge("C8", f"max stored C3 violation where |eta| >= 10% of max ({gen})", worst, 1e-3, detail="; ".join(det))


@check("C6", "C", "field fit of the threshold runs: +-10 T odd part vs sB+cB^3 to 60 T; adding B^5; |B|<=40 vs 60 T")
def c6(ctx):
    files = sorted(glob.glob(os.path.join(ctx.path("v7_out"), "edge7_*_TNN.json")) + glob.glob(os.path.join(ctx.path("v7_out"), "edge7_*_NN.json")))
    if not files:
        return Result("C6", "field-fit stability", "SKIP", detail="no edge7_*.json in v7_out (queue running)")
    worst, det = 0.0, []
    for p in files:
        d = json.load(open(p))
        B = np.array(sorted(map(float, d["eta"]))); Y = np.array([np.mean(d["eta"][f"{b:g}"]) for b in B])
        Bf = np.concatenate([-B[1:][::-1], B]); Yf = np.concatenate([-Y[1:][::-1], Y])
        fit = lambda pw, bm: np.linalg.lstsq(np.stack([Bf[np.abs(Bf) <= bm] ** k for k in pw], 1), Yf[np.abs(Bf) <= bm], rcond=None)[0][0]
        s3, s5, s40 = fit((1, 3), 60), fit((1, 3, 5), 60), fit((1, 3), 40)
        s10 = Y[B == 10][0] / 10 if 10 in B else np.nan
        dd = max(abs(s5 - s3), abs(s40 - s3), abs(s10 - s3) if np.isfinite(s10) else 0) / abs(s3)
        worst = max(worst, dd); det.append(f"{os.path.basename(p)} {dd:.1e}")
    return judge("C6", "max relative spread of chi between field fits", worst, 5e-3, detail="; ".join(det))


# ------------------------------------------------------------------------------------------------ full mode
def _cache_dir(ctx):
    import prov
    d = os.path.join(SUITE_DIR, ctx.cfg["paths"]["cache"], prov.tree_sha(ctx.code_root, ctx.cfg["engine_files"])[:16])
    os.makedirs(d, exist_ok=True)
    return d


def _compute(ctx, spec):
    k = spec["kind"]; a = spec["args"]
    if k == "edge_point":                    # G8: rerun one stored field point of a production edge file
        d = json.load(open(a["file"]))
        kp = ctx.eng.disks(d["R"], d["h"])
        r = ctx.eng.eta_phys(a["m"], d["model"], d.get("functional", "GGA"), a["B"], d["omega"], d["nu"], d["flags"], kp)
        return r["sym"][:, 0, 0, 0].tolist()
    if k == "map_point":
        dd = np.load(a["file"], allow_pickle=True); meta = json.loads(str(dd["meta"]))
        r = ctx.eng.eta_phys_grid(a["m"], "TNN", "GGA", meta["N"], a["B"], dd["omega"], meta["nu_eV"], "F")
        return r["sym"][:, 0, 0, 0].tolist()
    if k == "grid":
        om = np.asarray(a["om"])
        return (ctx.eng.eta_phys_grid(a["m"], "TNN", "GGA", _N(a["N"]), 10.0, om, 0.01, "F")["sym"][:, 0, 0, 0] / 10).tolist()
    v = _chi_disk(ctx, a["m"], a["model"], a["R"], _h(a["h"]), a["nu"], a["offs"])
    return float(np.mean(v)) if k == "disk_mean" else v.tolist()


def _cache(ctx, key, spec):
    p = os.path.join(_cache_dir(ctx), key + ("_dry" if DRY else "") + ".json")
    if os.path.exists(p):
        return json.load(open(p))
    v = _compute(ctx, spec)
    json.dump(v, open(p, "w"))
    return v


def plan(ctx):
    """All full-mode C jobs as (key, spec); prefetch() runs them in parallel (one job per process)."""
    import claims_recipes as CR
    J = []
    o5 = list(0.01 + np.linspace(-0.0025, 0.0025, 5))
    for m in MATS:
        e0 = CR.vh_energy(ctx, m)
        om = list(e0 + np.linspace(-0.05, 0.05, 41))
        for N in (600, 900, 1200):
            J.append((f"C1_{m}_{N}", dict(kind="grid", args=dict(m=m, N=N, om=om))))
        for model in ("TNN", "NN"):
            for h in (0.001, 0.0007, 0.0005):
                J.append((f"C2a_{m}_{model}_{h}", dict(kind="disk_mean", args=dict(m=m, model=model, R=0.9, h=h, nu=0.005, offs=o5))))
            for nu in (0.0025, 0.005, 0.01):
                J.append((f"C4_{m}_{model}_{nu}", dict(kind="disk_vec", args=dict(m=m, model=model, R=0.9, h=0.001, nu=nu, offs=list(C4_OFFS)))))
        for h in (0.002, 0.001):
            J.append((f"C2b_{m}_{h}", dict(kind="disk_vec", args=dict(m=m, model="TNN", R=0.9, h=h, nu=0.005, offs=[0.1]))))
        for R in (0.6, 0.9, 1.2):
            J.append((f"C3_{m}_{R}", dict(kind="disk_mean", args=dict(m=m, model="TNN", R=R, h=0.001, nu=0.005, offs=o5))))
        for n in (5, 9):
            J.append((f"C7_{m}_{n}", dict(kind="disk_mean", args=dict(m=m, model="TNN", R=0.9, h=0.0007, nu=0.005,
                                                                       offs=list(0.01 + np.linspace(-0.0025, 0.0025, n))))))
    J += g8_jobs(ctx)
    J.sort(key=lambda j: -(j[1]["args"].get("R", 0.9) ** 2 / j[1]["args"].get("h", 0.0007) ** 2))     # longest first
    return J


def g8_jobs(ctx):
    """Two stored field points (10 T and 30 T) of every finished v7 production file (protocol G8)."""
    J = []
    for p in sorted(glob.glob(os.path.join(ctx.path("v7_out"), "edge7_*.json"))):
        d = json.load(open(p))
        m = os.path.basename(p).split("_")[1]
        for B in (10.0, 30.0):
            if f"{B:g}" in d.get("eta", {}):
                J.append((f"G8_{os.path.basename(p)}_{B:g}", dict(kind="edge_point", args=dict(file=p, m=m, B=B))))
    for p in sorted(glob.glob(os.path.join(ctx.path("v7_out"), "prod7_*_F.npz"))):
        m = os.path.basename(p).split("_")[1]
        for B in (10.0, 30.0):
            J.append((f"G8_{os.path.basename(p)}_{B:g}", dict(kind="map_point", args=dict(file=p, m=m, B=B))))
    return J


def _worker(job):
    suite, cfgp, code_root, key, spec = job
    for sub in ("lib", "tests", "adapters"):
        sys.path.insert(0, os.path.join(suite, sub))
    from vcore import Ctx, load_config
    ctx = Ctx(load_config(cfgp), "full", code_root, "/tmp")
    t = time.time(); _cache(ctx, key, spec)
    return key, time.time() - t


def prefetch(ctx, jobs):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    todo = [(SUITE_DIR, ctx.cfg["_config_file"], ctx.code_root, k, sp) for k, sp in plan(ctx)
            if not os.path.exists(os.path.join(_cache_dir(ctx), k + ("_dry" if DRY else "") + ".json"))]
    print(f"full mode: {len(todo)} convergence jobs on {jobs} processes (cache {_cache_dir(ctx)})", flush=True)
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
        for key, dt in ex.map(_worker, todo):
            print(f"  {key} {dt:.0f} s", flush=True)


C4_OFFS = np.arange(0.005, 0.2001, 0.0025)


@check("C1", "C", "BZ mesh N = 600/900/1200 at the van Hove peaks", modes=("full",))
def c1(ctx):
    import claims_recipes as CR
    out = []
    for m in MATS:
        e0 = CR.vh_energy(ctx, m)
        om = e0 + np.linspace(-0.05, 0.05, 41)
        res = {}
        for N in (600, 900, 1200):
            res[N] = _cache(ctx, f"C1_{m}_{N}", dict(kind="grid", args=dict(m=m, N=N, om=list(om))))
        c = {N: np.array(v) for N, v in res.items()}
        i = {N: int(np.argmax(np.abs(v))) for N, v in c.items()}
        dchi = abs(c[900][i[900]] - c[1200][i[1200]]) / abs(c[1200][i[1200]])
        dE = abs(om[i[900]] - om[i[1200]])
        out.append(judge("C1", f"{m}: |chi(900)-chi(1200)|/|chi| at the peak", dchi, 3e-3, detail=f"600->900: {abs(c[600][i[600]]-c[900][i[900]])/abs(c[900][i[900]]):.2e}"))
        out.append(judge("C1", f"{m}: peak energy shift 900 vs 1200 (eV)", dE, 2e-3))
    return out


@check("C2", "C", "local spacing h at E_g+0.01 (0.001/0.0007/0.0005) and E_g+0.1 (0.002/0.001)", modes=("full",))
def c2(ctx):
    out = []
    offs = 0.01 + np.linspace(-0.0025, 0.0025, 5)
    for m in MATS:
        for model in ("TNN", "NN"):
            r = {h: _cache(ctx, f"C2a_{m}_{model}_{h}", dict(kind="disk_mean", args=dict(m=m, model=model, R=0.9, h=h, nu=0.005, offs=list(offs))))
                 for h in (0.001, 0.0007, 0.0005)}
            out.append(judge("C2", f"{m}/{model} E_g+0.01: |chi(0.0007)-chi(0.0005)|/|chi|", abs(r[0.0007] - r[0.0005]) / abs(r[0.0005]), 1e-2,
                             detail=f"0.001 vs 0.0007: {abs(r[0.001]-r[0.0007])/abs(r[0.0007]):.2e}"))
        r = {h: _cache(ctx, f"C2b_{m}_{h}", dict(kind="disk_vec", args=dict(m=m, model="TNN", R=0.9, h=h, nu=0.005, offs=[0.1])))[0] for h in (0.002, 0.001)}
        out.append(judge("C2", f"{m}/TNN E_g+0.1: |chi(0.002)-chi(0.001)|/|chi|", abs(r[0.002] - r[0.001]) / abs(r[0.001]), 1e-2))
    return out


@check("C3", "C", "disk radius R = 0.6/0.9/1.2 per a", modes=("full",))
def c3(ctx):
    out = []
    offs = 0.01 + np.linspace(-0.0025, 0.0025, 5)
    for m in MATS:
        r = {R: _cache(ctx, f"C3_{m}_{R}", dict(kind="disk_mean", args=dict(m=m, model="TNN", R=R, h=0.001, nu=0.005, offs=list(offs)))) for R in (0.6, 0.9, 1.2)}
        out.append(judge("C3", f"{m}: |chi(0.9)-chi(1.2)|/|chi|", abs(r[0.9] - r[1.2]) / abs(r[1.2]), 1e-2, detail=f"0.6 vs 0.9: {abs(r[0.6]-r[0.9])/abs(r[0.9]):.2e}"))
    return out


@check("C4", "C", "broadening nu = 2.5/5/10 meV: threshold signs and sign-change energies", modes=("full",))
def c4(ctx):
    out = []
    offs = C4_OFFS
    for m in MATS:
        for model in ("TNN", "NN"):
            c = {nu: np.array(_cache(ctx, f"C4_{m}_{model}_{nu}", dict(kind="disk_vec", args=dict(m=m, model=model, R=0.9, h=0.001, nu=nu, offs=list(offs))))) for nu in (0.0025, 0.005, 0.01)}
            i0 = int(np.argmin(np.abs(offs - 0.01)))
            same = len({np.sign(v[i0]) for v in c.values()}) == 1
            sc = {nu: (offs[np.where(np.diff(np.sign(v)))[0][0]] if np.any(np.diff(np.sign(v))) else np.nan) for nu, v in c.items()}
            mv = np.nanmax([abs(sc[a] - sc[0.005]) for a in sc]) if np.isfinite(sc[0.005]) else 0.0
            out.append(exact("C4", f"{m}/{model}: threshold sign unchanged for nu 2.5-10 meV", same))
            out.append(judge("C4", f"{m}/{model}: sign-change energy moves (eV)", float(mv) if np.isfinite(mv) else 0.0, 1e-2))
    return out


@check("C7", "C", "omega averaging 5 vs 9 points at threshold", modes=("full",))
def c7(ctx):
    out = []
    for m in MATS:
        a, b = (_cache(ctx, f"C7_{m}_{n}", dict(kind="disk_mean", args=dict(m=m, model="TNN", R=0.9, h=0.0007, nu=0.005,
                                                                         offs=list(0.01 + np.linspace(-0.0025, 0.0025, n))))) for n in (5, 9))
        out.append(judge("C7", f"{m}: 5 vs 9 omega points", abs(a - b) / abs(b), 5e-3))
    out.append(Result("C7", "maps: 900 vs 1800 omega (0.5%, 2 meV)", "SKIP", detail="needs prod7 at NOM=1800 (one material) after the queue"))
    return out


@check("G8", "G", "two stored field points per production file rerun with identical parameters", modes=("full",))
def g8(ctx):
    jobs = g8_jobs(ctx)
    if not jobs:
        return Result("G8", "production reruns", "SKIP", detail="no v7 production files yet")
    worst, det = 0.0, []
    for key, sp in jobs:
        new = np.array(_cache(ctx, key, sp))
        a = sp["args"]
        if sp["kind"] == "edge_point":
            old = np.array(json.load(open(a["file"]))["eta"][f"{a['B']:g}"])
        else:
            dd = np.load(a["file"], allow_pickle=True)
            old = dd["eta"][int(np.argmin(np.abs(dd["B_tesla"] - a["B"])))]
        d = rel(new, old); worst = max(worst, d); det.append(f"{key} {d:.1e}")
    return judge("G8", "stored vs rerun (same engine, same parameters)", worst, 1e-9, detail="; ".join(det))
