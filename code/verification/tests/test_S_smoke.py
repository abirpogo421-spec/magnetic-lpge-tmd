"""S: script smoke tests. Each runs a temporary copy of the real production script with reduced argv
(and the exact-string size patches of smoke_patches.json), then checks its stored output against the reference.
Without these, sign, mirroring and b bugs in the run scripts are invisible to the engine tests.
The reference always uses its OWN area element (h^2, BZ_AREA/N^2), never the engine's.

S1 edge_v7.py for MoS2 TNN GGA, MoS2 NN GGA and MoS2 TNN LDA (0.03/a, 30 T steps): stored eta vs the reference for that
   model AND functional; stored E_g vs the reference gap of that model/functional; chi_SI_per_T vs an independent odd
   extension + lstsq; stored h, R, nu, functional, model, flags.  (Also run for the other materials to feed S6.)
S2 maps_v7.py MoS2 24 40 10 0.03   : eta vs reference on the engine grid N=24 (Emax grid 401 -> 101)
S3 chiw9_v7.py MoS2 TNN F 0.03     : chi vs reference
S4 decomp_v7.py MoS2, three isolated runs:
   (a) no prod7: vH = the fallback table; every variant vs the reference AT THE vH IN ITS OWN META; the fallback table
       equals the registered vH claims (vH.E.*);
   (b) a finished synthetic prod7 with a dominant NEGATIVE lobe at 60 T: vH = omega of max |eta(60 T)| of that file;
   (c) an unfinished prod7 (meta done < len(B), eta(60 T) = 0): must be refused (no vH = omega[0])
S5 lam_v7.py MoS2 F lam            : stored chi (code sign) x (-2) vs reference physical chi (first two lambda values)
S6 kp_fp_all_v7.py (Nk 401 -> 101) : chi vs reference two-band sum with inputs rebuilt from literature/; then the figure
   scripts v7jobs/fig_v7.py and fig_v7_all.py run on the smoke outputs under tests/figrecord.py, and the PLOTTED arrays
   are read back: odd mirroring y(-B) = -y(B), x(-B) = -x(B), and equality with the stored data (Figs. 7, 8, 9, 11)
"""
import os, sys, json, ast, shutil, subprocess, time
import numpy as np
from vcore import check, Result, judge, exact, rel, rel_scaled, SUITE_DIR

MUB = 5.7883818060e-5
FLAGS_F = dict(spin=True, mom=True, dos="v", vel=True)
MATS = ["MoS2", "MoSe2", "WS2", "WSe2"]
EDGE_RUNS = [(m, "TNN", "GGA") for m in MATS] + [(m, "NN", "GGA") for m in MATS] + [("MoS2", "TNN", "LDA")]
S1_CHECKED = [("MoS2", "TNN", "GGA"), ("MoS2", "NN", "GGA"), ("MoS2", "TNN", "LDA")]


def script_path(ctx, rel_):
    """Location of a producing script: code_root/rel unless config 'script_paths' overrides it."""
    o = ctx.cfg.get("script_paths", {}).get(rel_)
    if o:
        return o if os.path.isabs(o) else os.path.normpath(os.path.join(SUITE_DIR, o))
    return os.path.join(ctx.code_root, rel_)


def smoke_tree(ctx):
    """Copy the code under test (scripts only) to scratch and apply the smoke patches. Returns (root, stale list)."""
    def build():
        root = os.path.join(ctx.scratch, "smoke_tree")
        if os.path.exists(root):
            shutil.rmtree(root)
        for sub in ("pub_pipeline", "kp", "v7jobs"):
            os.makedirs(os.path.join(root, sub))
            for f in os.listdir(os.path.join(ctx.code_root, sub)):
                if f.endswith((".py", ".sh")):
                    shutil.copy2(os.path.join(ctx.code_root, sub, f), os.path.join(root, sub, f))
        fa = script_path(ctx, "v7jobs/fig_v7_all.py")
        if os.path.exists(fa):
            shutil.copy2(fa, os.path.join(root, "v7jobs", "fig_v7_all.py"))
        stale = []
        for pt in json.load(open(os.path.join(SUITE_DIR, "tests", "smoke_patches.json")))["patches"]:
            p = os.path.join(root, pt["file"]); s = open(p).read()
            n = s.count(pt["old"])
            if n != 1:
                stale.append(f"{pt['file']}: patch '{pt['old'][:40]}' occurs {n} times")
                continue
            open(p, "w").write(s.replace(pt["old"], pt["new"]))
        return root, stale
    return ctx.cached("smoke_tree", build)


def isolated_copy(ctx, name):
    """A private copy of the smoke tree (pub_pipeline + v7jobs scripts only, no outputs)."""
    root, stale = smoke_tree(ctx)
    d = os.path.join(ctx.scratch, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    for sub in ("pub_pipeline", "v7jobs", "kp"):
        os.makedirs(os.path.join(d, sub))
        for f in os.listdir(os.path.join(root, sub)):
            if f.endswith((".py", ".sh")):
                shutil.copy2(os.path.join(root, sub, f), os.path.join(d, sub, f))
    return d


def run_script(ctx, rel_script, argv, timeout=600, root=None, wrapper=None):
    r0, stale = smoke_tree(ctx)
    root = root or r0
    cwd = os.path.join(root, os.path.dirname(rel_script))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
    t = time.time()
    cmd = [sys.executable] + (wrapper or []) + [os.path.basename(rel_script)] + [str(a) for a in argv]
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return dict(rc=p.returncode, out=p.stdout, err=p.stderr[-2000:], cwd=cwd, seconds=time.time() - t,
                stale=[s for s in stale if s.startswith(rel_script)])


def _stale_or_fail(id, name, r):
    if r["stale"]:
        return Result(id, name, "STALE", detail="; ".join(r["stale"]))
    if r["rc"] != 0:
        return Result(id, name, "FAIL", detail=f"script exit {r['rc']}: {r['err'][-600:]}")
    return None


def _ref():
    sys.path.insert(0, os.path.join(SUITE_DIR, "adapters"))
    import refside
    return refside


# ------------------------------------------------------------------------------------------------- S1
def edge_runs(ctx):
    def build():
        return {(m, mo, f): run_script(ctx, "v7jobs/edge_v7.py", [m, mo, 0.03, 30, f]) for m, mo, f in EDGE_RUNS}
    return ctx.cached("smoke_edge", build)


def edge_file(cwd, m, mo, f):
    return os.path.join(cwd, f"edge7_{m}_{mo}.json" if f == "GGA" else f"edge7_{m}_{mo}_{f}.json")


def _odd_fit(eta_dict):
    B = np.array(sorted(float(b) for b in eta_dict))
    Y = np.array([np.mean(eta_dict[f"{b:g}"]) for b in B])
    assert B[0] == 0.0
    Bf = np.concatenate([-B[1:][::-1], B]); Yf = np.concatenate([-Y[1:][::-1], Y])
    A = np.stack([Bf, Bf ** 3], 1)
    return np.linalg.lstsq(A, Yf, rcond=None)[0][0]


@check("S1", "S", "edge_v7.py smoke (TNN GGA, NN GGA, TNN LDA): eta and E_g vs reference of that model/functional, chi fit, parameters")
def s1(ctx):
    runs = edge_runs(ctx)
    RS = _ref()
    kp = ctx.eng.disks(0.9, 0.03)
    kK, kKp = ctx.eng.split_valleys(kp)
    dA = RS.disk_dA(0.03)
    out = []
    for m, mo, f in S1_CHECKED:
        tag = f"{m} {mo} {f}"
        r = runs[(m, mo, f)]
        bad = _stale_or_fail("S1", f"edge_v7 run {tag}", r)
        if bad:
            out.append(bad); continue
        d = json.load(open(edge_file(r["cwd"], m, mo, f)))
        om = np.array(d["omega"])
        worst = 0.0; sc = None
        for b in (30.0, 60.0):
            rKv = RS.ref_eta(m, mo, f, b, om, 0.005, "F", kK, c3=True, dA=dA)[:, 0, 0, 0]
            rKp = RS.ref_eta(m, mo, f, b, om, 0.005, "F", kKp, c3=True, dA=dA)[:, 0, 0, 0]
            sc = rKv if sc is None else sc
            worst = max(worst, rel_scaled(np.array(d["eta"][f"{b:g}"]), rKv + rKp, rKv))
        out.append(judge("S1", f"{tag}: stored eta(B=30,60 T) vs reference (own dA, C3-averaged k-set)", worst, 1e-6))
        out.append(judge("S1", f"{tag}: stored eta(B=0) = 0", float(np.max(np.abs(d["eta"]["0"]))) / max(1e-300, float(np.max(np.abs(sc)))), 1e-10))
        s = _odd_fit(d["eta"])
        out.append(judge("S1", f"{tag}: chi_SI_per_T vs independent odd extension + lstsq", abs(d["chi_SI_per_T"] - s) / abs(s), 1e-12,
                         detail=f"stored {d['chi_SI_per_T']:+.6e} independent {s:+.6e}"))
        okp = (d["h"] == 0.03 and d["R"] == 0.9 and d["nu"] == 0.005 and d["functional"] == f and d["model"] == mo
               and d["flags"] == FLAGS_F and len(om) == 5 and abs(om[2] - (d["Eg"] + 0.01)) < 1e-12)
        out.append(exact("S1", f"{tag}: stored h, R, nu, functional, model, flags, omega = argv / script settings", okp,
                         detail=f"h={d['h']} R={d['R']} nu={d['nu']} FUNC={d['functional']} model={d['model']}"))
        eg_ref = RS.eg(m, mo, f)
        out.append(judge("S1", f"{tag}: script E_g == reference direct gap at K of the same model/functional (eV)", abs(d["Eg"] - eg_ref), 1e-9,
                         detail=f"script {d['Eg']:.6f} ref {eg_ref:.6f}"))
    return out


# ------------------------------------------------------------------------------------------------- S2
def maps_runs(ctx):
    def build():
        return {m: run_script(ctx, "v7jobs/maps_v7.py", [m, 24, 40, 10, 0.03]) for m in MATS}
    return ctx.cached("smoke_maps", build)


@check("S2", "S", "maps_v7.py smoke: eta vs reference on the engine grid N=24")
def s2(ctx):
    r = maps_runs(ctx)["MoS2"]
    bad = _stale_or_fail("S2", "maps_v7 run", r)
    if bad:
        return bad
    RS = _ref()
    d = np.load(os.path.join(r["cwd"], "prod7_MoS2_F.npz"), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    kp = ctx.eng.grid_cart(24)
    worst = 0.0
    for i, b in enumerate(d["B_tesla"]):
        if b == 0 or b not in (30.0, 60.0):
            continue
        rr = RS.ref_eta("MoS2", "TNN", "GGA", float(b), d["omega"], 0.03, "F", kp, c3=True, dA=RS.grid_dA(24))[:, 0, 0, 0]
        worst = max(worst, rel(d["eta"][i], rr))
    out = [judge("S2", "stored eta(omega, B=30,60 T) vs reference (own dA), MoS2 N=24 nu=30 meV", worst, 1e-4,
                 detail="full-BZ grid with the velocity shift: FD-limited near c/u near-degeneracies in both codes (see R4 S3); protocol 1e-6")]
    out.append(exact("S2", "stored N, nu, flags, units, done", meta["N"] == 24 and meta["nu_eV"] == 0.03 and meta["flags"] == FLAGS_F
                     and "physical sign" in meta["units"] and meta["done"] == len(d["B_tesla"]),
                     detail=str({k: meta[k] for k in ("N", "nu_eV", "units", "done")})))
    return out


# ------------------------------------------------------------------------------------------------- S3
@check("S3", "S", "chiw9_v7.py smoke: chi(omega) vs reference")
def s3(ctx):
    r = run_script(ctx, "v7jobs/chiw9_v7.py", ["MoS2", "TNN", "F", 0.03])
    bad = _stale_or_fail("S3", "chiw9_v7 run", r)
    if bad:
        return bad
    RS = _ref()
    d = json.load(open(os.path.join(r["cwd"], "chiwR9v7_MoS2_TNN_F_0.03.json")))
    om = d["Eg"] + np.array(d["off"])
    kp = ctx.eng.disks(0.9, 0.03); kK, kKp = ctx.eng.split_valleys(kp); dA = RS.disk_dA(0.03)
    rK = RS.ref_eta("MoS2", "TNN", "GGA", 30.0, om, 0.005, "F", kK, c3=True, dA=dA)[:, 0, 0, 0] / 30
    rKp = RS.ref_eta("MoS2", "TNN", "GGA", 30.0, om, 0.005, "F", kKp, c3=True, dA=dA)[:, 0, 0, 0] / 30
    return [judge("S3", "stored chi(E_g-0.03..E_g+0.45) vs reference eta(30 T)/30 T", rel_scaled(np.array(d["chi"]), rK + rKp, rK), 1e-6),
            exact("S3", "stored h, R, nu, B, flags", d["h"] == 0.03 and d["R"] == 0.9 and d["nu"] == 0.005 and d["B"] == 30.0
                  and d["flags"] == FLAGS_F),
            judge("S3", "stored E_g == reference gap (eV)", abs(d["Eg"] - RS.eg("MoS2")), 1e-9)]


# ------------------------------------------------------------------------------------------------- S4
def _vh_table(path):
    tree = ast.parse(open(path).read())
    for st in tree.body:
        if isinstance(st, ast.Assign) and getattr(st.targets[0], "id", "") == "VH":
            return ast.literal_eval(st.value)
    return None


def _write_prod7(path, done, negative_lobe=True):
    om = np.linspace(2.5, 3.8, 27); B = np.array([0.0, 20.0, 40.0, 60.0])
    eta = np.zeros((4, len(om)))
    for i, b in enumerate(B[:done]):
        eta[i, 5] = +1.0 * b / 60; eta[i, 12] = +0.4 * b / 60
        eta[i, 20] = (-2.5 if negative_lobe else -0.5) * b / 60      # dominant negative lobe at omega[20]
    meta = dict(material="MoS2", N=24, nu_eV=0.03, done=int(done), lpge_sha256="fixture", units="eta in A m V^-2 s^-1 (physical sign)")
    np.savez(path, B_tesla=B, omega=om, eta=eta, c3_violation=np.zeros_like(eta), meta=json.dumps(meta))
    return om


@check("S4", "S", "decomp_v7.py smoke: variants vs reference at its own vH; vH pick from prod7; partial prod7 refused; fallback table")
def s4(ctx):
    RS = _ref()
    out = []
    # (a) isolated, no prod7
    da = isolated_copy(ctx, "decomp_a")
    r = run_script(ctx, "v7jobs/decomp_v7.py", ["MoS2"], root=da)
    bad = _stale_or_fail("S4", "decomp_v7 run (no prod7)", r)
    VH = _vh_table(os.path.join(da, "v7jobs", "decomp_v7.py"))
    if bad:
        out.append(bad)
    else:
        d = json.load(open(os.path.join(r["cwd"], "decomp7_MoS2.json")))
        Eg = d["meta"]["Eg"]; vh = d["meta"]["vH_eV"]
        out.append(exact("S4", "(a) no prod7: vH in meta == fallback table value", VH is not None and vh == VH["MoS2"],
                         measured=f"meta {vh}, table {VH and VH.get('MoS2')}"))
        for where, om, kp, nu, dA in (("vH", vh, ctx.eng.grid_cart(24), 0.01, RS.grid_dA(24)),
                                      ("edge", Eg + 0.1, ctx.eng.disks(0.3, 0.03), 0.005, RS.disk_dA(0.03))):
            worst, det = 0.0, []
            kK = ctx.eng.split_valleys(kp)[0] if where == "edge" else None
            for v in ("Z", "D", "E", "V", "ED", "EV", "F"):
                cp = RS.ref_eta("MoS2", "TNN", "GGA", 10.0, [om], nu, v, kp, c3=True, dA=dA)[0, 0, 0, 0]
                cm = RS.ref_eta("MoS2", "TNN", "GGA", -10.0, [om], nu, v, kp, c3=True, dA=dA)[0, 0, 0, 0]
                cr = (cp - cm) / 20
                sc = abs(RS.ref_eta("MoS2", "TNN", "GGA", 10.0, [om], nu, v, kK, c3=True, dA=dA)[0, 0, 0, 0]) / 10 if kK else abs(cr)
                dd = abs(d[f"MoS2|{where}|{v}"] - cr) / sc
                worst = max(worst, dd); det.append(f"{v}:{d[f'MoS2|{where}|{v}']:+.3e}/{cr:+.3e}")
            out.append(judge("S4", f"(a) decomp {where} (at the meta energy): chi per variant vs reference", worst, 1e-6, detail=" ".join(det)))
        out.append(judge("S4", "(a) script E_g == reference gap (eV)", abs(Eg - RS.eg("MoS2")), 1e-9))
    # fallback table (v6 van Hove energies, used only without a finished prod7) vs the registered v7 values vH.E.*:
    # they must agree to half a printed digit (5 meV); a typo in the table (RT29) moves an entry by far more
    from claims_check import registry
    reg = registry()["claims"]
    badv = []
    for m in MATS:
        c = reg.get(f"vH.E.{m}")
        val = None if c is None else (c.get("value") if c.get("value") is not None else float(c["printed"]))
        if VH is None or val is None or not abs(VH.get(m, float("nan")) - val) <= 0.005:
            badv.append(f"{m}: table {VH and VH.get(m)} vs registered {val}")
    out.append(exact("S4", "fallback VH table within 5 meV of the registered van Hove energies vH.E.*", not badv, detail="; ".join(badv),
                     measured=", ".join(f"{m} {VH.get(m)}" for m in MATS) if VH else None))
    # (b) finished prod7 fixture with a dominant negative lobe
    db = isolated_copy(ctx, "decomp_b")
    om = _write_prod7(os.path.join(db, "v7jobs", "prod7_MoS2_F.npz"), done=4)
    r = run_script(ctx, "v7jobs/decomp_v7.py", ["MoS2"], root=db)
    bad = _stale_or_fail("S4", "decomp_v7 run (finished prod7 fixture)", r)
    if bad:
        out.append(bad)
    else:
        d = json.load(open(os.path.join(r["cwd"], "decomp7_MoS2.json")))
        out.append(exact("S4", "(b) finished prod7: vH == omega of max |eta(60 T)| of that file (negative lobe dominates)",
                         d["meta"]["vH_eV"] == float(om[20]), measured=f"meta {d['meta']['vH_eV']:.4f}, expected {om[20]:.4f}"))
    # (c) unfinished prod7 must be refused
    dc = isolated_copy(ctx, "decomp_c")
    om = _write_prod7(os.path.join(dc, "v7jobs", "prod7_MoS2_F.npz"), done=2)
    r = run_script(ctx, "v7jobs/decomp_v7.py", ["MoS2"], root=dc)
    fn = os.path.join(r["cwd"], "decomp7_MoS2.json")
    if r["rc"] != 0 and not os.path.exists(fn):
        out.append(Result("S4", "(c) unfinished prod7 refused", "PASS", detail="script aborted"))
    else:
        d = json.load(open(fn)) if os.path.exists(fn) else {"meta": {}}
        vh = d["meta"].get("vH_eV")
        ok = vh is not None and vh == (VH or {}).get("MoS2") and vh != float(om[0])
        out.append(exact("S4", "(c) unfinished prod7 refused (fallback used, not omega[0] of an empty 60 T row)", ok,
                         measured=f"vH {vh}", detail=f"vH_source={d['meta'].get('vH_source')}"))
    return out


# ------------------------------------------------------------------------------------------------- S5
@check("S5", "S", "lam_v7.py smoke: stored chi (code sign) x (-2) vs reference physical chi")
def s5(ctx):
    r = run_script(ctx, "v7jobs/lam_v7.py", ["MoS2", "F", "lam"])
    bad = _stale_or_fail("S5", "lam_v7 run", r)
    if bad:
        return bad
    RS = _ref()
    d = np.load(os.path.join(r["cwd"], "lam7_MoS2_F.npz"), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    OFF = np.array(meta["offsets_eV"])
    kp = ctx.eng.disks(0.3, 0.03); kK = ctx.eng.split_valleys(kp)[0]; dA = RS.disk_dA(0.03)
    worst, det = 0.0, []
    for i in range(meta["done"]):
        lam = float(d["lam"][i]); om = d["Eg"][i] + OFF
        cr = (RS.ref_eta("MoS2", "TNN", "GGA", 10.0, om, 0.005, "F", kp, c3=True, lam=lam, dA=dA)[:, 0, 0, 0]
              - RS.ref_eta("MoS2", "TNN", "GGA", -10.0, om, 0.005, "F", kp, c3=True, lam=lam, dA=dA)[:, 0, 0, 0]) / 20
        sc = RS.ref_eta("MoS2", "TNN", "GGA", 10.0, om, 0.005, "F", kK, c3=True, lam=lam, dA=dA)[:, 0, 0, 0] / 10
        worst = max(worst, rel_scaled(-2 * d["chi_SI_per_T"][i], cr, sc))
        det.append(f"lam={lam:g}: Eg={d['Eg'][i]:.4f} vs ref {RS.eg('MoS2', lam=lam):.4f}")
    eg_ok = all(abs(d["Eg"][i] - RS.eg("MoS2", lam=float(d["lam"][i]))) < 1e-9 for i in range(meta["done"]))
    return [judge("S5", f"stored chi x (-2) vs reference physical chi ({meta['done']} lambda values)", worst, 1e-6, detail="; ".join(det)),
            exact("S5", "E_g(lambda) of the script == reference gap with the same lambda", eg_ok),
            Result("S5", "sign convention of stored chi_SI_per_T", "WARN", detail="lam_v7 stores the CODE-sign chi under the key "
                   "chi_SI_per_T (no -2 factor); fig_main.py applies PHYS later (finding 9.2, P5). Checked here with x(-2).")]


# ------------------------------------------------------------------------------------------------- S6
def _lit_kp_params():
    K = json.load(open(os.path.join(SUITE_DIR, "literature", "kormanyos_2015.json")))
    liu = json.load(open(os.path.join(SUITE_DIR, "literature", "liu_tables.json")))
    it, t6 = K["interpretation"], K["table6"]["rows"]
    HB2M = (1.054571817e-34 ** 2 / (2 * 9.1093837015e-31)) / 1.602176634e-19 * 1e20   # hbar^2/2m_e [eV A^2]
    out = {}
    for i, m in enumerate(K["materials"]):
        EA = t6["Ebg,K [eV]"]["DFT"][i] + max(0.0, -K["table3"]["rows"]["2Dcb [meV] (HSE,LDA)"][i] / 1000)
        g = t6["|gamma| [eV A]"]["DFT"][i]; kap = t6[it["kappa_column"]]["DFT"][i]
        mc = K["table3"]["rows"][it["conduction_mass_row"][m]][i]; mv = K["table4"]["rows"][it["valence_mass_row"]][i]
        lam = liu["lambda_eV"][m]
        # paper Sec. II C: m_D = (E_A + lambda)/2, active mass m = E_A/2, (hbar^2/4m_e)(1/m_c+1/|m_v|) = w + v^2/2m
        out[m] = dict(v=g, kappa=kap, beta=0.5 * HB2M * (1 / mc + 1 / abs(mv)) - g * g / EA, mD=(EA + lam) / 2, lam=lam, E_A=EA)
    return out


def smoke_kp_fp(ctx):
    def build():
        r = run_script(ctx, "kp/kp_fp_all_v7.py", [])
        if r["stale"] or r["rc"] != 0:
            return dict(run=r, error="; ".join(r["stale"]) or f"exit {r['rc']}: {r['err'][-400:]}")
        return dict(run=r, json=json.load(open(os.path.join(r["cwd"], "kp_fp_all_v7.json"))))
    return ctx.cached("smoke_kp_fp", build)


def _mirror_ok(x, y, stored, name):
    """x, y plotted over -B..B; stored = the B >= 0 values. Returns list of problems."""
    x, y, stored = np.asarray(x, float), np.asarray(y, float), np.asarray(stored, float)
    n = len(stored); bad = []
    if len(x) != 2 * n - 1:
        return [f"{name}: {len(x)} plotted points for {n} stored fields"]
    if not np.array_equal(x, -x[::-1]):
        bad.append(f"{name}: field axis not mirrored")
    neg, pos = y[:n - 1], y[n:]                      # the B = 0 point is the stored eta(0) itself (not mirrored)
    if not np.allclose(neg, -pos[::-1], rtol=0, atol=1e-14 * max(np.max(np.abs(y)), 1e-300)):
        bad.append(f"{name}: plotted eta(-B) != -eta(B) (max dev {np.max(np.abs(neg + pos[::-1])):.2e})")
    yp = y[n - 1:]
    k = int(np.argmax(np.abs(stored)))
    fac = stored[k] / yp[k] if yp[k] != 0 else np.nan
    pw = np.log10(abs(fac)) if np.isfinite(fac) and fac > 0 else np.nan
    if not (np.isfinite(pw) and abs(pw - round(pw)) < 1e-9):
        bad.append(f"{name}: plotted/stored scale {fac} is not a positive power of ten")
    elif not np.allclose(yp * fac, stored, rtol=1e-12, atol=1e-14 * np.max(np.abs(stored))):
        bad.append(f"{name}: plotted B >= 0 branch differs from the stored data (max rel {np.max(np.abs(yp*fac-stored))/np.max(np.abs(stored)):.1e})")
    return bad


def _long_lines(ax, nmin=5):
    return [l for l in ax["lines"] if len(l["x"]) >= nmin]


def figure_readback(ctx, script_rel, argv, outdir):
    """Run a figure script on the smoke outputs under figrecord.py; return (recording or None, run)."""
    root, _ = smoke_tree(ctx)
    rec = os.path.join(ctx.scratch, os.path.basename(script_rel) + ".rec.json")
    r = run_script(ctx, script_rel, argv, wrapper=[os.path.join(SUITE_DIR, "tests", "figrecord.py"), rec])
    return (json.load(open(rec)) if os.path.exists(rec) else None), r


def check_figs(cwd, R, gen):
    """Odd mirroring + equality with stored data for the plotted arrays of Figs. 7, 8, 9 (and 11 markers)."""
    bad, n = [], 0
    P = {m: np.load(os.path.join(cwd, f"prod7_{m}_F.npz"), allow_pickle=True) for m in MATS}
    E = {(m, mo): json.load(open(os.path.join(cwd, f"edge7_{m}_{mo}.json"))) for m in MATS for mo in ("TNN", "NN")}
    if "lpge_field_merged" in R:                                   # Fig. 7
        ls = _long_lines(R["lpge_field_merged"][0])
        for m, l in zip(MATS, ls):
            Y = P[m]["eta"]; jv = int(np.argmax(np.abs(Y[-1])))
            bad += _mirror_ok(l["x"], l["y"], Y[:, jv], f"{gen} Fig.7 {m}"); n += 1
        if len(ls) != 4:
            bad.append(f"{gen} Fig.7: {len(ls)} curves")
    if "lpge_map_grid" in R:                                       # Fig. 8
        for i, m in enumerate(MATS):
            ms = R["lpge_map_grid"][i]["meshes"]
            if not ms:
                bad.append(f"{gen} Fig.8 {m}: no mesh"); continue
            A = np.array(ms[0]["array"], float)
            Y = np.asarray(P[m]["eta"], float)
            if A.ndim == 1:
                A = A.reshape(2 * len(Y) - 1, -1)
            nb = len(Y)
            if A.shape != (2 * nb - 1, Y.shape[1]):
                bad.append(f"{gen} Fig.8 {m}: mesh shape {A.shape} vs stored {Y.shape}"); continue
            if not np.allclose(A[:nb - 1], -A[nb:][::-1], rtol=0, atol=1e-14 * np.abs(A).max()):
                bad.append(f"{gen} Fig.8 {m}: map not odd in B")
            if not np.allclose(A[nb - 1:], Y, rtol=1e-12, atol=1e-14 * np.abs(Y).max()):
                bad.append(f"{gen} Fig.8 {m}: B >= 0 rows differ from the stored map")
            n += 1
    if "lpge_edge_sign_grid" in R:                                 # Fig. 9
        for i, m in enumerate(MATS):
            ls = _long_lines(R["lpge_edge_sign_grid"][i], 3)
            d = E[(m, "TNN")]["eta"]; Bs = sorted(map(float, d)); Ym = np.array([np.mean(d[f"{b:g}"]) for b in Bs])
            if not ls:
                bad.append(f"{gen} Fig.9 {m}: no curve"); continue
            bad += _mirror_ok(ls[0]["x"], ls[0]["y"], Ym, f"{gen} Fig.9 {m}"); n += 1
    if "threshold_models" in R:                                    # Fig. 11 markers (panel a)
        ax0 = R["threshold_models"][0]["lines"]
        for mk, mo in (("o", "TNN"), ("^", "NN")):
            l = [x for x in ax0 if x["marker"] == mk]
            if not l:
                bad.append(f"{gen} Fig.11 no '{mk}' markers"); continue
            y = np.array(l[0]["y"]); st = np.array([E[(m, mo)]["chi_SI_per_T"] * 1e4 for m in MATS])
            if not np.allclose(y, st, rtol=1e-12, atol=0):
                bad.append(f"{gen} Fig.11 {mo} points != stored chi x 1e4")
            n += 1
    return bad, n


@check("S6", "S", "kp_fp_all_v7.py smoke; figure scripts on smoke outputs: plotted arrays read back (mirroring, equality)")
def s6(ctx):
    sm = smoke_kp_fp(ctx)
    if sm.get("error"):
        return Result("S6", "kp_fp_all_v7 run", "STALE" if sm["run"]["stale"] else "FAIL", detail=sm["error"])
    RS = _ref()
    d = sm["json"]
    lit = _lit_kp_params()
    SI = 1e-10 * 1.602176634e-19 ** 3 / 1.054571817e-34 ** 2
    R, Nk = 0.2, 101
    ks = np.linspace(-R, R, Nk); KX, KY = np.meshgrid(ks, ks, indexing="ij"); msk = KX ** 2 + KY ** 2 <= R * R
    dA = (2 * R / (Nk - 1)) ** 2                      # reference's own spacing
    out = []
    worst, det = 0.0, []
    for m, p in lit.items():
        om = 2 * p["mD"] - p["lam"] + np.array(d[m]["offsets_eV"])
        c = [RS.two_band_eta(KX[msk], KY[msk], dA, om, 0.005, p, s * MUB * 30)[:, 0, 0, 0] for s in (1, -1)]
        chi = -2.0 * (c[0] - c[1]) / 60 * SI
        one = RS.two_band_eta(KX[msk], KY[msk], dA, om, 0.005, p, MUB * 30, pairs=((1, 1),))[:, 0, 0, 0]
        dd = rel_scaled(np.array(d[m]["chi_phys_SI_per_T"]), chi, 2 * one / 60 * SI)
        worst = max(worst, dd); det.append(f"{m}: script {d[m]['chi_phys_SI_per_T'][0]:+.4e} ref {chi[0]:+.4e}")
    out.append(judge("S6", "kp threshold chi (6 offsets) vs reference two-band sum, inputs rebuilt from literature/", worst, 1e-4,
                     detail="; ".join(det)))
    er, mr = edge_runs(ctx), maps_runs(ctx)
    if any(x["rc"] != 0 for x in list(er.values()) + list(mr.values())):
        out.append(Result("S6", "figure inputs", "FAIL", detail="edge/maps smoke runs failed"))
        return out
    # old Figs. 8-9 script
    R7, f = figure_readback(ctx, "v7jobs/fig_v7.py", [], None)
    if f["rc"] != 0 or R7 is None:
        out.append(Result("S6", "fig_v7.py on smoke outputs", "FAIL", detail=f["err"][-600:]))
    else:
        thr = json.load(open(os.path.join(f["cwd"], "threshold_chi_TNN_v7.json")))
        eq = all(thr[m] == json.load(open(os.path.join(f["cwd"], f"edge7_{m}_TNN.json")))["chi_SI_per_T"] for m in thr)
        bad, n = check_figs(f["cwd"], R7, "fig_v7")
        out.append(exact("S6", "fig_v7.py: threshold json == edge chi; plotted Figs. 8, 9 odd and equal to stored data", eq and not bad and n == 8,
                         measured=f"{n} panels checked", detail="; ".join(bad)))
    # production figure script Figs. 7-12
    if not os.path.exists(os.path.join(smoke_tree(ctx)[0], "v7jobs", "fig_v7_all.py")):
        out.append(Result("S6", "fig_v7_all.py", "SKIP", detail="fig_v7_all.py not found (config script_paths)"))
        return out
    RA, f = figure_readback(ctx, "v7jobs/fig_v7_all.py", ["--data", ".", "--kp", "../kp/kp_fp_all_v7.json", "--out", "figs_v7",
                                                          "--only", "7", "8", "9", "11", "--no-png"], None)
    if f["rc"] != 0 or RA is None:
        out.append(Result("S6", "fig_v7_all.py on smoke outputs", "FAIL", detail=f["err"][-800:]))
        return out
    bad, n = check_figs(f["cwd"], RA, "fig_v7_all")
    side = [x for x in ("lpge_field_merged", "lpge_map_grid", "lpge_edge_sign_grid", "threshold_models")
            if not os.path.exists(os.path.join(f["cwd"], "figs_v7", x + ".prov.json"))]
    out.append(exact("S6", "fig_v7_all.py: plotted Figs. 7, 8, 9, 11 odd and equal to stored data; sidecars written",
                     not bad and n == 4 + 4 + 4 + 2 and not side, measured=f"{n} panels checked",
                     detail="; ".join(bad + [f"no sidecar for {x}" for x in side])
                     + " | Figs. 10, 12 are not mirrored and are not read back (their inputs need all 8 chiw / 4 lam runs)"))
    return out
