"""T: static checks on the source code (no physics run).

T1 knob inventory: every numeric literal and default in code reachable from lpge.eta and kp2.response must be
   registered in knobs.json, in order, per function. An unregistered or changed literal fails.
T2 literature literals: literature tables may live only in tmd_params.py / literature/. Known mirrors
   (kp_fp_all_v7.KORM, kp2.KORM, kp2.LAMBDA, tb_core.G_VALLEY_MEASURED) are registered and value-checked.
T3 the reference imports nothing from the engine.
T4 the import closure of every producing script lies inside the engine file list.
"""
import os, sys, ast, json
import numpy as np
from vcore import check, Result, exact, SUITE_DIR

# Code reachable from lpge.eta and kp2.response (protocol decision 9). The GPU/geometry path of tb_core
# (NumpyBackend, TorchBackend, geometry, light_geometry, assemble_SO, H_external, H_exchange) is not called by
# lpge.eta: it is recorded as dead code for this pipeline and kept out of the inventory.
SCOPE = {
    "pub_pipeline/lpge.py": ["<module>", "grid", "_H6", "_block_quantities", "_moment_omega", "chunk_transitions",
                             "local_kpts", "eta"],
    "pub_pipeline/tb_core.py": ["<module>", "_load_cfg", "set_config", "material_constants", "build_grid",
                                "H_NN_np", "H_TNN_np", "make_H3", "C3_sym3"],
    "kp/kp2.py": ["<module>", "params", "_dv", "_P", "response", "C3"],
}
DEAD = {"pub_pipeline/tb_core.py": ["NumpyBackend", "TorchBackend", "make_backend", "geometry", "light_geometry",
                                    "assemble_SO", "H_external", "H_exchange", "_diag", "set_material"]}
# module-level names that are data tables (checked by L1/T2, not by T1)
TABLE_NAMES = {"KORM", "LAMBDA", "G_VALLEY_MEASURED", "LATTICE_A", "LAM_ALL"}


def _num(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _num(node.operand)
        if v is not None:
            return -v
    return None


def _literals(node):
    """Ordered numeric literals under node (a negated literal counts once, with its sign)."""
    out = []

    def visit(n):
        v = _num(n)
        if v is not None:
            out.append(v)
            return
        for c in ast.iter_child_nodes(n):
            visit(c)
    visit(node)
    return out


def _enc(v):
    if isinstance(v, complex):
        return f"({v.real!r}{'+' if v.imag >= 0 else ''}{v.imag!r}j)"
    return repr(v)


def inventory(code_root):
    inv = {}
    for rel_, scopes in SCOPE.items():
        tree = ast.parse(open(os.path.join(code_root, rel_)).read())
        funcs = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                funcs.setdefault(n.name, n)
        for s in scopes:
            if s == "<module>":
                lits = []
                for st in tree.body:
                    if isinstance(st, (ast.Assign, ast.AnnAssign)):
                        tg = st.targets[0] if isinstance(st, ast.Assign) else st.target
                        names = [t.id for t in ast.walk(tg) if isinstance(t, ast.Name)]
                        if any(nm in TABLE_NAMES for nm in names):
                            continue
                        lits += [f"{'/'.join(names)}={_enc(v)}" for v in _literals(st.value)]
                    elif isinstance(st, ast.Try):        # kp2: NU = 0.01; WIN = 1e9 inside try/except
                        for sub in ast.walk(st):
                            if isinstance(sub, ast.Assign) and not isinstance(sub.value, ast.Lambda):
                                names = [t.id for t in ast.walk(sub.targets[0]) if isinstance(t, ast.Name)]
                                lits += [f"{'/'.join(names)}={_enc(v)}" for v in _literals(sub.value)]
                inv[f"{rel_}::<module>"] = lits
            else:
                if s not in funcs:
                    inv[f"{rel_}::{s}"] = ["<missing function>"]
                    continue
                inv[f"{rel_}::{s}"] = [_enc(v) for v in _literals(funcs[s])]
    return inv


def register(code_root, path):
    """Write knobs.json from the current code (human review required: keep the 'roles')."""
    old = {}
    if os.path.exists(path):
        old = json.load(open(path)).get("roles", {})
    data = dict(_comment="Knob inventory (T1). Ordered numeric literals per function reachable from lpge.eta and "
                         "kp2.response. Regenerate with verify_all.py --register-knobs only after a reviewed code change.",
                dead_code=DEAD, inventory=inventory(code_root), roles=old)
    json.dump(data, open(path, "w"), indent=1)
    return data


@check("T1", "T", "knob inventory: every literal reachable from eta/kp2.response is registered")
def t1(ctx):
    reg = json.load(open(os.path.join(SUITE_DIR, "knobs.json")))
    now = inventory(ctx.code_root)
    diffs = []
    for k in sorted(set(now) | set(reg["inventory"])):
        a, b = reg["inventory"].get(k), now.get(k)
        if a != b:
            if a is None or b is None:
                diffs.append(f"{k}: {'new scope' if a is None else 'scope removed'}")
            else:
                import difflib
                sm = difflib.SequenceMatcher(a=a, b=b)
                for op, i1, i2, j1, j2 in sm.get_opcodes():
                    if op != "equal":
                        diffs.append(f"{k}: {op} {a[i1:i2]} -> {b[j1:j2]}")
    n = sum(len(v) for v in now.values())
    return exact("T1", "knob inventory", not diffs, measured=f"{len(diffs)} unregistered/changed of {n}",
                 detail="; ".join(diffs[:12]))


# ------------------------------------------------------------------------------------------------- T2
def _module_tables(path):
    tree = ast.parse(open(path).read())
    out = {}
    for st in tree.body:
        if isinstance(st, ast.Assign) and isinstance(st.value, ast.Dict):
            nm = st.targets[0].id if isinstance(st.targets[0], ast.Name) else None
            try:
                out[nm] = ast.literal_eval(st.value)
            except Exception:
                pass
        if isinstance(st, ast.Assign) and isinstance(st.value, ast.Dict) is False:
            continue
    return out


NOT_LITERATURE = {"v7jobs/decomp_v7.py::VH": "van Hove photon energies taken from the v6 maps; must be re-read from prod7 after maps_v7 (claims vH.E.*)"}


def _is_lit_table(d):
    vals = list(d.values())
    return len(vals) >= 3 and all(isinstance(x, (tuple, list, float, int)) and not isinstance(x, bool) for x in vals)


@check("T2", "T", "literature values appear as literals only in tmd_params.py / literature/")
def t2(ctx):
    lit = os.path.join(SUITE_DIR, "literature")
    K = json.load(open(os.path.join(lit, "kormanyos_2015.json")))
    liu = json.load(open(os.path.join(lit, "liu_tables.json")))
    mats = K["materials"]; it = K["interpretation"]
    t6 = K["table6"]["rows"]
    res, warns, fails = [], [], []
    files = []
    for sub in ("pub_pipeline", "kp", "v7jobs"):
        d = os.path.join(ctx.code_root, sub)
        files += [os.path.join(sub, f) for f in sorted(os.listdir(d)) if f.endswith(".py")]
    scope = set(ctx.cfg["engine_files"]) | set(ctx.cfg["scripts"])
    for rp in files:
        if rp not in scope:
            continue                       # only engine + producing scripts are production code
        tabs = _module_tables(os.path.join(ctx.code_root, rp))
        for nm, d in tabs.items():
            if not (isinstance(d, dict) and _is_lit_table(d)):
                continue
            key = f"{rp}::{nm}"
            if rp == "pub_pipeline/tmd_params.py":
                continue                   # the designated home (checked by L1)
            if key in NOT_LITERATURE:
                res.append(f"{key}: not literature ({NOT_LITERATURE[key]})")
                continue
            if key == "kp/kp_fp_all_v7.py::KORM":
                bad = []
                for i, m in enumerate(mats):
                    EA, g, kap, mc, mv, lam = d[m]
                    exp = (t6["Ebg,K [eV]"]["DFT"][i] + max(0.0, -K["table3"]["rows"]["2Dcb [meV] (HSE,LDA)"][i] / 1000),
                           t6["|gamma| [eV A]"]["DFT"][i], t6[it["kappa_column"]]["DFT"][i],
                           K["table3"]["rows"][it["conduction_mass_row"][m]][i], K["table4"]["rows"][it["valence_mass_row"]][i],
                           liu["lambda_eV"][m])
                    if not np.allclose((EA, g, kap, mc, mv, lam), exp, atol=5e-4, rtol=0):
                        bad.append(f"{m}: {d[m]} vs literature {tuple(round(x, 3) for x in exp)}")
                (fails if bad else res).append(f"{key} mirror of literature/kormanyos_2015.json: " + ("; ".join(bad) if bad else "equal"))
            elif key == "kp/kp2.py::KORM":
                kup = t6["kappa_up [eV A^2]"]["DFT"]
                same_up = all(abs(d[m][2] - kup[i]) < 1e-9 for i, m in enumerate(mats))
                warns.append(f"{key}: stale V-01 inputs (kappa = kappa_up column {'exactly' if same_up else 'partly'}, B-pair masses); "
                             "not used by the v7 scripts (kp_fp_all_v7 passes p=...), ticket: delete or point to literature/")
            elif key == "kp/kp2.py::LAMBDA":
                ok = all(abs(d[m] - liu["lambda_eV"][m]) < 1e-12 for m in d)
                (warns if ok else fails).append(f"{key}: duplicate of Liu lambda ({'equal' if ok else 'DIFFERENT'})")
            elif key == "pub_pipeline/tb_core.py::G_VALLEY_MEASURED":
                warns.append(f"{key}: measured exciton g, validation reference only (unused by eta)")
            else:
                fails.append(f"{key}: unregistered literature-like table outside tmd_params/literature")
    # fig_v7_all.py hard-codes Table II kappa (Fig. 11 x axis and the fit), LDA kappa and Liu lambda
    import prov
    fa = prov.resolve_script(ctx.cfg, ctx.code_root, "v7jobs/fig_v7_all.py")
    if os.path.exists(fa):
        tabs = {}
        for st_ in ast.parse(open(fa).read()).body:
            if isinstance(st_, ast.Assign) and isinstance(st_.targets[0], ast.Name) and st_.targets[0].id in ("KAPPA_TB", "KAPPA_LDA", "LAM_LIU"):
                try:
                    tabs[st_.targets[0].id] = ast.literal_eval(st_.value)
                except Exception:
                    tabs[st_.targets[0].id] = ("dict(zip", st_.value)
        ref = ctx.ref
        bad = []
        kt = tabs.get("KAPPA_TB", {})
        for model in ("TNN", "NN"):
            for m in mats:
                v = kt.get(model, {}).get(m) if isinstance(kt, dict) else None
                r = ref.kappa_lowdin(m, model, "GGA")["kappa"]
                if v is None or abs(v - r) > 5e-4:
                    bad.append(f"KAPPA_TB[{model}][{m}]={v} vs Lowdin {r:+.4f}")
        lam = tabs.get("LAM_LIU", {})
        srcf = open(fa).read()
        if not isinstance(lam, dict) and "import tmd_params" in srcf and "_tp.LAMBDA[m]" in srcf:
            lam = {m: float(ctx.eng.tp.LAMBDA[m]) for m in mats}          # read from the engine parameter file (L1 checks it)
            res.append("fig_v7_all.py LAM_LIU is read from pub_pipeline/tmd_params.py LAMBDA (single source)")
        for m in mats:
            if not isinstance(lam, dict) or abs(lam.get(m, -1) - liu["lambda_eV"][m]) > 1e-12:
                bad.append(f"LAM_LIU[{m}] vs Liu {liu['lambda_eV'][m]}")
        # KAPPA_LDA is built with dict(zip(...)): evaluate it in a sandbox namespace
        src = open(fa).read()
        i0 = src.find("KAPPA_LDA ="); line = src[i0:src.find("\n", i0)]
        try:
            ns = {"MATS": mats}; exec(line, {}, ns); kl = ns["KAPPA_LDA"]
            for model in ("TNN", "NN"):
                for m in mats:
                    r = ref.kappa_lowdin(m, model, "LDA")["kappa"]
                    if abs(kl[model][m] - r) > 5e-4:
                        bad.append(f"KAPPA_LDA[{model}][{m}]={kl[model][m]} vs Lowdin {r:+.4f}")
        except Exception as ex:
            bad.append(f"KAPPA_LDA not parsed: {ex}")
        (fails if bad else res).append("fig_v7_all.py KAPPA_TB/KAPPA_LDA/LAM_LIU vs reference Lowdin and Liu: " + ("; ".join(bad) if bad else "equal"))
    st = "FAIL" if fails else ("WARN" if warns else "PASS")
    return Result("T2", "literature literals", st, tol="0 FAIL", measured=f"{len(fails)} fail, {len(warns)} warn",
                  detail="; ".join(fails + warns + res))


@check("T3", "T", "reference imports nothing from the engine")
def t3(ctx):
    tree = ast.parse(open(os.path.join(SUITE_DIR, "ref", "refimpl.py")).read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    bad = mods - {"numpy", "math", "os", "sys", "functools", "itertools"}
    return exact("T3", "ref import isolation", not bad, measured=sorted(mods), detail=f"non-stdlib/numpy imports: {sorted(bad)}")


STDLIB = {"sys", "os", "json", "time", "hashlib", "datetime", "math", "numpy", "matplotlib", "argparse", "glob", "platform",
          "itertools", "functools", "re", "subprocess", "pathlib", "collections", "copy", "pickle", "warnings", "scipy"}
STDLIB |= set(getattr(sys, "stdlib_module_names", ()))
LOCAL2FILE = {"lpge": "pub_pipeline/lpge.py", "tb_core": "pub_pipeline/tb_core.py", "tmd_params": "pub_pipeline/tmd_params.py",
              "kp2": "kp/kp2.py"}


@check("T4", "T", "import closure of every producing script lies inside the engine file list")
def t4(ctx):
    allowed = set(ctx.cfg["engine_files"])
    bad, seen = [], {}
    todo = [s for s in ctx.cfg["scripts"] if s.endswith(".py")]
    done = set()
    while todo:
        rp = todo.pop()
        if rp in done:
            continue
        done.add(rp)
        import prov
        p = prov.resolve_script(ctx.cfg, ctx.code_root, rp)
        if not os.path.exists(p):
            bad.append(f"{rp}: missing"); continue
        tree = ast.parse(open(p).read())
        dead = set(DEAD.get(rp, []))
        nodes = []

        def walk(n):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in dead:
                return                     # dead GPU path (torch), not reachable from lpge.eta
            nodes.append(n)
            for c in ast.iter_child_nodes(n):
                walk(c)
        walk(tree)
        for n in nodes:
            names = []
            if isinstance(n, ast.Import):
                names = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                names = [(n.module or "").split(".")[0]]
            for nm in names:
                if nm in STDLIB:
                    continue
                f = LOCAL2FILE.get(nm)
                if f is None:
                    bad.append(f"{rp} imports '{nm}' (not in the tree file list)")
                else:
                    if f not in allowed:
                        bad.append(f"{rp} -> {f} not in engine_files")
                    todo.append(f)
            # sys.path manipulation pointing outside the tree
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("/home/"):
                bad.append(f"{rp}: absolute path {n.value}")
    return exact("T4", "script import closure", not bad, measured=f"{len(done)} files", detail="; ".join(bad))
