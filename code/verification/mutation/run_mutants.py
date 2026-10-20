#!/usr/bin/env python3
"""Mutation harness (protocol sec. 7).

  python3 mutation/run_mutants.py [--config config_cloud.json] [--jobs 2] [--only M01,M05] [--skip-x]

1. Copies pub_pipeline/, kp/, v7jobs/ (code and small json inputs) of the code root to a temp folder (the control).
2. Runs the mutation subset of the fast suite on the control; every test must PASS there (tests that do not are
   reported and cannot count as killers).
3. For each mutant of mutants.json: asserts that 'old' occurs exactly once (else STALE -> harness FAIL), applies it to a
   fresh copy, and runs
      phase A: its expected killers + the cheap static/Hamiltonian checks (T, U, L, H4-H6, H8-H10), no -x;
      phase B (only if phase A killed nothing): the whole mutation subset with -x.
   Equivalent mutants (M05, M28) and M10 run the whole subset without -x; equivalent ones must leave the fixture
   outputs unchanged at rel 1e-13 (else they must be killed); M10 must be killed by H5/R1 (Hamiltonian level) and by
   no eta-level test.
4. Non-code mutants X1-X4 mutate a copy of the paper, the registry or the data and run the G/P checks.
Writes mutation/report.md, mutation/report.json and mutation/kill_matrix_<date>.json.
"""
import os, sys, json, time, shutil, tempfile, argparse, subprocess, datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SUITE, "lib"))
from vcore import load_config

SUBSET = "T,U,L,H,R,I,S,C0,C5,G,P"
CHEAP = "T1,T2,T3,T4,U1,U2,U3,L1,L2,L3,L4,H4,H5,H6,H8,H9,H10"
ETA_LEVEL = {"R4", "R5", "R6", "H1", "H2", "H3", "H7", "I1", "I2", "I3", "I4", "I5", "S1", "S2", "S3", "S4", "S5", "S6", "C0", "C5", "R8"}
STATIC = {"T1", "T2", "T3", "T4"}
H_LEVEL = {"H4", "H5", "R1"}
FULL_RUN = {"M05", "M28", "M10"}


def copy_tree(src, dst):
    for sub in ("pub_pipeline", "kp", "v7jobs"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
        for f in os.listdir(os.path.join(src, sub)):
            p = os.path.join(src, sub, f)
            if os.path.isfile(p) and f.endswith((".py", ".sh", ".json", ".md")) and os.path.getsize(p) < 5e6:
                shutil.copy2(p, os.path.join(dst, sub, f))


def run_suite(cfg_path, code_root, only, order=None, x=False, fixtures=None, env_extra=None, timeout=900):
    out = tempfile.mktemp(suffix=".json")
    cmd = [sys.executable, os.path.join(SUITE, "verify_all.py"), "--fast", "--config", cfg_path, "--code-root", code_root,
           "--only", only, "--json-out", out, "--no-report", "--quiet"]
    if order:
        cmd += ["--order", order]
    if x:
        cmd += ["-x"]
    if fixtures:
        cmd += ["--fixtures-out", fixtures]
    env = dict(os.environ, OMP_NUM_THREADS="1", **(env_extra or {}))
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    try:
        res = json.load(open(out))["results"]
    except Exception:
        res = [dict(id="RUN", name="suite crashed", status="FAIL", detail=p.stderr[-800:])]
    return res, time.time() - t


def key(r):
    return f"{r['id']}|{r['name']}"


def killers(control, res):
    ok = {key(r) for r in control if r["status"] in ("PASS", "KNOWN", "WARN", "INFO", "SKIP")}
    k = []
    for r in res:
        if r["status"] in ("FAIL", "STALE") and (key(r) in ok or r["id"] == "RUN" or key(r) not in {key(c) for c in control}):
            k.append(r)
    return k


def fixture_diff(a_path, b_path):
    A, B = np.load(a_path), np.load(b_path)
    worst = 0.0
    for k in A.files:
        if k not in B.files:
            return float("inf")
        a, b = A[k], B[k]
        parts = k.split("|")              # eng|mat|model|func|var|B|kset|lam
        sc = np.max(np.abs(a))
        if parts[6] in ("S2", "S2C3"):         # K+K' sums: protocol sec. 3 rule, scale = single-valley value
            kK = "|".join(parts[:6] + ["S2K"] + parts[7:])
            if kK in A.files:
                sc = np.max(np.abs(A[kK]))
        if float(parts[5]) == 0.0:             # B = 0 arrays: compare against the B != 0 scale of the same config
            sc = max([np.max(np.abs(A[k2])) for k2 in A.files if k2.split("|")[1:5] == parts[1:5] and float(k2.split("|")[5]) != 0] or [sc])
        worst = max(worst, float(np.max(np.abs(a - b)) / max(sc, 1e-300)))
    return worst


def do_mutant(m, cfg_path, control_root, control, control_fix, work):
    t0 = time.time()
    d = os.path.join(work, m["id"])
    shutil.copytree(control_root, d)
    p = os.path.join(d, m["file"])
    s = open(p).read()
    for pt in [dict(old=m["old"], new=m["new"])] + m.get("extra", []):
        n = s.count(pt["old"])
        if n != 1:
            return dict(id=m["id"], status="STALE", detail=f"'old' occurs {n} times: {pt['old'][:50]}", killers=[], seconds=0)
        s = s.replace(pt["old"], pt["new"])
    os.chmod(p, 0o644)
    open(p, "w").write(s)
    exp = ",".join(m.get("expected_killers", []))
    if m["id"] in FULL_RUN:
        fx = os.path.join(work, m["id"] + "_fix.npz")
        res, dt = run_suite(cfg_path, d, SUBSET, order=exp or None, fixtures=fx)
        k = killers(control, res)
        ids = sorted({r["id"] for r in k})
        if m.get("equivalent"):
            fd = fixture_diff(control_fix, fx)
            eta = set(ids) & ETA_LEVEL
            st = "EQUIVALENT-OK" if fd <= 1e-13 and not eta else "EQUIVALENT-DIFF"
            det = (f"fixture outputs rel diff {fd:.1e} (<= 1e-13 required); eta-level failures: {sorted(eta) or 'none'}"
                   + (f"; flagged by static checks {sorted(set(ids) - eta)} (a literal edit always needs re-registration)" if set(ids) - eta else ""))
        else:
            eta = set(ids) & ETA_LEVEL
            st = "KILLED" if ids else "SURVIVED"
            det = (f"Hamiltonian-level killers {sorted(set(ids) & H_LEVEL)}; eta-level killers {sorted(eta) or 'none'}"
                   + (" -> OK (equivalent for eta, needs the H-level tests)" if not eta and set(ids) & H_LEVEL else " -> REQUIREMENT NOT MET"))
        return dict(id=m["id"], status=st, killers=ids, detail=det, seconds=round(time.time() - t0, 1),
                    fails=[f"{r['id']}: {r['name']} ({r['detail'][:120]})" for r in k][:6])
    only = ",".join(sorted(set(CHEAP.split(",")) | set(m.get("expected_killers", []))))
    res, dt = run_suite(cfg_path, d, only, order=exp)
    k = killers(control, res)
    phase = "A"
    if not k or {r["id"] for r in k} <= STATIC:
        # no physics/claims killer yet: run everything except the static tripwires, stop at the first kill
        res2, dt2 = run_suite(cfg_path, d, SUBSET.replace("T,", ""), order=exp, x=True)
        k = k + killers(control, res2); phase = "B"
    ids = sorted({r["id"] for r in k})
    st = "SURVIVED" if not ids else ("KILLED (static only)" if set(ids) <= STATIC else "KILLED")
    return dict(id=m["id"], status=st, killers=ids, phase=phase, seconds=round(time.time() - t0, 1),
                fails=[f"{r['id']}: {r['name']} ({r['detail'][:120]})" for r in k][:6])


# ------------------------------------------------------------------------------------------------ non-code mutants
def x_mutants(cfg_path, cfg, work):
    out = []
    paper = cfg["paths"]["paper_root"] if os.path.isabs(cfg["paths"]["paper_root"]) else os.path.join(SUITE, cfg["paths"]["paper_root"])

    def override(**paths):
        c = dict(extends=os.path.abspath(cfg_path), paths=paths)
        pth = tempfile.mktemp(suffix=".json", dir=work)
        json.dump(c, open(pth, "w"))
        return pth
    code = cfg["paths"]["code_root"] if os.path.isabs(cfg["paths"]["code_root"]) else os.path.join(SUITE, cfg["paths"]["code_root"])
    base_g, _ = run_suite(cfg_path, code, "G1,G2,G3,P2")
    # X1: change one printed digit of an active claim in a copy of the tex (+1.9 is pending, so Table II v(MoS2))
    pc = os.path.join(work, "X1_paper"); shutil.copytree(paper, pc)
    t = os.path.join(pc, "sections", "theory.tex"); s = open(t).read()
    s2 = s.replace("\\claim{tab2.v.TNN.MoS2}{4.93}", "\\claim{tab2.v.TNN.MoS2}{4.94}", 1)      # tagged paper
    if s2 == s:
        s2 = s.replace("$\\mathrm{MoS_2}$  & $4.93$", "$\\mathrm{MoS_2}$  & $4.94$", 1)
    assert s2 != s, "X1: Table II v(MoS2) not found in the paper copy"
    open(t, "w").write(s2)
    res, _ = run_suite(override(paper_root=pc), code, "G2,G3")
    hit = any(r["id"] == "G2" and "tab2.v.TNN.MoS2" in r["detail"] for r in res)
    out.append(dict(id="X1", status="KILLED" if hit else "SURVIVED", killers=["G2"] if hit else [],
                    detail="Table II v(MoS2) 4.93 -> 4.94 in a copy of theory.tex (the protocol's +1.9 -> +1.8 is a pending claim until the rerun)"))
    # X2: new untagged decimal
    pc = os.path.join(work, "X2_paper"); shutil.copytree(paper, pc)
    t = os.path.join(pc, "sections", "results.tex"); s = open(t).read()
    s2 = s
    for anchor in ("The sign of the current therefore", "The sign of the measured current"):
        s2 = s.replace(anchor, "A new estimate is $0.123$~eV. " + anchor, 1)
        if s2 != s:
            break
    assert s2 != s, "X2: insertion anchor not found in results.tex"
    open(t, "w").write(s2)
    res, _ = run_suite(override(paper_root=pc), code, "G3")
    hit = any(r["id"] == "G3" and r["status"] == "FAIL" and "0.123" in r["detail"] for r in res)
    out.append(dict(id="X2", status="KILLED" if hit else "SURVIVED", killers=["G3"] if hit else [], detail="untagged '0.123' inserted in results.tex"))
    # X3: data rerun, figure not replotted (figure older than its data)
    pc = os.path.join(work, "X3_paper"); shutil.copytree(paper, pc)
    os.makedirs(os.path.join(pc, "figures", "results"), exist_ok=True)
    fig = os.path.join(pc, "figures", "results", "lpge_map_grid.pdf"); open(fig, "wb").write(b"%PDF-1.4 dummy")
    os.utime(fig, (time.time() - 3600, time.time() - 3600))
    v7 = os.path.join(work, "X3_v7"); os.makedirs(v7)
    for m_ in ("MoS2", "MoSe2", "WS2", "WSe2"):
        np.savez(os.path.join(v7, f"prod7_{m_}_F.npz"), eta=np.zeros(1), meta=json.dumps(dict(lpge_sha256="ff4a692dbbd465b045bd82519ed7602d918793ddbfe9484d1912c59411a7b2f8")))
    res, _ = run_suite(override(paper_root=pc, v7_out=v7), code, "P2")
    hit = any(r["id"] == "P2" and r["status"] == "FAIL" and "data newer than figure" in r["detail"] for r in res)
    out.append(dict(id="X3", status="KILLED" if hit else "SURVIVED", killers=["P2"] if hit else [], detail="prod7 data newer than lpge_map_grid.pdf"))
    # X4: registry value edited without history
    reg = json.load(open(os.path.join(SUITE, "claims.json")))
    reg["claims"]["tab5.lpge.B30"]["value"] *= 1.01
    rp = os.path.join(work, "X4_claims.json"); json.dump(reg, open(rp, "w"))
    res, _ = run_suite(cfg_path, code, "G1", env_extra=dict(VERIF_CLAIMS=rp))
    hit = any(r["id"] == "G1" and r["status"] == "FAIL" and "tab5.lpge.B30" in r["detail"] for r in res)
    out.append(dict(id="X4", status="KILLED" if hit else "SURVIVED", killers=["G1"] if hit else [], detail="tab5.lpge.B30 value x1.01 in a copy of claims.json"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(SUITE, "config.json")); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only"); ap.add_argument("--skip-x", action="store_true")
    ap.add_argument("--merge", action="store_true", help="merge these mutants into the existing report.json")
    a = ap.parse_args()
    cfg_path = a.config if os.path.isabs(a.config) or os.path.exists(a.config) else os.path.join(SUITE, a.config)
    cfg_path = os.path.abspath(cfg_path)
    cfg = load_config(cfg_path)
    code = cfg["paths"]["code_root"] if os.path.isabs(cfg["paths"]["code_root"]) else os.path.join(SUITE, cfg["paths"]["code_root"])
    spec = json.load(open(os.path.join(HERE, "mutants.json")))
    muts = [m for m in spec["mutants"] if not a.only or m["id"] in a.only.split(",")]
    work = tempfile.mkdtemp(prefix="mut_")
    control_root = os.path.join(work, "control"); copy_tree(code, control_root)
    t0 = time.time()
    print("control run ...", flush=True)
    control_fix = os.path.join(work, "control_fix.npz")
    control, dt = run_suite(cfg_path, control_root, SUBSET, fixtures=control_fix)
    bad_ctrl = [f"{r['id']}: {r['name']}" for r in control if r["status"] == "FAIL"]
    stale_ctrl = [f"{r['id']}: {r['name']}" for r in control if r["status"] == "STALE"]
    print(f"control: {len(control)} checks, {len(bad_ctrl)} not passing, {dt:.0f} s", flush=True)
    for b_ in bad_ctrl:
        print("  control FAIL:", b_, flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(do_mutant, m, cfg_path, control_root, control, control_fix, work): m for m in muts}
        for f in futs:
            m = futs[f]
            r = f.result()
            r.update(file=m["file"], cls=m["class"], expected=m.get("expected_killers", []), equivalent=m.get("equivalent", False))
            results.append(r)
            print(f"{r['id']} {r['status']:16s} killers={r['killers']} ({r['seconds']} s)", flush=True)
    xres = [] if a.skip_x else x_mutants(cfg_path, cfg, work)
    if a.merge and os.path.exists(os.path.join(HERE, "report.json")):
        old = json.load(open(os.path.join(HERE, "report.json")))
        mine = {r["id"] for r in results}
        results = [r for r in old["mutants"] if r["id"] not in mine] + results
        if not xres:
            xres = old.get("non_code", [])
    for r in xres:
        print(f"{r['id']} {r['status']} {r['killers']}")
    rep = write_reports(results, xres, control_info=dict(n=len(control), not_passing=bad_ctrl, stale=stale_ctrl, seconds=round(dt, 1)),
                        cfg_path=cfg_path, code=code, wall=time.time() - t0)
    verdict = rep["summary"]["verdict"]
    return 0 if verdict == "PASS" else 1


def write_reports(results, xres, control_info, cfg_path, code, wall):
    results = sorted(results, key=lambda r: r["id"])
    nonequiv = [r for r in results if not r["equivalent"]]
    killed = [r for r in nonequiv if r["status"] == "KILLED"]
    m10 = next((r for r in results if r["id"] == "M10"), None)
    bad_ctrl = control_info["not_passing"]
    verdict = ("PASS" if not bad_ctrl and len(killed) == len(nonequiv) and all(r["status"] == "EQUIVALENT-OK" for r in results if r["equivalent"])
               and all(r["status"] == "KILLED" for r in xres) and (m10 is None or "REQUIREMENT NOT MET" not in m10.get("detail", "")) else "FAIL")
    rep = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), config=cfg_path, code_root=code, subset=SUBSET,
               control=control_info, mutants=results, non_code=xres,
               summary=dict(non_equivalent=len(nonequiv), killed=len(killed),
                            survived=[r["id"] for r in nonequiv if r["status"] != "KILLED"], verdict=verdict),
               wall_s=round(wall, 1))
    json.dump(rep, open(os.path.join(HERE, "report.json"), "w"), indent=1)
    json.dump(rep, open(os.path.join(HERE, f"kill_matrix_{datetime.date.today().isoformat()}.json"), "w"), indent=1)
    L = [f"# Mutation report {rep['date']}", "", f"- verdict: **{verdict}**; killed by a physics/claims test: {len(killed)}/{len(nonequiv)} non-equivalent mutants;"
         f" control: {control_info['n']} checks, {len(bad_ctrl)} failing (undecidable in control and excluded: {len(control_info.get('stale', []))} STALE); wall {rep['wall_s']:.0f} s",
         f"- subset: `{SUBSET}`; a kill by the static tripwires T1-T4 alone is reported as 'KILLED (static only)' and does not count", "",
         "| mutant | file | class | expected killers | status | actual killers | note |", "|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['id']} | {r['file']} | {r['cls']} | {', '.join(r['expected'])} | {r['status']} | {', '.join(r['killers'])} | "
                 f"{r.get('detail', '') or ('phase ' + r.get('phase', ''))} |")
    L += ["", "| non-code | status | killers | change |", "|---|---|---|---|"]
    for r in xres:
        L.append(f"| {r['id']} | {r['status']} | {', '.join(r['killers'])} | {r['detail']} |")
    if bad_ctrl:
        L += ["", "Control checks failing (cannot count as killers): " + "; ".join(bad_ctrl)]
    open(os.path.join(HERE, "report.md"), "w").write("\n".join(L) + "\n")
    print(verdict, f"{len(killed)}/{len(nonequiv)} killed", f"{rep['wall_s']:.0f} s")
    return rep


def resummarize():
    rep = json.load(open(os.path.join(HERE, "report.json")))
    c = rep["control"]
    if "stale" not in c:      # reports written before STALE was separated from FAIL
        c["stale"] = [x for x in c["not_passing"] if x.startswith(("P1", "P2", "G2: claims not yet"))]
        c["not_passing"] = [x for x in c["not_passing"] if x not in c["stale"]]
    write_reports(rep["mutants"], rep["non_code"], c, rep["config"], rep["code_root"], rep["wall_s"])


if __name__ == "__main__":
    if "--resummarize" in sys.argv:
        sys.exit(resummarize())
    sys.exit(main())
