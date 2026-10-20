#!/usr/bin/env python3
"""Single entry point of the verification suite (protocol sec. 1, 6).

  python3 verify_all.py --fast                 engine/script/claims checks, about 6-10 min
  python3 verify_all.py --full --jobs 10       adds production-grid convergence and recompute (Mac, hours)
  python3 verify_all.py --mutants              mutation harness (mutation/run_mutants.py)
  python3 verify_all.py --scan-numbers         propose numbers_allowlist.json entries (Abir approves)
  python3 verify_all.py --ledger-legacy        hash the engine/scripts and record the legacy mapping
  python3 verify_all.py --register-knobs       rewrite knobs.json from the current code (after review)

Options: --config FILE (default config.json; config_cloud.json for the cloud copy), --code-root DIR (override),
         --only IDS (comma list of test ids or families, e.g. R4,H or T), --order IDS (run these first),
         -x (stop at the first FAIL), --fixtures-out FILE.npz, --no-report, --label TEXT, --json-out FILE.
Writes reports/YYYY-MM-DD_HHMM_<mode>.md and .json; exits nonzero on any FAIL (--full: also SKIP/STALE).
"""
import os, sys, json, time, glob, argparse, datetime, traceback, importlib, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("lib", "tests", "adapters"):
    sys.path.insert(0, os.path.join(HERE, sub))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import vcore
from vcore import Result, REGISTRY, Ctx, load_config, env_info, fmt_meas
import prov

MODULES = ["test_T_static", "test_U_units", "test_L_literature", "test_H_symmetry", "test_R_reference",
           "test_I_invariance", "test_S_smoke", "test_C_convergence", "claims_check", "test_P_prov", "test_X_refself"]
FAMILY_ORDER = "TULHRISCGPX"


def load_tests():
    for m in MODULES:
        importlib.import_module(m)


def select(mode, only, order):
    tests = [t for t in REGISTRY if mode in t["modes"]]
    if only:
        keys = set(only.split(","))
        tests = [t for t in tests if t["id"] in keys or t["family"] in keys]
    tests.sort(key=lambda t: (FAMILY_ORDER.index(t["family"]) if t["family"] in FAMILY_ORDER else 99, t["id"]))
    if order:
        first = order.split(",")
        rank = {k: i for i, k in enumerate(first)}
        tests.sort(key=lambda t: rank.get(t["id"], rank.get(t["family"], len(first))))
    return tests


def run(args):
    cfg = load_config(args.config)
    code_root = os.path.abspath(args.code_root or (cfg["paths"]["code_root"] if os.path.isabs(cfg["paths"]["code_root"])
                                                   else os.path.join(HERE, cfg["paths"]["code_root"])))
    scratch = tempfile.mkdtemp(prefix="verif_")
    mode = "full" if args.full else "fast"
    ctx = Ctx(cfg, mode, code_root, scratch)
    ctx.args = args
    load_tests()
    tests = select(mode, args.only, args.order)
    t0 = time.time()
    if mode == "full" and any(t["family"] == "C" for t in tests):
        import test_C_convergence
        test_C_convergence.prefetch(ctx, max(1, args.jobs))
    results = []
    stop = False
    for t in tests:
        if stop:
            break
        ts = time.time()
        try:
            r = t["func"](ctx)
            rs = r if isinstance(r, list) else [r]
        except Exception as e:
            rs = [Result(t["id"], t["title"], "FAIL", detail="EXCEPTION " + "".join(traceback.format_exception_only(type(e), e)).strip()
                         + " | " + traceback.format_exc().strip().splitlines()[-3][:200])]
        dt = time.time() - ts
        for r in rs:
            r.seconds = dt / len(rs)
            r.title = t["title"]
        results += rs
        if not args.quiet:
            for r in rs:
                print(f"[{r.status:5s}] {r.id:4s} {r.name[:90]:90s} {fmt_meas(r.measured):>12s} tol {r.tol}  ({dt:.1f}s)", flush=True)
        if args.x and any(r.status == "FAIL" for r in rs):
            stop = True
    wall = time.time() - t0
    apply_known(results)
    if args.accept_known:
        accept_known(results, args.accept_known)
    if not args.quiet:
        for r in results:
            if r.status == "KNOWN" or "known failure" in r.detail:
                print(f"  -> {r.id} {r.name[:70]}: {r.status} ({r.detail[:110]})")
    bad = [r for r in results if r.status == "FAIL" or (mode == "full" and r.status in ("SKIP", "STALE"))]
    summary = {s: sum(r.status == s for r in results) for s in vcore.STATUSES}
    meta = dict(mode=mode, label=args.label, started=datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
                wall_s=round(wall, 1), code_root=code_root, config=cfg["_config_file"],
                tree_sha=prov.tree_sha(code_root, cfg["engine_files"]), engine_files=prov.file_shas(code_root, cfg["engine_files"]),
                script_shas=prov.script_shas(cfg, code_root),
                ref_sha=prov.sha256_file(os.path.join(HERE, "ref", "refimpl.py")), env=env_info(), summary=summary,
                verdict="PASS" if not bad else "FAIL")
    if wall > cfg.get("budget_warn_s", 600) and mode == "fast":
        print(f"WARNING: fast mode took {wall:.0f} s (> {cfg.get('budget_warn_s', 600)} s budget)")
    if args.fixtures_out:
        import numpy as np
        np.savez(args.fixtures_out, **{k: v for k, v in ctx.fixtures.items()})
    if args.json_out:
        json.dump(dict(meta=meta, results=[r.as_dict() for r in results]), open(args.json_out, "w"), indent=1)
    if not args.no_report:
        write_report(cfg, meta, results)
    if not bad and not args.only and not args.no_report:
        prov.write_lock(code_root, cfg["scripts"], mode, meta.get("report", ""), os.path.join(HERE, "verified.lock"), cfg)
    print(f"\n{meta['verdict']}: {summary}  wall {wall:.0f} s  tree {meta['tree_sha'][:16]}")
    if not args.keep_scratch:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
    return 0 if not bad else 1


KNOWN_FILE = os.path.join(HERE, "known_failures.json")


def _items(detail):
    return set(x.strip() for x in (detail or "").split("; ") if x.strip())


def apply_known(results):
    """Known-failure baselines (red team RT26/RT29): a FAIL whose measured value and detail items equal the recorded
    baseline becomes KNOWN; any NEW or CHANGED item keeps it FAIL (and the detail lists only what changed); a known
    failure that partly went away becomes WARN (update the baseline)."""
    if not os.path.exists(KNOWN_FILE):
        return
    base = json.load(open(KNOWN_FILE))["entries"]
    for r in results:
        b = base.get(f"{r.id}|{r.name}")
        if b is not None and r.status == "PASS":
            r.status = "WARN"; r.detail = (f"known failure resolved ({b['reason']}): the check passes now; move the baseline to "
                                           f"'resolved' in known_failures.json. " + (r.detail or ""))
            continue
        if r.status != "FAIL" or b is None:
            continue
        cur, old = _items(r.detail), set(b["items"])
        new, gone = sorted(cur - old), sorted(old - cur)
        if not new and not gone and str(r.measured) == str(b["measured"]):
            r.status = "KNOWN"; r.detail = f"known failure ({b['reason']}), unchanged since {b['accepted']}: " + r.detail
        elif new:
            r.detail = f"CHANGED known failure ({b['reason']}): NEW/CHANGED: " + "; ".join(new) + (" | no longer: " + "; ".join(gone) if gone else "")
        else:
            r.status = "WARN"; r.detail = f"known failure partly resolved (update known_failures.json): no longer: " + "; ".join(gone)


def accept_known(results, reason):
    data = json.load(open(KNOWN_FILE)) if os.path.exists(KNOWN_FILE) else dict(
        _comment="Per-check expected-failure baselines (documented paper/data findings). A FAIL equal to its baseline is "
                 "reported as KNOWN; any new or changed item is a FAIL. Rewrite only with --accept-known after review.", entries={})
    for r in results:
        if r.status == "FAIL":
            k = f"{r.id}|{r.name}"
            old = data["entries"].get(k, {})
            data["entries"][k] = dict(measured=str(r.measured), items=sorted(_items(r.detail)), reason=old.get("reason", reason),
                                      accepted=datetime.date.today().isoformat())
            r.status = "KNOWN"
    json.dump(data, open(KNOWN_FILE, "w"), indent=1)
    print("known_failures.json written:", len(data["entries"]), "entries")


def write_report(cfg, meta, results):
    rd = os.path.join(HERE, cfg["paths"]["reports"]) if not os.path.isabs(cfg["paths"]["reports"]) else cfg["paths"]["reports"]
    os.makedirs(rd, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    base = os.path.join(rd, f"{stamp}_{meta['mode']}")
    meta["report"] = os.path.relpath(base + ".md", HERE)
    json.dump(dict(meta=meta, results=[r.as_dict() for r in results]), open(base + ".json", "w"), indent=1)
    L = [f"# Verification report ({meta['mode']} mode) {meta['started']}", "",
         f"- verdict: **{meta['verdict']}**  {meta['summary']}",
         f"- wall time: {meta['wall_s']} s; host {meta['env']['host']} ({meta['env']['machine']}), python {meta['env']['python']}, numpy {meta['env']['numpy']}",
         f"- code root: `{meta['code_root']}`", f"- tree_sha: `{meta['tree_sha']}`",
         f"- ref/refimpl.py sha256: `{meta['ref_sha'][:16]}`", ""]
    known = [r for r in results if r.status == "KNOWN"]
    if known:
        L += ["## Known failures (documented findings, unchanged from known_failures.json)", "", "| id | check | measured | detail |", "|---|---|---|---|"]
        for r in known:
            L.append(f"| {r.id} | {r.name} | {fmt_meas(r.measured)} | {r.detail.replace('|', '/')[:900]} |")
        L.append("")
    fails = [r for r in results if r.status in ("FAIL", "STALE", "WARN")]
    if fails:
        L += ["## Not passing", "", "| id | check | status | measured | tolerance | detail |", "|---|---|---|---|---|---|"]
        for r in fails:
            L.append(f"| {r.id} | {r.name} | {r.status} | {fmt_meas(r.measured)} | {r.tol} | {r.detail.replace('|', '/')[:600]} |")
        L.append("")
    L += ["## All checks", "", "| id | check | status | measured | tolerance | s | detail |", "|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r.id} | {r.name} | {r.status} | {fmt_meas(r.measured)} | {r.tol} | {r.seconds:.1f} | {r.detail.replace('|', '/')[:300]} |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("report:", base + ".md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    for f in ("--fast", "--full", "--mutants", "--scan-numbers", "--ledger-legacy", "--register-knobs"):
        g.add_argument(f, action="store_true")
    ap.add_argument("--config"); ap.add_argument("--code-root"); ap.add_argument("--only"); ap.add_argument("--order")
    ap.add_argument("-x", action="store_true"); ap.add_argument("--fixtures-out"); ap.add_argument("--json-out")
    ap.add_argument("--no-report", action="store_true"); ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true"); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--accept-known", metavar="REASON", help="record the current FAILs as known-failure baselines")
    ap.add_argument("--keep-scratch", action="store_true")
    args, rest = ap.parse_known_args()
    if args.mutants:
        sys.argv = [os.path.join(HERE, "mutation", "run_mutants.py")] + rest + (["--config", args.config] if args.config else [])
        sys.path.insert(0, os.path.join(HERE, "mutation"))
        import run_mutants
        return run_mutants.main()
    if args.ledger_legacy:
        import ledger
        return ledger.main(load_config(args.config), args.code_root)
    if args.register_knobs:
        import test_T_static
        cfg = load_config(args.config)
        cr = args.code_root or (cfg["paths"]["code_root"] if os.path.isabs(cfg["paths"]["code_root"]) else os.path.join(HERE, cfg["paths"]["code_root"]))
        test_T_static.register(os.path.abspath(cr), os.path.join(HERE, "knobs.json"))
        print("knobs.json written: review the diff before committing")
        return 0
    if args.scan_numbers:
        import claims_check
        return claims_check.scan_numbers_cli(load_config(args.config))
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
