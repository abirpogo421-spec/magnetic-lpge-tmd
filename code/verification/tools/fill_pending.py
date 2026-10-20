#!/usr/bin/env python3
"""After the v7 queue: compute the pending and provisional claims from the v7 outputs and propose registry updates.

    python3 tools/fill_pending.py [--config config.json] [--write]

Without --write it prints claim, printed text, new value, and what the paper would print. With --write it stores the
values in claims.json, sets status 'active', and appends a history record {date, old, new, reason, report}.
Claims it cannot compute yet (missing outputs) stay pending. Afterwards: verify_all.py --fast, then tag the paper.
"""
import os, sys, json, glob, argparse, datetime
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "tests", "adapters"):
    sys.path.insert(0, os.path.join(HERE, sub))
sys.path.insert(0, HERE)
from vcore import Ctx, load_config
import claims_recipes as CR
from claims_check import check_printed, fmt



ACCESS = {}


def compute(ctx, reg):
    """Recompute every pending_rerun / provisional claim from its registered recipe (claims_recipes.py, v7 outputs).
    Claims whose v7 output is still missing or unfinished raise PendingData and stay pending."""
    v, still = {}, []
    for cid, c in reg["claims"].items():
        if c["status"] not in ("pending_rerun", "provisional") or not c.get("recipe"):
            continue
        r = c["recipe"]
        ctx.memo.pop("_access", None)
        try:
            v[cid] = CR.RECIPES[r["name"]](ctx, **r.get("args", {}))
            ACCESS[cid] = sorted(ctx.memo.get("_access", set()))
        except CR.PendingData as ex:
            still.append(f"{cid}: {ex}")
    return v, still


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    code = cfg["paths"]["code_root"] if os.path.isabs(cfg["paths"]["code_root"]) else os.path.join(HERE, cfg["paths"]["code_root"])
    ctx = Ctx(cfg, "fast", code, "/tmp")
    reg = json.load(open(os.path.join(HERE, "claims.json")))
    vals, still = compute(ctx, reg)
    for s_ in still:
        print("still pending:", s_)
    if not vals:
        print("nothing to fill (outputs in", ctx.path("v7_out"), ")"); return 1
    today = datetime.date.today().isoformat()
    for cid, val in sorted(vals.items()):
        c = reg["claims"].get(cid)
        if c is None:
            continue
        ok, rend = check_printed(val, c["display"], c["printed"])
        print(f"{cid:28s} paper '{c['printed']}'  v7 value {val:.6g} -> '{rend}'  {'OK' if ok else 'PAPER MUST CHANGE (edit the text, then re-run tools/build_claims.py)'}")
        if a.write:
            c["history"].append(dict(date=today, old=c.get("value"), new=val, old_printed=c["printed"], new_printed=c["printed"],
                                     old_status=c["status"], new_status="active", changed=["value", "status"],
                                     reason="filled from v7 outputs (tools/fill_pending.py)"))
            c["value"] = val; c["status"] = "active"
            if ACCESS.get(cid):                                   # register the data files (sha) the value came from (G1)
                from vcore import sha256_file
                src = c.setdefault("source", {}); files = src.get("data", [])
                files = sorted(set(([files] if isinstance(files, str) else files) + ACCESS[cid]))
                src["data"] = files
                src["data_sha"] = {f: sha256_file(os.path.join(ctx.path(f.split(":", 1)[0]), f.split(":", 1)[1])) for f in files}
    if a.write:
        json.dump(reg, open(os.path.join(HERE, "claims.json"), "w"), indent=1, ensure_ascii=False)
        print("claims.json updated; run verify_all.py --fast")
    return 0


if __name__ == "__main__":
    sys.exit(main())
