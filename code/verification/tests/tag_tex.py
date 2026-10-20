#!/usr/bin/env python3
"""Insert \\claim{id}{printed} macros into a COPY of the paper source, from the anchors in claims.json.

    python3 tests/tag_tex.py PAPER_DIR [--claims claims.json] [--dry-run] [--only-status active,provisional,...]

* adds  \\newcommand{\\claim}[2]{#2}  to main.tex (after the custom-commands block) if it is not there;
* every claim's anchor must occur exactly once in its file and show the registered printed text at 'offset';
  otherwise the claim is reported and left untagged (re-anchor it, or tag by hand);
* already-tagged claims are skipped, so the tool is idempotent.
The macro prints its second argument verbatim, so the PDF text must not change: check with pdftotext (X0).
The lead runs this on the Overleaf copy after the reruns, once the pending values are filled in.
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import texscan

MACRO = "\\newcommand{\\claim}[2]{#2}          % verification tag: \\claim{id}{printed} prints #2 (claims.json)"


def tag(paper, claims, dry=False, statuses=None):
    files = {}
    done, skipped = [], []
    for cid, c in claims.items():
        if statuses and c["status"] not in statuses:
            continue
        f = c["tex"]["file"]
        if f not in files:
            files[f] = open(os.path.join(paper, f)).read()
        files.setdefault("__edits__" + f, [])
    for cid, c in claims.items():
        if statuses and c["status"] not in statuses:
            continue
        f = c["tex"]["file"]; t = files[f]
        if any(x[0] == cid for x in texscan.claims_in(t)):
            skipped.append(f"{cid}: already tagged"); continue
        an = c["tex"]["anchor"]
        if t.count(an) != 1:
            skipped.append(f"{cid}: anchor found {t.count(an)} times"); continue
        p = t.find(an) + c["tex"]["offset"]
        if t[p:p + len(c["printed"])] != c["printed"]:
            skipped.append(f"{cid}: text at anchor is '{t[p:p + len(c['printed'])]}', not '{c['printed']}'"); continue
        files["__edits__" + f].append((p, p + len(c["printed"]), f"\\claim{{{cid}}}{{{c['printed']}}}", cid))
    for f in [k for k in files if not k.startswith("__edits__")]:
        t = files[f]
        eds = sorted(files["__edits__" + f], reverse=True)
        spans = sorted((a, b) for a, b, _, _ in eds)
        for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
            if a2 < b1:
                raise SystemExit(f"overlapping claims in {f} at {a1}-{b1} and {a2}-{b2}")
        for a, b, rep, cid in eds:
            t = t[:a] + rep + t[b:]
            done.append(cid)
        files[f] = t
    mp = os.path.join(paper, "main.tex")
    main = files.get("main.tex", open(mp).read())
    if "\\newcommand{\\claim}" not in main:
        anchor = "\\newcommand{\\TODO}"
        i = main.find(anchor)
        j = main.find("\n", i) + 1 if i >= 0 else main.find("\\begin{document}")
        main = main[:j] + MACRO + "\n" + main[j:]
        files["main.tex"] = main
    if not dry:
        for f, t in files.items():
            if not f.startswith("__edits__"):
                open(os.path.join(paper, f), "w").write(t)
    return done, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paper"); ap.add_argument("--claims", default=os.path.join(os.path.dirname(HERE), "claims.json"))
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--only-status")
    a = ap.parse_args()
    claims = json.load(open(a.claims))["claims"]
    done, skipped = tag(a.paper, claims, a.dry_run, set(a.only_status.split(",")) if a.only_status else None)
    print(f"tagged {len(done)} claims; skipped {len(skipped)}")
    for s in skipped:
        print("  ", s)
    return 0 if not [s for s in skipped if "already" not in s] else 1


if __name__ == "__main__":
    sys.exit(main())
