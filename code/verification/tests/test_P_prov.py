"""P: provenance of figures and data (protocol sec. 6).

P1 every \\includegraphics target and every claim data source has a provenance record
   (a <file>.prov.json sidecar, or until the v7 queue finishes, legacy embedded metadata + the engine ledger)
P2 sidecar/legacy hash is current or in the ledger; pre-fix (v6) engines are STALE; a figure older than any of its
   data files is FAIL (data rerun without replot)
P4 PDF metadata (/Keywords = tree_sha, /Subject = sidecar) equals the sidecar
P5 resumed/production files: stored parameters equal the argv/settings, and the sign convention is declared
P3 (full) rebuilt figures byte-identical on the same host; raster diff <= 0.1% across hosts
"""
import os, re, json, glob
import numpy as np
from vcore import check, Result, exact, SUITE_DIR, sha256_file

# figure (as included in the paper) -> producing data (v7 names; v6 names when the v7 file is absent)
FIG_DATA = {
    "results/lpge_map_grid.pdf": (["v7_out:prod7_{m}_F.npz"], ["data_v6:prod_{m}_F.npz"], "v7jobs/fig_v7.py"),
    "results/lpge_edge_sign_grid.pdf": (["v7_out:edge7_{m}_TNN.json"], ["data_v6:fig10_thr.json"], "v7jobs/fig_v7.py"),
    "results/lpge_field_merged.pdf": (["v7_out:prod7_{m}_F.npz"], ["data_v6:prod_{m}_F.npz"], "fig_main.py (v6)"),
    "results/chi_vs_offset.pdf": (["v7_out:chiwR9v7_{m}_{model}_F_0.001.json"], ["data_v6:chiwR9_{m}_{model}_F_0.001.json"], "fig_main.py (v6)"),
    "results/threshold_models.pdf": (["v7_out:edge7_{m}_{model}.json", "kp_out:kp_fp_all_v7.json"], ["data_v6:threshold_models_thr.json"], "fig_main.py (v6)"),
    "results/lpge_lambda_grid.pdf": (["v7_out:lam7_{m}_F.npz"], ["data_v6:lam_{m}_F.npz"], "fig_main.py (v6)"),
    "figures/theory/kp_spectra.pdf": (["kp_out:kp_spectra.json"], [], "fig_main.py (v6)"),
}
MATS = ["MoS2", "MoSe2", "WS2", "WSe2"]


def sidecar(path):
    """Provenance sidecar of a file: <file>.prov.json (prov.py) or <stem>.prov.json (fig_v7_all.py)."""
    for c in (path + ".prov.json", os.path.splitext(path)[0] + ".prov.json"):
        if os.path.exists(c):
            return c
    return path + ".prov.json"


def _ledger():
    return json.load(open(os.path.join(SUITE_DIR, "engine_ledger.json")))


def _expand(ctx, spec):
    key, pat = spec.split(":")
    out = []
    for m in MATS:
        for model in ("TNN", "NN"):
            out.append(os.path.join(ctx.path(key), pat.format(m=m, model=model)))
    return sorted(set(out))


def _engine_hash(path):
    """Embedded legacy engine hash of a data file: (kind, hash) or (None, None)."""
    try:
        if path.endswith(".json"):
            d = json.load(open(path))
            for k in ("lpge_sha256", "kp2_sha256"):
                if k in d:
                    return k, d[k]
                if isinstance(d.get("meta"), dict) and k in d["meta"]:
                    return k, d["meta"][k]
        elif path.endswith(".npz"):
            d = np.load(path, allow_pickle=True)
            if "meta" in d.files:
                m = json.loads(str(d["meta"]))
                if "lpge_sha256" in m:
                    return "lpge_sha256", m["lpge_sha256"]
    except Exception:
        return None, None
    return None, None


def classify(path, led=None):
    """PASS (current/ledger) | STALE (pre-fix or unknown engine) | MISSING; with detail."""
    led = led or _ledger()
    if not os.path.exists(path):
        return "MISSING", "not found"
    side = sidecar(path)
    if os.path.exists(side):
        s = json.load(open(side))
        if s.get("sha256") != sha256_file(path):
            return "FAIL", "sidecar sha256 does not match the file"
        ok = s.get("tree_sha") in {a["tree_sha"] for a in led["accepted"]} | {c.get("tree_sha") for c in led["certificates"]}
        return ("PASS", "sidecar, accepted tree") if ok else ("STALE", f"sidecar tree {s.get('tree_sha', '')[:12]} not accepted")
    k, h = _engine_hash(path)
    if h is None:
        return "STALE", "no sidecar and no embedded engine hash"
    if k == "lpge_sha256":
        if h in {e["legacy_lpge_sha256"] for e in led["legacy"]}:
            return "PASS", f"legacy lpge {h[:12]} -> ledger tree"
        if h in led.get("pre_fix", {}):
            return "STALE", f"pre-fix engine {h[:12]} ({led['pre_fix'][h][:40]}...)"
        return "STALE", f"unknown engine lpge {h[:12]}"
    if k == "kp2_sha256":
        kp2 = next(iter(e["files"]["kp/kp2.py"] for e in led["legacy"]))
        return ("PASS", "kp2 hash == ledger kp2") if h == kp2 else ("STALE", f"kp2 {h[:12]} differs from ledger")
    return "STALE", "?"


def includegraphics(ctx):
    from claims_check import load_tex
    out = []
    for f, t in load_tex(ctx).items():
        for m in re.finditer(r"(?<!%)\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", t):
            out.append(m.group(2))
    return out


def _fig_path(ctx, rel):
    root = ctx.path("paper_root")
    for c in (os.path.join(root, "figures", rel), os.path.join(root, rel)):
        if os.path.exists(c):
            return c
    return None


def figure_vs_data(fig, datas):
    """X3 rule: a figure older than any of its data files was not replotted after a rerun."""
    if fig is None or not os.path.exists(fig):
        return "SKIP", "figure not in this paper copy"
    newer = [d for d in datas if os.path.exists(d) and os.path.getmtime(d) > os.path.getmtime(fig) + 1]
    return ("FAIL", "data newer than figure: " + ", ".join(os.path.basename(d) for d in newer)) if newer else ("PASS", "figure newer than data")


@check("P1", "P", "provenance record exists for every figure and every claim data source")
def p1(ctx):
    led = _ledger()
    figs = includegraphics(ctx)
    miss_fig = [f for f in figs if _fig_path(ctx, f) is None]
    no_side = [f for f in figs if _fig_path(ctx, f) and not os.path.exists(sidecar(_fig_path(ctx, f)))]
    from claims_check import registry
    srcs = set()
    for c in registry()["claims"].values():
        d = c.get("source", {}).get("data")
        srcs.update([d] if isinstance(d, str) else (d or []))
    srcs = sorted(srcs)
    bad = []
    for s in srcs:
        key, rel_ = s.split(":", 1)
        st, why = classify(os.path.join(ctx.path(key), rel_), led)
        if st != "PASS":
            bad.append(f"{rel_}: {why}")
    out = [Result("P1", "figures in the paper: file present and sidecar", "STALE" if (miss_fig or no_side) else "PASS",
                  measured=f"{len(figs)} figures, {len(miss_fig)} not in this copy, {len(no_side)} without sidecar",
                  detail=("not in this paper copy: " + ", ".join(miss_fig) + "; " if miss_fig else "") +
                         ("no sidecar (fig scripts do not call prov.savefig_prov yet): " + ", ".join(no_side) if no_side else "")),
           Result("P1", "claim data sources with provenance (sidecar or ledger-known embedded hash)", "STALE" if bad else "PASS",
                  measured=f"{len(srcs)} sources", detail="; ".join(bad))]
    return out


@check("P2", "P", "data provenance: engine hash current or in ledger; figure newer than its data")
def p2(ctx):
    led = _ledger()
    rows, stale, fail, figchk = [], [], [], []
    for fig, (v7, v6, script) in FIG_DATA.items():
        files = [p for s in v7 for p in _expand(ctx, s) if os.path.exists(p)]
        gen = "v7"
        if not files:
            files = [p for s in v6 for p in _expand(ctx, s) if os.path.exists(p)]
            gen = "v6"
        sts = [classify(p, led) for p in files]
        n_ok = sum(s == "PASS" for s, _ in sts)
        why = sorted({w for s, w in sts if s != "PASS"})
        (stale if n_ok < len(files) or not files else rows).append(f"{fig}: {gen} data, {n_ok}/{len(files)} current" + (f" ({'; '.join(why)})" if why else ""))
        st, w = figure_vs_data(_fig_path(ctx, fig), files)
        if st == "FAIL":
            fail.append(f"{fig}: {w}")
        figchk.append(st)
    return [Result("P2", "figure data produced by an accepted engine", "STALE" if stale else "PASS", measured=f"{len(rows)} ok, {len(stale)} stale",
                   detail="; ".join(stale + rows)),
            Result("P2", "figure replotted after its data (mtime)", "FAIL" if fail else ("SKIP" if all(s == "SKIP" for s in figchk) else "PASS"),
                   detail="; ".join(fail) or "figures not present in this paper copy" if all(s == "SKIP" for s in figchk) else "; ".join(fail))]


@check("P4", "P", "PDF metadata (/Keywords tree_sha, /Subject sidecar) equals the sidecar")
def p4(ctx):
    figs = [(f, _fig_path(ctx, f)) for f in includegraphics(ctx)]
    have = [(f, p) for f, p in figs if p and p.endswith(".pdf")]
    if not have:
        return Result("P4", "PDF metadata", "SKIP", detail="no figure PDFs in this paper copy")
    bad, legacy = [], []
    try:
        import pypdf
    except Exception:
        return Result("P4", "PDF metadata", "SKIP", detail="pypdf not installed")
    for f, p in have:
        side = sidecar(p)
        meta = pypdf.PdfReader(p).metadata or {}
        if not os.path.exists(side):
            legacy.append(f); continue
        s = json.load(open(side))
        if "tree_sha" in s:                                   # prov.savefig_prov format
            ok_ = meta.get("/Keywords") == s.get("tree_sha") and json.loads(meta.get("/Subject", "{}")).get("tree_sha") == s.get("tree_sha")
        else:                                                 # fig_v7_all.py format: the PDF /Subject holds the sidecar record
            try:
                subj = json.loads(meta.get("/Subject", "{}"))
            except Exception:
                subj = {}
            ok_ = all(subj.get(k) == s.get(k) for k in ("script_sha256", "engine_sha256", "date")) and \
                str(meta.get("/Keywords", "")).endswith(str(s.get("engine_sha256")))
        if not ok_:
            bad.append(f)
    st = "FAIL" if bad else ("STALE" if legacy else "PASS")
    return Result("P4", "PDF metadata vs sidecar", st, measured=f"{len(have)} PDFs, {len(have) - len(legacy) - len(bad)} match, {len(bad)} differ",
                  detail="; ".join((["metadata differs from sidecar: " + ", ".join(bad)] if bad else []) + (["legacy (no sidecar): " + ", ".join(legacy)] if legacy else [])))


@check("P5", "P", "resumed/production files: stored parameters == argv, sign convention declared")
def p5(ctx):
    out = []
    v7 = ctx.path("v7_out")
    bad, n = [], 0
    for p in sorted(glob.glob(os.path.join(v7, "edge7_*.json"))):
        d = json.load(open(p)); n += 1
        parts = os.path.basename(p)[6:-5].split("_")
        func = parts[2] if len(parts) > 2 else "GGA"
        if not (d.get("model") == parts[1] and d.get("functional") == func and d.get("R") == 0.9 and d.get("nu") == 0.005
                and d.get("h") == 0.0007):
            bad.append(f"{os.path.basename(p)}: stored model/functional/R/nu/h = {d.get('model')}/{d.get('functional')}/{d.get('R')}/{d.get('nu')}/{d.get('h')}")
        if d.get("lpge_sha256") not in {e["legacy_lpge_sha256"] for e in json.load(open(os.path.join(SUITE_DIR, "engine_ledger.json")))["legacy"]}:
            bad.append(f"{os.path.basename(p)}: engine {str(d.get('lpge_sha256'))[:12]} not in ledger")
    for p in sorted(glob.glob(os.path.join(v7, "prod7_*_F.npz"))):
        m = json.loads(str(np.load(p, allow_pickle=True)["meta"])); n += 1
        if not (m.get("N") == 600 and m.get("nu_eV") == 0.01 and "physical" in m.get("units", "")):
            bad.append(f"{os.path.basename(p)}: N/nu/units {m.get('N')}/{m.get('nu_eV')}/{m.get('units')}")
    out.append(Result("P5", "v7 outputs present: stored settings == queue argv", ("FAIL" if bad else "PASS") if n else "SKIP",
                      measured=f"{n} files", detail="; ".join(bad) or ("no v7 outputs in " + v7 if not n else "")))
    src = open(os.path.join(ctx.code_root, "v7jobs", "edge_v7.py")).read()
    resume_full = re.search(r"assert out\.get\('(h|R|nu|functional)'\)", src)
    out.append(Result("P5", "edge_v7 resume guard checks the run parameters, not only the engine hash", "PASS" if resume_full else "WARN",
                      detail="edge_v7 resumes edge7_*.json after checking only lpge_sha256: a rerun with different h/DB/FUNC appends to "
                             "an old file (finding 9.3). P5 above checks the stored h, R, nu, model, functional of every output."))
    lam = open(os.path.join(ctx.code_root, "v7jobs", "lam_v7.py")).read()
    codesign = "chi_SI_per_T=chi" in lam and "-2" not in lam.split("chi[i] =")[1].split("\n")[0]
    out.append(Result("P5", "sign convention declared for every stored array", "WARN" if codesign else "PASS",
                      detail="lam_v7 stores the CODE-sign chi under 'chi_SI_per_T' without a sign_convention field; fig_main.py "
                             "applies PHYS = -2 later (finding 9.2). S5 checks the value with x(-2)." if codesign else ""))
    return out


@check("P3", "P", "figure rebuild reproducibility (byte-identical on host, raster <= 0.1% across hosts)", modes=("full",))
def p3(ctx):
    return Result("P3", "figure rebuild", "SKIP", detail="needs the v7 data and fig scripts that call prov.savefig_prov (after the queue)")


# ------------------------------------------------------------------------------------------------ P6
V7_FIGS = ["results/lpge_field_merged.pdf", "results/lpge_map_grid.pdf", "results/lpge_edge_sign_grid.pdf",
           "results/chi_vs_offset.pdf", "results/threshold_models.pdf", "results/lpge_lambda_grid.pdf"]


@check("P6", "P", "paper figures (Figs. 7-12) are the v7 figure-script outputs; Fig. 7 legend energies == vH.E claims")
def p6(ctx):
    """A figure PDF in the paper must carry the sidecar written by fig_v7_all.py (script + input shas); the inputs must be
    the current v7 files. Independently of sidecars, the van Hove energies printed in the Fig. 7 legend (pdftotext)
    must equal the registered vH.E.* claims: a legend that still shows the v6 energies means the PDF was not replaced."""
    import subprocess
    from claims_check import registry
    reg = registry()["claims"]
    no_side, wrong_in, ok = [], [], []
    for f in V7_FIGS:
        p = _fig_path(ctx, f)
        if p is None:
            continue
        side = sidecar(p)
        if not os.path.exists(side):
            no_side.append(f); continue
        s = json.load(open(side)); bad = []
        ins = s.get("inputs") or []
        ins = [dict(path=k, sha256=v) for k, v in ins.items()] if isinstance(ins, dict) else ins
        for i in ins:
            q = os.path.join(ctx.path("v7_out"), os.path.basename(i["path"]))
            if os.path.exists(q) and sha256_file(q) != i.get("sha256"):
                bad.append(os.path.basename(i["path"]))
        if s.get("output_sha256") and sha256_file(p) != s["output_sha256"]:
            bad.append("the PDF itself differs from the sidecar's output_sha256")
        if os.path.basename(s.get("script", "")) != "fig_v7_all.py":
            bad.append(f"script {s.get('script')}")
        (wrong_in if bad else ok).append(f + (f" (inputs changed: {', '.join(bad)})" if bad else ""))
    res = [Result("P6", "figure PDFs drawn by the v7 figure script from the current v7 data (sidecar)",
                  "FAIL" if wrong_in else ("STALE" if no_side else "PASS"), measured=f"{len(ok)} ok, {len(no_side)} without sidecar, {len(wrong_in)} stale inputs",
                  detail=("no sidecar, provenance unknown: " + ", ".join(no_side) + "; " if no_side else "") + "; ".join(wrong_in))]
    p = _fig_path(ctx, "results/lpge_field_merged.pdf")
    if p is None:
        res.append(Result("P6", "Fig. 7 legend energies vs vH.E claims", "SKIP", detail="figure not in this paper copy"))
        return res
    try:
        txt = subprocess.run(["pdftotext", "-layout", p, "-"], capture_output=True, text=True, timeout=60).stdout
    except Exception as ex:
        res.append(Result("P6", "Fig. 7 legend energies vs vH.E claims", "SKIP", detail=f"pdftotext unavailable ({ex})"))
        return res
    found, bad = {}, []
    for m, sub in (("MoS2", "MoS ?2"), ("MoSe2", "MoSe ?2"), ("WS2", "WS ?2"), ("WSe2", "WSe ?2")):
        mm = re.search(sub + r"\s*\((\d)[:.](\d{2})\s*eV\)", txt)
        found[m] = f"{mm.group(1)}.{mm.group(2)}" if mm else None
        c = reg.get(f"vH.E.{m}")
        if c is None or found[m] != c["printed"]:
            bad.append(f"{m}: legend {found[m]} vs text {c and c['printed']}")
    res.append(Result("P6", "Fig. 7 legend energies (pdftotext) == registered van Hove energies vH.E.*", "FAIL" if bad else "PASS",
                      measured=", ".join(f"{k} {v}" for k, v in found.items()), detail="; ".join(bad)))
    return res
