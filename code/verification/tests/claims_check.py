"""G: the claims registry (claims.json) against data, recipes, provenance and the paper source.

G1 reload data (sha must match) / data-reading recipes, compare with 'value'            rel 1e-12
G2 check(value, display) == printed == text at the claim's anchor (or inside \\claim{id}{...} once tagged)
G3 every claim anchor found exactly once; no orphan numerals (scanner + allow-list)
G4 fast recompute of computational recipes                                              rel 1e-9 / tol_vs_value
G6 method_param claims == the settings actually used by the producing scripts (regex on the script source)
G7 assertion predicates (human ones are listed for sign-off)
G5 production-grid recompute + digit stability (full mode; needs the C tests)  -- full
G8 two stored field points per production file rerun with identical params      -- full

Before the paper is tagged ("pre-tag mode") the anchors stand in for the \\claim macros.
Also provides: fmt/check helpers, scan_numbers_cli (--scan-numbers), fill_pending (after the v7 queue).
"""
import os, re, json, math, decimal
import numpy as np
from vcore import check, Result, judge, exact, SUITE_DIR, sha256_file
import texscan

WORDS = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                     "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                                     "eighteen", "nineteen", "twenty"])}
DATA_RECIPES = {"vh7_energy", "vh7_slope", "vh7_slope_range", "vh7_eta60", "vh7_dev20", "vh7_below60", "maps7_weight_fraction",
                "maps7_banding", "maps7_banding_others", "decomp7_ratio_range", "edge7_chi", "edge7_sign", "edge7_dev60", "edge7_lin30",
                "edge7_lin30_local", "edge7_omega_window", "fit11", "lda_count_opposite", "all16_opposite", "chiw7_signchange",
                "chiw7_early_count", "chiw7_persist", "lam7_R2", "lam7_slope_sign", "conv_c2", "conv_c3", "c4_change",
                "vh_energy", "vh_eta60", "decomp_v6", "decomp_v6_ratio_range", "decomp_v6_chi_range", "lin_v6_dev60",
                "lin_v6_linear_to", "vh_slope", "vh_slope_range", "lin_v6_dev", "kp_spectra", "kp_sweep_lam_change", "kp_fp_chi", "kp_fp_sign", "kp_fp_order",
                "kp_table_data", "lit", "lit_range"}
TEX_FILES = ["main.tex", "sections/introduction.tex", "sections/theory.tex", "sections/results.tex", "sections/discussion.tex",
             "sections/appendix.tex"]


def registry():
    return json.load(open(os.environ.get("VERIF_CLAIMS") or os.path.join(SUITE_DIR, "claims.json")))


def _round_str(x, dec, rounding="half_even"):
    q = decimal.Decimal(repr(float(x))).quantize(decimal.Decimal(1).scaleb(-dec),
                                                 rounding=decimal.ROUND_HALF_UP if rounding == "half_away" else decimal.ROUND_HALF_EVEN)
    s = f"{q:f}"
    return "0" + s[2:] if False else s


def fmt(value, d):
    """Rendered printed form for plain rounding modes (round/floor/ceil/sci/word/sign)."""
    mode = d.get("mode", "round"); sc = d.get("scale", 1.0); dec = d.get("decimals", 0)
    if value is None:
        return None
    x = value / sc
    if mode == "round":
        s = _round_str(x, dec, d.get("rounding", "half_even"))
        if s.startswith("-") and float(s) == 0:
            s = s[1:]
        if d.get("sign") == "always" and not s.startswith("-"):
            s = "+" + s
        return s
    if mode in ("floor", "ceil"):
        st = d.get("step", 1)
        v = (math.floor if mode == "floor" else math.ceil)(x / st + (1e-12 if mode == "floor" else -1e-12)) * st
        return _round_str(v, dec)
    if mode == "sci":
        e = int(math.floor(math.log10(abs(x))))
        m = _round_str(x / 10 ** e, dec)
        if float(m) >= 10:
            e += 1; m = _round_str(x / 10 ** e, dec)
        return f"{m}e{e}"
    if mode == "word":
        inv = {v: k for k, v in WORDS.items()}
        return inv.get(int(round(x)), str(x))
    if mode == "sign":
        return "+" if x > 0 else "-"
    return None


def check_printed(value, d, printed):
    """(ok, rendered) for every display mode."""
    mode = d.get("mode", "round"); sc = d.get("scale", 1.0)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False, "no value"
    x = value / sc
    if mode == "less_than":
        return x < float(printed), f"{x:.4g} < {printed}"
    if mode == "greater_than":
        return x > float(printed), f"{x:.4g} > {printed}"
    if mode == "at_least_word":
        n = WORDS.get(printed, None) if not printed[0].isdigit() else float(printed)
        return x >= n, f"{x:.4g} >= {n}"
    if mode == "order":
        return int(round(value)) == d["exponent"] and printed == "10", f"order 10^{int(round(value)) if np.isfinite(value) else 'nan'}"
    if mode == "order_of":
        return d["lo"] <= x <= d["hi"] and printed == "10", f"{x:.3g} in [{d['lo']}, {d['hi']}]"
    r = fmt(value, d)
    return r == printed, r


def load_tex(ctx):
    root = ctx.path("paper_root")
    return {f: open(os.path.join(root, f)).read() for f in TEX_FILES if os.path.exists(os.path.join(root, f))}


def locate(claim_id, c, texts):
    """Returns (pos, printed_in_tex, mode) with mode 'tagged' | 'anchor' | error string."""
    t = texts.get(c["tex"]["file"])
    if t is None:
        return None, None, "tex file missing"
    for cid, pr, a, b in texscan.claims_in(t):
        if cid == claim_id:
            return a, pr, "tagged"
    an = c["tex"]["anchor"]
    n = t.count(an)
    if n != 1:
        # the printed digits themselves may have been edited: match the context around them
        off, L = c["tex"]["offset"], len(c["printed"])
        pre, post = an[:off], an[off + L:]
        ms = list(re.finditer(re.escape(pre) + r"(.{1,40}?)" + re.escape(post), t, re.S)) if pre and post else []
        if n == 0 and len(ms) == 1:
            return ms[0].start(1), ms[0].group(1), "anchor-edited"
        return None, None, f"anchor found {n} times (text changed: re-anchor or tag)"
    p = t.find(an) + c["tex"]["offset"]
    return p, t[p:p + len(c["printed"])], "anchor"


# ------------------------------------------------------------------------------------------------ G1
@check("G1", "G", "claims: reload data (sha) and data-reading recipes, compare with registered value")
def g1(ctx):
    """Every data file behind a claim must still have its registered sha (else the value changed without a history record),
    and every data-reading recipe must reproduce the registered value. Claims marked g1='full' (the cloud copy of their data is
    partial) are checked only in full mode."""
    import claims_recipes as CR
    reg = registry()["claims"]
    bad, n, stale, deferred = [], 0, [], []
    for cid, c in reg.items():
        if c.get("g1") == "full" and ctx.mode != "full":
            deferred.append(cid); continue
        src = c.get("source", {})
        files = src.get("data", [])
        files = [files] if isinstance(files, str) else files
        shas = src.get("data_sha") if isinstance(src.get("data_sha"), dict) else {files[0]: src.get("data_sha")} if files else {}
        for f in files:
            key, rel_ = f.split(":", 1)
            p = os.path.join(ctx.path(key), rel_)
            if not os.path.exists(p):
                stale.append(f"{cid}: {p} missing"); continue
            if shas.get(f) and sha256_file(p) != shas[f]:
                bad.append(f"{cid}: data file {rel_} changed (sha) without a history record")
        rcp = c.get("recipe", {})
        if rcp.get("name") in DATA_RECIPES and c.get("value") is not None:
            n += 1
            try:
                v = CR.RECIPES[rcp["name"]](ctx, **rcp.get("args", {}))
            except CR.PendingData as ex:
                stale.append(f"{cid}: {ex}"); continue
            if not (abs(v - c["value"]) <= 1e-12 * max(abs(c["value"]), 1e-300)):
                bad.append(f"{cid}: data gives {v:.10g}, registry {c['value']:.10g}")
        ex = src.get("extract", "")
        if ex.startswith("recipe:") and c.get("value") is not None:
            _, nm, arg = ex.split(":", 2)
            n += 1
            v = CR.RECIPES[nm](ctx, entry=arg)
            if not abs(v - c["value"]) <= 1e-6 * abs(c["value"]):     # stored data vs recompute recipe value (Nk=401)
                bad.append(f"{cid}: stored data {v:.6g} vs registry {c['value']:.6g}")
    st = "FAIL" if bad else ("STALE" if stale else "PASS")
    res = [Result("G1", "registry values == data", st, tol="rel 1e-12 (sha exact)", measured=f"{n} data claims, {len(bad)} bad",
                  detail="; ".join(bad + stale)[:1500])]
    if deferred:
        res.append(Result("G1", "claims whose data are partial in this copy (checked in full mode, on the Mac)", "SKIP",
                          measured=len(deferred), detail=", ".join(deferred)))
    return res


# ------------------------------------------------------------------------------------------------ G2
@check("G2", "G", "claims: value -> printed digits == text in the paper (anchor or \\claim)")
def g2(ctx):
    reg = registry()["claims"]
    texts = load_tex(ctx)
    fails, stale, prov, modes = [], [], [], set()
    npass = 0
    for cid, c in reg.items():
        pos, ptex, mode = locate(cid, c, texts)
        if pos is None:
            stale.append(f"{cid}: {mode}"); continue
        modes.add(mode)
        if ptex != c["printed"]:
            fails.append(f"{cid}: paper shows '{ptex}', registry printed '{c['printed']}'"); continue
        if c["status"] in ("pending_rerun", "pending_full") and c.get("value") is None:
            stale.append(f"{cid}: {c['status']}"); continue
        if c.get("human"):
            continue                                   # human sign-off (listed by G7)
        ok, rend = check_printed(c["value"], c["display"], c["printed"])
        if not ok:
            fails.append(f"{cid}: value {c['value']:.6g} -> '{rend}' but paper prints '{c['printed']}'"
                         + (f" [{c.get('notes')}]" if c.get("notes") else "") + (" (provisional v6 source)" if c["status"] == "provisional" else ""))
        else:
            npass += 1
            if c["status"] == "provisional":
                prov.append(cid)
    res = [Result("G2", "printed digits vs registry values", "FAIL" if fails else "PASS", tol="exact (half-even)",
                  measured=f"{npass} ok, {len(fails)} wrong", detail=f"mode {sorted(modes)}; " + "; ".join(fails)),
           Result("G2", "claims not yet decidable (pending reruns / full mode / drifted anchors)", "STALE" if stale else "PASS",
                  measured=f"{len(stale)}", detail="; ".join(stale)[:2000]),
           Result("G2", "claims passing only against v6 (pre-fix) data", "WARN" if prov else "PASS", measured=f"{len(prov)}",
                  detail="re-point to the v7 successor when the queue lands: " + ", ".join(prov))]
    return res


# ------------------------------------------------------------------------------------------------ G3
def _allow():
    p = os.path.join(SUITE_DIR, "numbers_allowlist.json")
    return json.load(open(p))["entries"] if os.path.exists(p) else []


def orphans(ctx, texts=None, allow=None):
    reg = registry()["claims"]
    texts = texts or load_tex(ctx)
    covered = {f: [] for f in texts}
    for cid, c in reg.items():
        pos, ptex, mode = locate(cid, c, texts)
        if pos is not None:
            ln = len(ptex) if mode == "anchor" else len(texts[c["tex"]["file"]][pos:].split("}", 2)[0]) + 200
            if mode == "tagged":
                for cid2, pr, a, b in texscan.claims_in(texts[c["tex"]["file"]]):
                    if cid2 == cid:
                        covered[c["tex"]["file"]].append((a, b))
            else:
                covered[c["tex"]["file"]].append((pos, pos + ln))
    allow = allow if allow is not None else _allow()
    out = []
    for f, t in texts.items():
        for h in texscan.scan(t):
            if h["exempt"]:
                continue
            if any(a <= h["start"] < b for a, b in covered[f]):
                continue
            win = t[max(0, h["start"] - 80):h["end"] + 80]
            al = [e for e in allow if e["file"] == f and e["value"] == h["value"] and e["context"] in win]
            out.append(dict(file=f, line=texscan.line_of(t, h["start"]), value=h["value"], context=h["context"],
                            allow=al[0] if al else None, start=h["start"], inmath=h["inmath"]))
    return out


@check("G3", "G", "claims: anchors unique; no orphan numerals in the paper")
def g3(ctx):
    texts = load_tex(ctx)
    orph = orphans(ctx, texts)
    hard = [o for o in orph if o["allow"] is None]
    unapproved = [o for o in orph if o["allow"] is not None and not o["allow"].get("approved")]
    approved = [o for o in orph if o["allow"] is not None and o["allow"].get("approved")]
    res = [Result("G3", "untagged numerals that are neither claims nor allow-listed", "FAIL" if hard else "PASS",
                  tol="0", measured=len(hard), detail="; ".join(f"{o['file']}:{o['line']} '{o['value']}' ...{o['context'][25:75]}..." for o in hard[:30])),
           Result("G3", "allow-list entries awaiting Abir's approval (proposed by --scan-numbers)", "WARN" if unapproved else "PASS",
                  measured=f"{len(unapproved)} proposed, {len(approved)} approved")]
    return res


# ------------------------------------------------------------------------------------------------ G4
@check("G4", "G", "claims: fast recompute of computational recipes (reference / literature / kp2)")
def g4(ctx):
    import claims_recipes as CR
    reg = registry()["claims"]
    bad, n, skipped = [], 0, []
    for cid, c in reg.items():
        rcp = c.get("recipe")
        if not rcp or rcp["name"] in DATA_RECIPES or c.get("value") is None:
            continue
        fast = c.get("recompute", {}).get("fast", {})
        if fast.get("skip"):
            skipped.append(cid); continue
        args = dict(rcp.get("args", {}), **fast.get("args", {}))
        try:
            v = CR.RECIPES[rcp["name"]](ctx, **args)
        except CR.PendingData as ex:
            skipped.append(f"{cid} ({ex})"); continue
        n += 1
        tol = fast.get("tol_vs_value", fast.get("tol_rel", 1e-9))
        if not (abs(v - c["value"]) <= tol * max(abs(c["value"]), 1e-300)) or np.sign(v) != np.sign(c["value"]):
            bad.append(f"{cid}: recompute {v:.8g} vs registry {c['value']:.8g} (tol {tol:g})")
    res = [Result("G4", "fast recompute", "FAIL" if bad else "PASS", tol="rel 1e-9 (or tol_vs_value)", measured=f"{n} recomputed, {len(bad)} bad",
                  detail="; ".join(bad))]
    if skipped:
        res.append(Result("G4", "expensive recipes (engine / kp2 at production grids): recomputed in full mode (G5)", "INFO",
                          measured=len(skipped), detail=", ".join(skipped)))
    return res


# ------------------------------------------------------------------------------------------------ G6
@check("G6", "G", "claims: method parameters printed in the paper == settings of the producing scripts")
def g6(ctx):
    reg = registry()["claims"]
    bad, ok, nochk = [], 0, []
    for cid, c in reg.items():
        if c["kind"] != "method_param":
            continue
        chk = c.get("check")
        if not chk:
            nochk.append(cid); continue
        if ":" in chk["file"]:                         # 'pathkey:file' for scripts outside code_root (v6 figure code)
            k_, r_ = chk["file"].split(":", 1)
            p = os.path.join(ctx.path(k_), r_)
        else:
            try:
                import prov
                p = prov.resolve_script(ctx.cfg, ctx.code_root, chk["file"])
            except Exception:
                p = os.path.join(ctx.code_root, chk["file"])
        if not os.path.exists(p):
            bad.append(f"{cid}: {chk['file']} missing"); continue
        mm = None
        for alt in chk["regex"].split("|") if "(" in chk["regex"] and chk["regex"].count("(") > 1 and "|" in chk["regex"] else [chk["regex"]]:
            mm = re.search(alt, open(p).read())
            if mm and mm.groups() and mm.group(1):
                break
        if not mm or not mm.groups() or mm.group(1) is None:
            bad.append(f"{cid}: setting not found in {chk['file']} (regex /{chk['regex']}/)"); continue
        actual = float(mm.group(1)) * chk.get("scale", 1.0)
        rel_ = chk.get("relation", "==")
        good = abs(actual - c["value"]) <= 1e-12 * max(1, abs(c["value"])) if rel_ == "==" else actual <= c["value"]
        if good:
            ok += 1
        else:
            bad.append(f"{cid}: paper '{c['printed']}' (= {c['value']:g}) but {chk['file']} uses {actual:g}" +
                       (f" [{c['notes']}]" if c.get("notes") else ""))
    return [Result("G6", "printed method parameters vs producing scripts", "FAIL" if bad else "PASS", tol="exact",
                   measured=f"{ok} ok, {len(bad)} differ", detail="; ".join(bad)),
            Result("G6", "method parameters without a machine check (human sign-off)", "INFO", measured=len(nochk), detail=", ".join(nochk))]


# ------------------------------------------------------------------------------------------------ G7
@check("G7", "G", "claims: assertion predicates hold; human assertions listed")
def g7(ctx):
    reg = registry()["claims"]
    bad, ok, human, pend = [], 0, [], []
    for cid, c in reg.items():
        if c["kind"] not in ("assertion", "count"):
            continue
        if c.get("human"):
            human.append(cid); continue
        if c.get("value") is None:
            pend.append(cid); continue
        good, rend = check_printed(c["value"], c["display"], c["printed"])
        if good:
            ok += 1
        else:
            bad.append(f"{cid}: {rend} fails for printed '{c['printed']}'")
    return [Result("G7", "assertion predicates", "FAIL" if bad else "PASS", measured=f"{ok} hold, {len(bad)} fail, {len(pend)} pending",
                   detail="; ".join(bad)),
            Result("G7", "assertions needing a human reading of a figure", "INFO", measured=len(human), detail=", ".join(human))]


@check("G5", "G", "claims: production-grid recompute and digit stability", modes=("full",))
def g5(ctx):
    import claims_recipes as CR
    reg = registry()["claims"]
    bad, n = [], 0
    for cid, c in reg.items():
        rcp = c.get("recipe")
        if not rcp or rcp["name"] in DATA_RECIPES or c.get("value") is None:
            continue
        full = c.get("recompute", {}).get("full", {})
        try:
            v = CR.RECIPES[rcp["name"]](ctx, **dict(rcp.get("args", {}), **full.get("args", {})))
        except CR.PendingData:
            continue
        n += 1
        if abs(v - c["value"]) > full.get("tol_rel", 3e-3) * abs(c["value"]):
            bad.append(f"{cid}: {v:.6g} vs {c['value']:.6g}")
    pend = [cid for cid, c in reg.items() if c["status"] in ("pending_rerun", "pending_full") and c.get("value") is None]
    return [Result("G5", "full recompute of recipe claims", "FAIL" if bad else "PASS", measured=f"{n} recomputed", detail="; ".join(bad)),
            Result("G5", "production-grid claims (edge/maps/chiw/lam/decomp) and digit stability", "SKIP" if pend else "PASS",
                   measured=f"{len(pend)} claims", detail="needs the v7 outputs + C1-C7 (fill_pending, then rerun --full)")]


# ------------------------------------------------------------------------------------------------ scan-numbers
def _guess_class(o):
    v, ctxs = o["value"], o["context"]
    if not v[0].isdigit():
        return "count-of-objects", "count word"
    if re.search(r"(B|B_z|T|b|\\kappa|\\lambda|\\varepsilon|m|\\tau|p|q)\s*\\?(=|\\neq|\\ge|>|\\to|\\approx)\s*" + re.escape(v), ctxs) or v in ("0",):
        return "definition", "limit/condition (B=0, T=0, ...)"
    if "indep" in ctxs:
        return "count-of-objects", "number of independent components"
    if re.search(r"2D|2H", ctxs):
        return "definition", "dimension / polytype label"
    if "\\pm" in ctxs or "\\{0" in ctxs or "L}_z" in ctxs:
        return "definition", "orbital quantum numbers"
    if re.search(r"tau_r|mW|\\mu\\mathrm\{m\}|\\mu\$m|ps|\\chi=10", ctxs):
        return "illustrative", "input of the order-of-magnitude current estimate"
    return "illustrative", "REVIEW"


def scan_numbers_cli(cfg):
    from vcore import Ctx
    code_root = cfg["paths"]["code_root"] if os.path.isabs(cfg["paths"]["code_root"]) else os.path.join(SUITE_DIR, cfg["paths"]["code_root"])
    ctx = Ctx(cfg, "fast", code_root, "/tmp")
    texts = load_tex(ctx)
    old_all = _allow()
    orph = orphans(ctx, texts, old_all)
    used = {id(o["allow"]) for o in orph if o["allow"] is not None}
    old = [e for e in old_all if id(e) in used]          # minimal list: drop entries that no longer match an orphan numeral
    for e in old:
        e.setdefault("status", "approved" if e.get("approved") else "proposed")
    new = []
    for o in orph:
        if o["allow"] is not None:
            continue
        t = texts[o["file"]]
        a = o["start"]
        w = 14
        while True:
            c = t[max(0, a - w):a + len(o["value"]) + w]
            if t.count(c) == 1 or w > 200:
                break
            w += 6
        cls, why = _guess_class(o)
        new.append(dict(file=o["file"], value=o["value"], context=c if len(c) >= 25 else t[max(0, a - 20):a + 20], cls=cls))
        new[-1] = dict(file=o["file"], value=o["value"], context=new[-1]["context"], **{"class": cls}, reason=why, approved=False,
                       status="proposed")
    data = dict(_comment="Numbers in the paper that are not claims (protocol sec. 5 allow-list). Proposed by verify_all.py "
                         "--scan-numbers (status 'proposed' for Abir); set approved=true and status='approved' after reading each one. Inputs, numerical settings and data counts "
                         "may NOT be listed here: they must be claims.", entries=old + new)
    json.dump(data, open(os.path.join(SUITE_DIR, "numbers_allowlist.json"), "w"), indent=1, ensure_ascii=False)
    print(f"numbers_allowlist.json: {len(old)} kept, {len(old_all) - len(old)} dropped (no longer in the text), {len(new)} proposed")
    for e in new:
        print(f"  {e['file']}: {e['value']:>6s} [{e['class']}] {e['context']!r}")
    return 0
