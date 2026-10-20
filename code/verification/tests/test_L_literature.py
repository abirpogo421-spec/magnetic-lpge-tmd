"""L: literature inputs (transcription and interpretation, not truth).

L1 tmd_params.py == independent Liu transcription (literature/liu_tables.json, from the reference) == ref.params, exact.
L2 Gamma/K/M energies: engine NN == Liu Table I closed forms (1e-12); engine TNN == reference H3 (1e-12).
   Liu prints no TNN band energies in a table we hold, so the protocol's "TNN vs Liu printed values (1 meV)" is SKIP.
L3 valence spin splitting at K = 2 lambda (lambda from the literature table, not from the engine).
L4 Kormanyos: spin-label predicate, A-pair mass predicates against TB, script KORM == literature table.
L5 2 m_D - lambda = E_A in the k.p output (stored kp_fp_all_v7.json and the S6 smoke output).
"""
import os, json, ast
import numpy as np
from vcore import check, Result, judge, exact, SUITE_DIR

MAT4 = ["MoS2", "MoSe2", "WS2", "WSe2"]
MAT6 = ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]


def _lit(name):
    return json.load(open(os.path.join(SUITE_DIR, "literature", name)))


@check("L1", "L", "tmd_params == Liu transcription (24 sets + lambda), exact")
def l1(ctx):
    tp, ref = ctx.eng.tp, ctx.ref
    liu = _lit("liu_tables.json")
    bad = []
    nn_keys = liu["NN_keys"]
    for f in ("GGA", "LDA"):
        for m in MAT6:
            lit_nn = dict(zip(nn_keys, liu["NN"][f][m])); lit_tnn = dict(zip(liu["TNN_keys"], liu["TNN"][f][m]))
            for model in ("NN", "TNN"):
                p = tp.get_params(m, f, model)
                q = ref.params(m, model, f)
                exp = dict(lit_nn if model == "NN" else lit_tnn); exp["a"] = lit_nn["a"]; exp.pop("zXX", None)
                for k, v in exp.items():
                    if p.get(k) != v:
                        bad.append(f"{m}/{f}/{model}/{k}: tmd_params {p.get(k)} vs Liu {v}")
                    if q.get(k) != v:
                        bad.append(f"{m}/{f}/{model}/{k}: refimpl {q.get(k)} vs literature file {v}")
                if p["lambda"] != liu["lambda_eV"][m]:
                    bad.append(f"{m}: lambda {p['lambda']} vs {liu['lambda_eV'][m]}")
    return exact("L1", "Liu parameter transcription", not bad, measured=f"{len(bad)} mismatches", detail="; ".join(bad[:8]))


def _nn_closed(p):
    """Liu Table I closed forms for the NN model at Gamma, K, M."""
    e1, e2, t0, t1, t2, t11, t12, t22 = (p[x] for x in ("eps1", "eps2", "t0", "t1", "t2", "t11", "t12", "t22"))
    G = sorted([e1 + 6 * t0, e2 + 3 * (t11 + t22), e2 + 3 * (t11 + t22)])
    Kk = sorted([e2 - 1.5 * (t11 + t22) - 3 * np.sqrt(3) * t12, e1 - 3 * t0, e2 - 1.5 * (t11 + t22) + 3 * np.sqrt(3) * t12])
    f1 = 0.5 * (e1 + e2) - t0 - 1.5 * t11 + 0.5 * t22
    f2 = 0.5 * np.sqrt((e1 - e2 - 2 * t0 + 3 * t11 - t22) ** 2 + 64 * t2 ** 2)
    M = sorted([f1 - f2, e2 + t11 - 3 * t22, f1 + f2])
    return G, Kk, M


@check("L2", "L", "band energies at Gamma, K, M (NN closed forms; TNN vs reference)")
def l2(ctx):
    eng, ref = ctx.eng, ctx.ref
    pts = {"G": (0.0, 0.0), "K": (4 * np.pi / 3, 0.0), "M": (np.pi, np.pi / np.sqrt(3))}
    dnn = dtnn = 0.0
    for f in ("GGA", "LDA"):
        for m in MAT6:
            closed = dict(zip("GKM", _nn_closed(ref.params(m, "NN", f))))
            for nm, (x, y) in pts.items():
                eng.tb.set_config(m, f)
                H = eng.tb.make_H3("NN", np.array([[x / 2]]), np.array([[np.sqrt(3) * y / 2]]))[0, 0]
                dnn = max(dnn, np.max(np.abs(np.linalg.eigvalsh(H) - closed[nm])))
                Ht = eng.tb.make_H3("TNN", np.array([[x / 2]]), np.array([[np.sqrt(3) * y / 2]]))[0, 0]
                Er = np.linalg.eigvalsh(ref.H3(x, y, m, "TNN", f))
                dtnn = max(dtnn, np.max(np.abs(np.linalg.eigvalsh(Ht) - Er)))
    return [judge("L2", "NN energies at Gamma/K/M vs Liu Table I closed forms (12 sets)", dnn, 1e-12),
            judge("L2", "TNN energies at Gamma/K/M vs reference H3 (12 sets)", dtnn, 1e-12),
            Result("L2", "TNN energies vs Liu's printed values (1 meV)", "SKIP", tol="1e-3 eV",
                   detail="no transcription of printed TNN band energies held in literature/")]


@check("L3", "L", "valence spin splitting at K = 2 lambda (literature lambda)")
def l3(ctx):
    eng = ctx.eng
    liu = _lit("liu_tables.json")
    worst = 0.0; det = ""
    for f in ("GGA", "LDA"):
        for m in MAT6:
            for model in ("NN", "TNN"):
                up, dn = eng.H_blocks(m, model, f, [4 * np.pi / 3], [0.0], 0.0)
                d = abs((np.linalg.eigvalsh(up[0])[0] - np.linalg.eigvalsh(dn[0])[0]) - 2 * liu["lambda_eV"][m])
                if d > worst:
                    worst, det = d, f"{m}/{f}/{model}"
    return judge("L3", "E_v,up(K) - E_v,dn(K) = 2 lambda", worst, 1e-12, detail=f"worst {det}")


def tb_masses(ref, m, s=+1, h=1e-3):
    """Band-edge masses at K of spin block s (TB, B=0) from radial curvature (average of x and y), in m_e."""
    p = ref.params(m, "TNN", "GGA")
    a = p["a"] * 1e-10
    ka = 9.1093837015e-31 * a * a / 1.054571817e-34 ** 2 * 1.602176634e-19      # m_e a^2/hbar^2 [1/eV]
    curv = 0
    for ex, ey in ((1, 0), (0, 1)):
        Es = [np.linalg.eigvalsh(ref.H_block(ref.K_POINT[0] + d * ex, d * ey, s, 0.0, p=p)) for d in (-h, 0, h)]
        curv = curv + (Es[0] - 2 * Es[1] + Es[2]) / h ** 2 / 2
    # E = (hbar^2/2 m* ) q^2  ->  m*/m_e = 1/(ka * d2E/dq2) with q in 1/a
    return 1 / (ka * curv[1]), 1 / (ka * curv[0])        # (m_c, m_v)


@check("L4", "L", "Kormanyos inputs: spin label, A-pair masses, script KORM == literature")
def l4(ctx):
    ref = ctx.ref
    K = _lit("kormanyos_2015.json"); it = K["interpretation"]; t6 = K["table6"]["rows"]
    out = []
    # L4a spin label
    q = K["spin_label_quote"]["text"]
    upper_label = "↓" if "upper index (1) ((2)) is equivalent to ↓ (↑)" in q else None
    lab = t6[it["kappa_column"]]["spin_label"]
    out.append(exact("L4", "a: chosen kappa column label == label of the upper VB (1) in the p.20 quote", lab == upper_label,
                     measured=f"column {lab}, quote {upper_label}",
                     detail="TB convention: upper VB at K is the s=+1 block; Kormanyos label it ↓ (opposite valley/spin convention); the physical pair is the band-edge (A) pair in both"))
    # L4b/c masses
    bad, info = [], []
    for i, m in enumerate(K["materials"]):
        mcA = K["table3"]["rows"][it["conduction_mass_row"][m]][i]; mvA = K["table4"]["rows"][it["valence_mass_row"]][i]
        other_c = [r for r in K["table3"]["rows"] if r.startswith("mcb") and r != it["conduction_mass_row"][m]][0]
        mcB = K["table3"]["rows"][other_c][i]; mvB = K["table4"]["rows"]["mvb(2)/me (HSE,LDA)"][i]
        tcA, tvA = tb_masses(ref, m, +1); tcB, tvB = tb_masses(ref, m, -1)
        okA = (mcA > 0 and mvA < 0 and tcA > 0 and tvA < 0 and (abs(mvA) > mcA) == (abs(tvA) > tcA)
               and max(mcA / tcA, tcA / mcA, mvA / tvA, tvA / mvA) <= 1.6)
        eA = abs(mcA / tcA - 1) + abs(mvA / tvA - 1); eB = abs(mcB / tcA - 1) + abs(mvB / tvA - 1)
        info.append(f"{m}: TB(A) mc={tcA:.3f} mv={tvA:.3f}; Korm A ({mcA},{mvA}) err {eA:.2f}, B-pair ({mcB},{mvB}) err {eB:.2f}")
        if not okA:
            bad.append(f"{m}: A-pair mass predicate fails")
        if not eA < eB:
            bad.append(f"{m}: B-pair fits TB active pair better than the A pair")
    out.append(exact("L4", "b,c: A-pair masses agree with TB (sign, ordering, x1.6) and beat the B pair", not bad,
                     detail="; ".join(bad + info)))
    # L4d script KORM (AST) == literature
    p = os.path.join(ctx.code_root, "kp", "kp_fp_all_v7.py")
    tree = ast.parse(open(p).read())
    korm = None
    for st in tree.body:
        if isinstance(st, ast.Assign) and getattr(st.targets[0], "id", "") == "KORM":
            korm = ast.literal_eval(st.value)
    liu = _lit("liu_tables.json")
    bad = []
    for i, m in enumerate(K["materials"]):
        EA = t6["Ebg,K [eV]"]["DFT"][i] + max(0.0, -K["table3"]["rows"]["2Dcb [meV] (HSE,LDA)"][i] / 1000)
        exp = (EA, t6["|gamma| [eV A]"]["DFT"][i], t6[it["kappa_column"]]["DFT"][i],
               K["table3"]["rows"][it["conduction_mass_row"][m]][i], K["table4"]["rows"][it["valence_mass_row"]][i],
               liu["lambda_eV"][m])
        got = korm[m]
        for nm, a, b in zip(("E_A", "gamma", "kappa", "m_c", "m_v", "lambda"), got, exp):
            if abs(a - b) > 5e-4:
                bad.append(f"{m}.{nm}: script {a} vs literature {b:.4g}")
    out.append(exact("L4", "d: kp_fp_all_v7.KORM == literature table (A pair)", not bad, detail="; ".join(bad)))
    return out


def _l5_check(name, data):
    worst = 0.0; det = []
    for m in MAT4:
        pp = data[m]["params"]
        d = abs(2 * pp["mD"] - pp["lam"] - data[m]["E_A"])
        worst = max(worst, d); det.append(f"{m}: 2mD-lam={2*pp['mD']-pp['lam']:.4f} E_A={data[m]['E_A']}")
    return judge("L5", name, worst, 1e-12, detail="; ".join(det))


@check("L5", "L", "k.p gap convention 2 m_D - lambda = E_A")
def l5(ctx):
    out = []
    p = os.path.join(ctx.path("kp_out"), "kp_fp_all_v7.json")
    if os.path.exists(p):
        out.append(_l5_check("stored kp_fp_all_v7.json", json.load(open(p))))
    else:
        out.append(Result("L5", "stored kp_fp_all_v7.json", "SKIP", detail="file not found"))
    from test_S_smoke import smoke_kp_fp
    sm = smoke_kp_fp(ctx)
    if sm.get("error"):
        out.append(Result("L5", "smoke kp_fp_all_v7 output", "FAIL", detail=sm["error"]))
    else:
        out.append(_l5_check("smoke kp_fp_all_v7 output (current code)", sm["json"]))
    return out
