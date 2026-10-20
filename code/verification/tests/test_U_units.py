"""U: units and constants.

U1 si_factor = e^3 a/hbar^2, kappa_orb = m_e a^2/hbar^2 (1/eV), psm = e a^2/hbar, from CODATA 2018 + Liu a.
U2 mu_B in every file vs CODATA 2018 (e hbar / 2 m_e in eV/T); the CODATA-2014 value of the engine is reported.
U3 gap(K') - gap(K) = 4 mu_B B at the Hamiltonian level, B = 10 T, four materials.
"""
import os, re, ast
import numpy as np
from vcore import check, Result, judge, exact, rel

E = 1.602176634e-19; HBAR = 1.054571817e-34; ME = 9.1093837015e-31   # CODATA 2018 (hbar as tabulated, 10 digits)
MUB_2018 = 5.7883818060e-5                 # CODATA 2018 recommended mu_B/e in eV/T
MAT4 = ["MoS2", "MoSe2", "WS2", "WSe2"]


@check("U1", "U", "unit conversion factors from CODATA (si, kappa_orb, psm), all materials and functionals")
def u1(ctx):
    eng, ref = ctx.eng, ctx.ref
    worst = 0.0; det = []
    for m in MAT4 + ["MoTe2", "WTe2"]:
        for f in ("GGA", "LDA"):
            a = ref.params(m, "NN", f)["a"] * 1e-10
            eng.tb.set_config(m, f)
            ka, psm = eng.tb.material_constants()
            r = eng.L.eta(m, "NN", None, 0.0, np.array([2.0]), 0.01, dict(spin=True),
                          functional=f, kpts=(np.array([1.0]), np.array([0.2]), 1e-4))
            d = [abs(r["si_factor"] - E ** 3 * a / HBAR ** 2) / (E ** 3 * a / HBAR ** 2),
                 abs(ka - ME * a * a / HBAR ** 2 * E) / (ME * a * a / HBAR ** 2 * E),
                 abs(psm - E * a * a / HBAR) / (E * a * a / HBAR)]
            if max(d) > worst:
                worst = max(d); det = [m, f] + [f"{x:.1e}" for x in d]
    return judge("U1", "si, kappa_orb, psm vs CODATA", worst, 1e-12, detail=f"worst {det} (si, ka, psm)")


@check("U2", "U", "mu_B per file vs CODATA 2018")
def u2(ctx):
    vals = {}
    for rp in ["pub_pipeline/tb_core.py", "kp/kp_fp_all_v7.py", "pub_pipeline/lpge_v6.py"]:
        p = os.path.join(ctx.code_root, rp)
        if not os.path.exists(p):
            continue
        for mm in re.finditer(r"\b(MU_B|MUB|mu_B)\s*=\s*([0-9.eE+-]+)", open(p).read()):
            vals[f"{rp}:{mm.group(1)}"] = float(mm.group(2))
    vals["ref/refimpl.py:MU_B"] = ctx.ref.MU_B
    worst = max(abs(v - MUB_2018) / MUB_2018 for v in vals.values())
    det = ", ".join(f"{k}={v:.10e} ({(v - MUB_2018) / MUB_2018:+.1e})" for k, v in vals.items())
    r = judge("U2", "mu_B values", worst, 1e-9, detail="CODATA2018=%.10e; " % MUB_2018 + det)
    if r.status == "PASS" and worst > 1e-12:
        r.detail = "PASS within tol; engine uses CODATA 2014 (finding 9.5): " + r.detail
    return r


@check("U3", "U", "gap(K') - gap(K) = 4 mu_B B at the H level (B = 10 T)")
def u3(ctx):
    eng = ctx.eng
    B = 10.0; out = []; worst = 0.0
    for m in MAT4:
        for model in ("TNN", "NN"):
            K = np.array([4 * np.pi / 3]); z = np.array([0.0])
            up, dn = eng.H_blocks(m, model, "GGA", K, z, B)
            upm, dnm = eng.H_blocks(m, model, "GGA", -K, z, B)
            eK = np.linalg.eigvalsh(up[0]); eKp = np.linalg.eigvalsh(dnm[0])     # active blocks: up at K, dn at K'
            gK, gKp = eK[1] - eK[0], eKp[1] - eKp[0]
            d = abs((gKp - gK) - 4 * MUB_2018 * B) / (4 * MUB_2018 * B)
            worst = max(worst, d); out.append(f"{m}/{model}: {1e3*(gKp-gK):.6f} meV")
    return judge("U3", "valley gap splitting 4 mu_B B", worst, 1e-9, detail="; ".join(out[:4]) + f" (4 mu_B B = {4e3*MUB_2018*B:.6f} meV)")


@check("U4", "U", "absolute k-space measure: disk dA = h^2, BZ grid sum dA = 8 pi^2/sqrt3, disk point sets == reference")
def u4(ctx):
    eng, ref = ctx.eng, ctx.ref
    out = []
    bad = []
    for R, h in ((0.3, 0.01), (0.9, 0.03), (0.9, 0.002), (0.25, 0.0025), (0.004, 0.0005)):
        kx, ky, dA = eng.L.local_kpts(R, h)
        if dA != h * h:
            bad.append(f"local_kpts({R},{h}) dA = {dA!r} != h^2 = {h*h!r}")
    out.append(exact("U4", "local_kpts area element == h^2 exactly", not bad, detail="; ".join(bad)))
    worst = 0.0
    for N in (24, 60, 600, 900):
        al, be, Ji, dA = eng.L.grid(N)
        worst = max(worst, abs(dA * N * N - 8 * np.pi ** 2 / np.sqrt(3)) / (8 * np.pi ** 2 / np.sqrt(3)))
    out.append(judge("U4", "engine full-BZ grid: N^2 dA == BZ area 8 pi^2/sqrt3 (1/a^2)", worst, 1e-14,
                     detail=f"reference BZ_AREA = {ref.BZ_AREA:.15g}"))
    bad = []
    for R, h in ((0.3, 0.01), (0.9, 0.03), (0.3, 0.03), (0.9, 0.01)):
        kx, ky, _ = eng.L.local_kpts(R, h)
        k, _ = ref.disk_grid(R, h)
        e = {(int(round(x / h * 1000)), int(round(y / h * 1000))) for x, y in zip(np.round(kx, 12), np.round(ky, 12))}
        r = {(int(round(x / h * 1000)), int(round(y / h * 1000))) for x, y in zip(np.round(k[:, 0], 12), np.round(k[:, 1], 12))}
        if e != r:
            bad.append(f"R={R} h={h}: engine {len(e)} pts, reference {len(r)}, symmetric difference {len(e ^ r)}")
    out.append(exact("U4", "engine disk point sets == reference disk_grid (K and K' = -K disk)", not bad, detail="; ".join(bad)))
    return out
