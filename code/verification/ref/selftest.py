"""
selftest.py -- internal checks of refimpl.py that need no trust in the code.

Run:  python3 selftest.py            (about 1-2 minutes on one core)
Each check prints PASS/FAIL with the measured deviation and the tolerance that
was fixed in advance.  Lines marked INFO are reported values (not pass/fail),
compared where possible with numbers quoted in the paper.
"""
import sys
import time
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refimpl as R

RESULTS = []
RNG = np.random.default_rng(20261014)
MAT4 = ["MoS2", "MoSe2", "WS2", "WSe2"]


PAPER = []


def check(name, value, tol, detail=""):
    ok = bool(np.all(np.isfinite(value)) and value <= tol)
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: dev={float(value):.3e} (tol {tol:.1e}) {detail}")
    return ok


def paper_check(name, value, tol, detail=""):
    """Comparison of a reference-code number with a number printed in the paper.
    A miss is reported as DISCREPANCY (a finding about the paper or its inputs),
    not as a failure of the reference code."""
    ok = bool(np.all(np.isfinite(value)) and value <= tol)
    PAPER.append((name, ok))
    print(f"[{'AGREE' if ok else 'DISCREPANCY'}] {name}: dev={float(value):.3e} (tol {tol:.1e}) {detail}")
    return ok


def info(name, msg):
    print(f"[INFO] {name}: {msg}")


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def rand_k(n=200):
    return RNG.uniform(-6, 6, size=(n, 2))


# ---------------------------------------------------------------------------
# Independent literal transcription of the Liu / paper-appendix TNN formulas
# (factored form exactly as printed), to catch errors in the expanded term list
# ---------------------------------------------------------------------------
def H3_literal(kx, ky, p):
    s3 = np.sqrt(3.0)
    al, be = kx / 2, s3 * ky / 2
    c, s = np.cos, np.sin
    e1, e2 = p["eps1"], p["eps2"]
    t0, t1, t2, t11, t12, t22 = (p[k] for k in ("t0", "t1", "t2", "t11", "t12", "t22"))
    r0, r1, r2, r11, r12 = (p[k] for k in ("r0", "r1", "r2", "r11", "r12"))
    u0, u1, u2, u11, u12, u22 = (p[k] for k in ("u0", "u1", "u2", "u11", "u12", "u22"))
    V0 = (e1 + 2 * t0 * (2 * c(al) * c(be) + c(2 * al)) + 2 * r0 * (2 * c(3 * al) * c(be) + c(2 * be))
          + 2 * u0 * (2 * c(2 * al) * c(2 * be) + c(4 * al)))
    ReV1 = (-2 * s3 * t2 * s(al) * s(be) + 2 * (r1 + r2) * s(3 * al) * s(be)
            - 2 * s3 * u2 * s(2 * al) * s(2 * be))
    ImV1 = (2 * t1 * s(al) * (2 * c(al) + c(be)) + 2 * (r1 - r2) * s(3 * al) * c(be)
            + 2 * u1 * s(2 * al) * (2 * c(2 * al) + c(2 * be)))
    ReV2 = (2 * t2 * (c(2 * al) - c(al) * c(be)) - 2 / s3 * (r1 + r2) * (c(3 * al) * c(be) - c(2 * be))
            + 2 * u2 * (c(4 * al) - c(2 * al) * c(2 * be)))
    ImV2 = (2 * s3 * t1 * c(al) * s(be) + 2 / s3 * s(be) * (r1 - r2) * (c(3 * al) + 2 * c(be))
            + 2 * s3 * u1 * c(2 * al) * s(2 * be))
    V11 = (e2 + (t11 + 3 * t22) * c(al) * c(be) + 2 * t11 * c(2 * al) + 4 * r11 * c(3 * al) * c(be)
           + 2 * (r11 + s3 * r12) * c(2 * be) + (u11 + 3 * u22) * c(2 * al) * c(2 * be) + 2 * u11 * c(4 * al))
    ReV12 = (s3 * (t22 - t11) * s(al) * s(be) + 4 * r12 * s(3 * al) * s(be)
             + s3 * (u22 - u11) * s(2 * al) * s(2 * be))
    ImV12 = 4 * t12 * s(al) * (c(al) - c(be)) + 4 * u12 * s(2 * al) * (c(2 * al) - c(2 * be))
    V22 = (e2 + (3 * t11 + t22) * c(al) * c(be) + 2 * t22 * c(2 * al)
           + 2 * r11 * (2 * c(3 * al) * c(be) + c(2 * be))
           + 2 / s3 * r12 * (4 * c(3 * al) * c(be) - c(2 * be))
           + (3 * u11 + u22) * c(2 * al) * c(2 * be) + 2 * u22 * c(4 * al))
    V1, V2, V12 = ReV1 + 1j * ImV1, ReV2 + 1j * ImV2, ReV12 + 1j * ImV12
    H = np.zeros(np.shape(kx) + (3, 3), complex)
    H[..., 0, 0], H[..., 0, 1], H[..., 0, 2] = V0, V1, V2
    H[..., 1, 0], H[..., 1, 1], H[..., 1, 2] = np.conj(V1), V11, V12
    H[..., 2, 0], H[..., 2, 1], H[..., 2, 2] = np.conj(V2), np.conj(V12), V22
    return H


def sect(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def main():
    t_start = time.time()
    allp = [R.params(m, mo, f) for m in R.MATERIALS for mo in ("NN", "TNN") for f in ("GGA", "LDA")]

    # -------------------------------------------------------------- constants
    sect("1. Constants and unit conversions")
    muB = R.E_CHARGE * R.HBAR / (2 * R.M_E) / R.E_CHARGE
    check("mu_B = e hbar/2m_e (eV/T) vs 5.7883818060e-5", abs(muB - R.MU_B) / R.MU_B, 1e-9)
    vals = [R.M_E * (R.params(m)["a"] * R.ANG) ** 2 / R.HBAR ** 2 * R.E_CHARGE for m in MAT4]
    info("m_e a^2/hbar^2 (1/eV), GGA, MoS2/MoSe2/WS2/WSe2", np.round(vals, 3).tolist()
         + ["paper: 1.3-1.5"])
    paper_check("m_e a^2/hbar^2 inside paper's 1.3-1.5 1/eV", max(0, 1.3 - min(vals), max(vals) - 1.5), 1e-12)
    B80 = 0.080 / R.MU_B
    info("mu_B B = 80 meV corresponds to B (T)", f"{B80:.0f}  (paper: ~1.4e3 T)")
    info("prefactor -2 pi e^3 a/hbar^2, MoS2 (A m V^-2 s^-1)", f"{R.prefactor_injection(3.19):.6e}")
    info("hbar/e (T m^2) used in D = 1 + B Omega a^2 /(hbar/e)", f"{R.HBAR / R.E_CHARGE:.6e}")

    # ------------------------------------------------------------- Hamiltonian
    sect("2. Hamiltonian: transcription, Hermiticity, symmetries, derivatives")
    k = rand_k(300)
    dev = max(rel(R.H3(k[:, 0], k[:, 1], p=p), H3_literal(k[:, 0], k[:, 1], p)) for p in allp)
    check("term-list H3 == literal factored Liu/appendix formulas (24 parameter sets)", dev, 1e-13)
    dev = 0
    for p in allp:
        H = R.H3(k[:, 0], k[:, 1], p=p)
        dev = max(dev, np.max(np.abs(H - np.conj(np.swapaxes(H, -1, -2)))))
        for s in (1, -1):
            Hb = R.H_block(k[:, 0], k[:, 1], s, 37.0, p=p)
            dev = max(dev, np.max(np.abs(Hb - np.conj(np.swapaxes(Hb, -1, -2)))))
    check("Hermiticity of H3 and spin blocks (B=37 T)", dev, 1e-14)
    p = R.params("WSe2", "TNN")
    H6 = R.H6(k[:, 0], k[:, 1], 23.0, p=p)
    dev = max(np.max(np.abs(H6[:, :3, :3] - R.H_block(k[:, 0], k[:, 1], 1, 23.0, p=p))),
              np.max(np.abs(H6[:, 3:, 3:] - R.H_block(k[:, 0], k[:, 1], -1, 23.0, p=p))),
              np.max(np.abs(H6[:, :3, 3:])), np.max(np.abs(H6[:, 3:, :3])))
    check("H6 = blockdiag(H_up, H_down), zero spin-flip blocks", dev, 1e-15)
    dev = 0
    for p in allp:
        for G in (R.B1, R.B2, R.B1 - 2 * R.B2):
            dev = max(dev, rel(R.H3(k[:, 0] + G[0], k[:, 1] + G[1], p=p), R.H3(k[:, 0], k[:, 1], p=p)))
    check("periodicity H3(k+G) = H3(k)", dev, 1e-12)
    dev = max(rel(R.H3(-k[:, 0], -k[:, 1], p=p), np.conj(R.H3(k[:, 0], k[:, 1], p=p))) for p in allp)
    check("time reversal H3(-k) = H3(k)^*", dev, 1e-14)
    Mx = np.diag([1.0, -1.0, 1.0])
    dev = max(rel(R.H3(-k[:, 0], k[:, 1], p=p), Mx @ R.H3(k[:, 0], k[:, 1], p=p) @ Mx) for p in allp)
    check("vertical mirror x->-x: H3(-kx,ky) = Mx H3 Mx, Mx=diag(1,-1,1) (d_xy odd)", dev, 1e-14)
    c, s_ = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    Rm = np.array([[c, -s_], [s_, c]])
    kr = k @ Rm.T
    dev = 0
    for p in allp:
        for sp in (1, -1):
            E0 = np.linalg.eigvalsh(R.H_block(k[:, 0], k[:, 1], sp, 41.0, p=p))
            E1 = np.linalg.eigvalsh(R.H_block(kr[:, 0], kr[:, 1], sp, 41.0, p=p))
            dev = max(dev, np.max(np.abs(E0 - E1)))
    check("C3 invariance of spin-block spectra (SOC + B=41 T), all sets", dev, 1e-12)
    dev = 0
    h = 1e-4
    for p in allp[::3]:
        d = R.H3(k[:, 0], k[:, 1], p=p, derivs=("H", "dx", "dy", "dxx", "dxy", "dyy"))
        f = lambda x, y, key="H": R.H3(x, y, p=p, derivs=(key,))[key] if key != "H" else R.H3(x, y, p=p)
        dx = (f(k[:, 0] + h, k[:, 1]) - f(k[:, 0] - h, k[:, 1])) / (2 * h)
        dy = (f(k[:, 0], k[:, 1] + h) - f(k[:, 0], k[:, 1] - h)) / (2 * h)
        dxx = (f(k[:, 0] + h, k[:, 1], "dx") - f(k[:, 0] - h, k[:, 1], "dx")) / (2 * h)
        dxy = (f(k[:, 0], k[:, 1] + h, "dx") - f(k[:, 0], k[:, 1] - h, "dx")) / (2 * h)
        dyy = (f(k[:, 0], k[:, 1] + h, "dy") - f(k[:, 0], k[:, 1] - h, "dy")) / (2 * h)
        dev = max(dev, rel(d["dx"], dx), rel(d["dy"], dy), rel(d["dxx"], dxx), rel(d["dxy"], dxy),
                  rel(d["dyy"], dyy))
    check("analytic dH, d2H vs central differences (h=1e-4)", dev, 1e-7)

    # ---------------------------------------------------------- literature
    sect("3. Literature values (Liu et al.)")
    dev = 0
    for f in ("GGA", "LDA"):
        for m in R.MATERIALS:
            p = R.params(m, "NN", f)
            e1, e2, t0, t1, t2, t11, t12, t22 = (p[x] for x in ("eps1", "eps2", "t0", "t1", "t2", "t11", "t12", "t22"))
            G = sorted([e1 + 6 * t0, e2 + 3 * (t11 + t22), e2 + 3 * (t11 + t22)])
            Kk = sorted([e2 - 1.5 * (t11 + t22) - 3 * np.sqrt(3) * t12, e1 - 3 * t0,
                         e2 - 1.5 * (t11 + t22) + 3 * np.sqrt(3) * t12])
            f1 = 0.5 * (e1 + e2) - t0 - 1.5 * t11 + 0.5 * t22
            f2 = 0.5 * np.sqrt((e1 - e2 - 2 * t0 + 3 * t11 - t22) ** 2 + 64 * t2 ** 2)
            Mm = sorted([f1 - f2, e2 + t11 - 3 * t22, f1 + f2])
            for kpt, ref in ((np.zeros(2), G), (R.K_POINT, Kk), (R.M_POINT, Mm)):
                E = np.linalg.eigvalsh(R.H3(kpt[0], kpt[1], p=p))
                dev = max(dev, np.max(np.abs(E - ref)))
    check("NN band energies at Gamma, K, M = Liu Table I closed forms (12 sets)", dev, 1e-12)
    for mo in ("NN", "TNN"):
        p = R.params("MoS2", mo)
        g0 = R.direct_gap_K(p=p, lam=0.0)
        info(f"MoS2 {mo} GGA gap at K, lambda=0 (eV)", f"{g0:.4f}  (Liu Fig.10: Delta = 1.663 eV)")
    p = R.params("MoS2", "NN")
    paper_check("MoS2 NN GGA gap at K (lambda=0) = 1.663 eV (Liu Fig. 10, 3 digits)",
          abs(R.direct_gap_K(p=p, lam=0.0) - 1.663), 5e-4)
    dev = 0
    for p in allp:
        Eu = np.linalg.eigvalsh(R.H_block(*R.K_POINT, 1, 0.0, p=p))
        Ed = np.linalg.eigvalsh(R.H_block(*R.K_POINT, -1, 0.0, p=p))
        dev = max(dev, abs((Eu[0] - Ed[0]) - 2 * p["lam"]))
    check("valence spin splitting at K = 2 lambda (first-order SOC, Liu Sec. IV)", dev, 1e-12)
    info("Liu Table IV Delta_SOC^v vs 2*lambda (GGA)",
         ", ".join(f"{m}: {R.DELTA_SOC_V[m]:.3f} vs {2 * R.params(m)['lam']:.3f}" for m in MAT4)
         + "  (difference = Liu's lambda^2/E_12 correction)")
    # orientation convention of the paper
    dev = 0
    for p in allp:
        for tau in (1, -1):
            E, U = np.linalg.eigh(R.H_block(*(tau * R.K_POINT), tau, 0.0, p=p))
            Lz = np.real(U[:, 0].conj() @ R.LZ @ U[:, 0])
            dev = max(dev, abs(Lz - 2 * tau))
    check("valence state at K=(4pi/3a,0) has L_z=+2, at K' L_z=-2 (paper sign convention)", dev, 1e-12)

    # --------------------------------------------------------- geometry
    sect("4. Band geometry: Berry curvature, orbital moment, D, velocity shift")
    p = R.params("MoS2", "TNN")
    kk = R.K_POINT + RNG.uniform(-0.3, 0.3, size=(60, 2))
    bq = R.band_quantities(kk[:, 0], kk[:, 1], 1, 30.0, p=p)
    check("sum_n Omega_n = 0", np.max(np.abs(bq["Omega"].sum(1))) / np.max(np.abs(bq["Omega"])), 1e-12)
    # Fukui-Hatsugai plaquette Berry curvature (independent of the sum rule)
    hh = 1e-4
    Kz = R.zeeman_soc_block(1, 30.0, p["lam"])

    def vecs(x, y):
        return np.linalg.eigh(R.H3(x, y, p=p) + Kz)[1]
    U1 = vecs(kk[:, 0] - hh / 2, kk[:, 1] - hh / 2)
    U2 = vecs(kk[:, 0] + hh / 2, kk[:, 1] - hh / 2)
    U3 = vecs(kk[:, 0] + hh / 2, kk[:, 1] + hh / 2)
    U4 = vecs(kk[:, 0] - hh / 2, kk[:, 1] + hh / 2)
    ov = lambda A, Bm: np.einsum("kin,kin->kn", np.conj(A), Bm)
    loop = ov(U1, U2) * ov(U2, U3) * ov(U3, U4) * ov(U4, U1)
    Om_fh = -np.angle(loop) / hh ** 2
    check("Omega_n (sum rule) vs plaquette Berry phase -Im ln prod<u|u'>/h^2", rel(bq["Omega"], Om_fh), 1e-5)
    # D-factor magnitude claim and M near K
    Kq = np.array([R.K_POINT])
    for m in MAT4:
        pm = R.params(m, "TNN")
        b60 = R.band_quantities(Kq[:, 0], Kq[:, 1], 1, 60.0, p=pm, need_dM=False)
        info(f"{m} TNN at K (spin up, B=60T)",
             f"M_v={b60['M'][0, 0]:+.3f} M_c={b60['M'][0, 1]:+.3f} mu_B ; Omega_v={b60['Omega'][0, 0]:+.3f} a^2 ;"
             f" |D_v-1|={abs(b60['D'][0, 0] - 1):.4f} |D_c-1|={abs(b60['D'][0, 1] - 1):.4f}")
    worst = 0
    for m in MAT4:
        pm = R.params(m, "TNN")
        kd, _ = R.disk_grid(0.3, 0.02)
        for sp in (1, -1):
            b60 = R.band_quantities(kd[:, 0], kd[:, 1], sp, 60.0, p=pm, need_dM=False)
            worst = max(worst, np.max(np.abs(b60["D"][:, :2] - 1)))
    paper_check("|D_n - 1| < 3% at 60 T for v,c bands within 0.3/a of K,K' (paper claim)", worst / 0.03, 1.0,
          f"max |D-1| = {worst:.4f}")
    Ms = []
    for m in MAT4:
        pm = R.params(m, "TNN")
        b0 = R.band_quantities(*R.K_POINT, 1, 0.0, p=pm, need_dM=False)
        Ms += [abs(b0["M"][0, 0]), abs(b0["M"][0, 1])]
    info("|M_v|,|M_c| at K, four materials TNN (mu_B)", np.round(Ms, 2).tolist() + ["paper: 3-6 mu_B"])
    # valley Zeeman g of the A transition at K (literature: about -4)
    for m in ("MoS2", "WSe2"):
        pm = R.params(m, "TNN")
        def Ecv(tau, B, mom):
            bq_ = R.band_quantities(*(tau * R.K_POINT), tau, B, p=pm, need_dM=False)
            E = bq_["E_shift"] if mom else bq_["E"]
            return E[0, 1] - E[0, 0]
        gL = (Ecv(1, 1.0, False) - Ecv(-1, 1.0, False)) / R.MU_B
        gM = (Ecv(1, 1.0, True) - Ecv(-1, 1.0, True)) / R.MU_B
        check(f"{m}: valley g of A transition from H_Z alone = -4 (on-site L_z = 2 tau)", abs(gL + 4), 1e-8)
        info(f"{m} valley g = [E_A(K)-E_A(K')]/mu_B B", f"H_Z only {gL:+.3f} ; with itinerant moment {gM:+.3f}")
    # dM/dk step convergence and check vs finite difference of M
    b1 = R.band_quantities(kk[:, 0], kk[:, 1], 1, 30.0, p=p, dM_step=1e-3)
    b2 = R.band_quantities(kk[:, 0], kk[:, 1], 1, 30.0, p=p, dM_step=5e-4)
    check("dM/dk: step 1e-3 vs 5e-4 (4th-order stencil)", rel(b1["dM"], b2["dM"]), 1e-8)
    check("velocity shift = -mu_B B dM/dk", rel(b1["v_shift"], -R.MU_B * 30.0 * b1["dM"]), 1e-15)
    vfd = (np.linalg.eigvalsh(R.H3(kk[:, 0] + 1e-5, kk[:, 1], p=p) + Kz)
           - np.linalg.eigvalsh(R.H3(kk[:, 0] - 1e-5, kk[:, 1], p=p) + Kz)) / 2e-5
    check("Hellmann-Feynman band velocity vs d(eps)/dkx (FD)", rel(b1["v"][0], vfd), 1e-8)

    # ------------------------------------------------------- two-band machinery
    sect("5. Generic machinery on a pure two-band (warped Dirac) Hamiltonian")
    v0, m0, ka0 = 1.0, 0.83, 0.25
    q = RNG.uniform(-0.2, 0.2, size=(200, 2))
    for tau in (1, -1):
        H, Hx, Hy = R.kp_H(q[:, 0], q[:, 1], v0, ka0, m0, tau)
        E, U, Vx, Vy = R.eig_velocity(H, Hx, Hy)
        Om, Mn = R.berry_and_moment(E, Vx, Vy, 3.19)
        gxx, _ = R.interband_geometry(E, Vx, Vx)
        gxy, _ = R.interband_geometry(E, Vx, Vy)
        kp = R.kp_kernels(q[:, 0], q[:, 1], v0, ka0, m0, tau)
        dl = np.real(Vx[:, 1, 1] - Vx[:, 0, 0])
        check(f"tau={tau:+d}: TB-machinery Delta^x_cv == 2 d_x|d|", rel(dl, kp["Delta"]["x"]), 1e-12)
        check(f"tau={tau:+d}: TB-machinery g^xx_cv == 1/4 |d_x n|^2", rel(gxx[:, 0, 1], kp["g"]["xx"]), 1e-12)
        check(f"tau={tau:+d}: TB-machinery g^xy_cv == 1/4 d_x n.d_y n", rel(gxy[:, 0, 1], kp["g"]["xy"]), 1e-11)
        check(f"tau={tau:+d}: Omega_v (sum rule) == +1/2 n.(dn x dn)", rel(Om[:, 0], kp["Omega_v"]), 1e-12)
        check(f"tau={tau:+d}: two-band M_c == M_v (paper Sec. II C)", rel(Mn[:, 1], Mn[:, 0]), 1e-12)
    # closed form of the two-band moment at the valley centre
    H, Hx, Hy = R.kp_H(np.zeros(1), np.zeros(1), v0, 0.0, m0, 1)
    E, U, Vx, Vy = R.eig_velocity(H, Hx, Hy)
    Om, Mn = R.berry_and_moment(E, Vx, Vy, 3.19)
    cM = R.M_E * (3.19e-10) ** 2 / R.HBAR ** 2 * R.E_CHARGE
    check("two-band M(k=0) = -(m_e a^2/hbar^2) tau v^2/m", abs(Mn[0, 0] - (-cM * v0 ** 2 / m0)) / abs(Mn[0, 0]), 1e-12)
    check("two-band Omega_v(k=0) = +v^2/(2 m^2) (tau=+1)", abs(Om[0, 0] - v0 ** 2 / (2 * m0 ** 2)), 1e-12)

    # ------------------------------------------------------- k.p closed forms
    sect("6. Two-band k.p ring averages vs paper closed forms (v=1, m=0.83, kappa=0.25)")
    table = {0.03: (3.922e-4, 3.935e-4), 0.05: (1.083e-3, 1.093e-3), 0.08: (2.733e-3, 2.798e-3)}
    for kr_, (ex, cf) in table.items():
        r = R.kp_ring(kr_, v0, ka0, m0)
        paper_check(f"k={kr_}: exact ring <Delta^x g_xx> vs paper Table (exact column {ex:.3e})",
              abs(r["inj"]["xxx"] - ex) / ex, 1.5e-3, f"got {r['inj']['xxx']:.5e}")
        paper_check(f"k={kr_}: closed form kappa v^3 k^2/m^3 vs paper Table ({cf:.3e})",
              abs(R.kp_inj_closed_lead(kr_, v0, ka0, m0) - cf) / cf, 1.5e-3)
        nxt = R.kp_inj_closed_next(kr_, v0, ka0, m0)
        ratio6 = (r["inj"]["xxx"] - nxt) / kr_ ** 6
        r_small = R.kp_ring(kr_ / 2, v0, ka0, m0)["inj"]["xxx"]
        ratio6_half = (r_small - R.kp_inj_closed_next(kr_ / 2, v0, ka0, m0)) / (kr_ / 2) ** 6
        check(f"k={kr_}: next-order closed form exact through O(k^4): (exact-next)/k^6 constant (k vs k/2)",
              abs(ratio6 - ratio6_half) / abs(ratio6), 2e-2, f"(exact-next)/k^6 = {ratio6:+.4f} vs {ratio6_half:+.4f}")
        check(f"k={kr_}: C3 family <D^x g_xx> = -<D^x g_yy> = -<D^y g_xy> = -<D^y g_yx>",
              max(abs(r["inj"]["xxx"] + r["inj"]["xyy"]), abs(r["inj"]["xxx"] + r["inj"]["yxy"]),
                  abs(r["inj"]["xxx"] + r["inj"]["yyx"])) / abs(r["inj"]["xxx"]), 1e-12)
        check(f"k={kr_}: mirror-forbidden <D^y g_yy>, <D^y g_xx>, <D^x g_xy> vanish",
              max(abs(r["inj"]["yyy"]), abs(r["inj"]["yxx"]), abs(r["inj"]["xxy"])) / abs(r["inj"]["xxx"]), 1e-12)
        check(f"k={kr_}: electric/gyration-type <Delta^a Omega> vanish (C3)",
              max(abs(r["el"]["x"]), abs(r["el"]["y"])) / abs(r["inj"]["xxx"]), 1e-12)
        sh = R.kp_shift_closed(kr_, v0, ka0, m0)
        check(f"k={kr_}: <Im C^cv_y;(yy)> - kappa v(2m^2-3k^2v^2)/4m^4 is O(k^4)",
              abs(r["shift"]["yyy"] - sh) / kr_ ** 4, 5.0, f"ratio/k^4={(r['shift']['yyy'] - sh) / kr_ ** 4:+.3f}")
        check(f"k={kr_}: shift beta family Im C_y;yy = -Im C_y;xx = -Im C_x;xy",
              max(abs(r["shift"]["yyy"] + r["shift"]["yxx"]), abs(r["shift"]["yyy"] + r["shift"]["xxy"])) / abs(r["shift"]["yyy"]), 1e-12)
        check(f"k={kr_}: Im C^cv == Im Cbar^cv (two-band)", abs(r["shift"]["yyy"] - r["shiftbar"]["yyy"]), 1e-12)
    r0 = R.kp_ring(1e-4, v0, ka0, m0)
    paper_check("k->0: shift constant kappa v/2m^2 = 1.8145e-1 (paper Table)", abs(r0["shift"]["yyy"] - 0.18145) / 0.18145, 5e-5,
          f"got {r0['shift']['yyy']:.6f}")
    rp, rm = R.kp_ring(0.05, v0, ka0, m0, +1), R.kp_ring(0.05, v0, ka0, m0, -1)
    check("valley parity: injection odd in tau", abs(rp["inj"]["xxx"] + rm["inj"]["xxx"]) / abs(rp["inj"]["xxx"]), 1e-12)
    check("valley parity: shift even in tau", abs(rp["shift"]["yyy"] - rm["shift"]["yyy"]) / abs(rp["shift"]["yyy"]), 1e-12)
    # K+K' sum odd in b and its linear coefficient 6 kappa v^3 b/m^4 (fixed k)
    mD, lam_, kr_ = 0.9, 0.15, 0.02
    for bb in (1e-4, 1e-3):
        mp, mm = mD - bb - lam_ / 2, mD + bb - lam_ / 2
        s = R.kp_ring(kr_, v0, ka0, mp, +1)["inj"]["xxx"] + R.kp_ring(kr_, v0, ka0, mm, -1)["inj"]["xxx"]
        lin = 6 * ka0 * v0 ** 3 * bb / (mD - lam_ / 2) ** 4 * kr_ ** 2
        check(f"b={bb}: <eta>_K+K' ~ 6 kappa v^3 b k^2/(m_D-lam/2)^4 (fixed k, O(k^4,b^3))",
              abs(s - lin) / abs(lin), 3e-3, f"sum={s:.5e} lin={lin:.5e}")

    # ------------------------------------------------------------- Lowdin
    sect("7. Lowdin reduction at K (kappa, v, w)")
    paper_tab = {"MoS2": (4.93, -0.24, -0.04), "MoSe2": (4.30, -0.02, 0.03),
                 "WS2": (5.90, -0.02, -0.51), "WSe2": (5.21, 0.09, -0.26)}
    for m in MAT4:
        rT = R.kappa_lowdin(m, "TNN", "GGA")
        rN = R.kappa_lowdin(m, "NN", "GGA")
        vT, kT, kN = paper_tab[m]
        info(f"{m} GGA active spin", f"v_TNN={rT['v']:.4f} eV A, kappa_TNN={rT['kappa']:+.4f}, kappa_NN={rN['kappa']:+.4f} eV A^2,"
             f" w_TNN={rT['w']:+.3f} eV A^2 ; paper: {vT}, {kT:+.2f}, {kN:+.2f}")
        paper_check(f"{m}: Lowdin v, kappa(TNN), kappa(NN) = paper Table II to rounding",
              max(abs(rT["v"] - vT) / 0.005, abs(rT["kappa"] - kT) / 0.005, abs(rN["kappa"] - kN) / 0.005), 1.0 + 1e-9)
        check(f"{m} NN: numerical kappa = paper closed form Eq.(kappa_nn) with SOC-shifted E's",
              abs(rN["kappa"] - rN["kappa_closed_paper"]) / max(abs(rN["kappa"]), 1e-3), 1e-10)
        check(f"{m} NN: numerical v/a = t = 3 sqrt2/4 (t1+sqrt3 t2)", abs(rN["v_a"] - rN["t_closed_paper"]) / rN["v_a"], 1e-12)
        res = max(rT["lin_resid"], rT["quad_resid"], rT["diag_isotropy_resid"], abs(rT["kappa_imag_a2"]), rT["offdiag_at_K"])
        check(f"{m} TNN: H(K) diagonal in Lowdin basis; cv block = v(tau qx-iqy)+kappa(tau qx+iqy)^2 exactly", res, 1e-12)
        rTm = R.kappa_lowdin(m, "TNN", "GGA", tau=-1)
        check(f"{m} TNN: kappa, v identical at K' (time reversal)", max(abs(rTm["kappa"] - rT["kappa"]), abs(rTm["v"] - rT["v"])), 1e-12)
    # independent check of w: band-edge curvature of the TB bands at K
    for m in MAT4:
        pm = R.params(m, "TNN")
        rT = R.kappa_lowdin(m, "TNN", "GGA")
        hh = 1e-3
        Es = [np.linalg.eigvalsh(R.H_block(R.K_POINT[0] + d_, 0.0, 1, 0.0, p=pm)) for d_ in (-hh, 0, hh)]
        curv = (Es[0] - 2 * Es[1] + Es[2]) / hh ** 2           # d2 eps / dkx^2 in eV a^2
        lhs = 0.25 * (curv[1] - curv[0]) * pm["a"] ** 2        # (hbar^2/4m_e)(1/m_c+1/|m_v|) in eV A^2
        mhalf = (Es[1][1] - Es[1][0]) / 2
        rhs = rT["w"] + rT["v"] ** 2 / (2 * mhalf)
        check(f"{m} TNN: (hbar^2/4m_e)(1/m_c+1/|m_v|) from TB curvature = w + v^2/2m",
              abs(lhs - rhs) / abs(rhs), 1e-5, f"lhs={lhs:.5f} rhs={rhs:.5f}")
    r = R.kappa_lowdin("MoS2", "TNN", "GGA", with_soc=False)
    info("MoS2 TNN kappa without SOC", f"{r['kappa']:+.4f} eV A^2 (paper Fig. kp_spectra uses -0.18)")
    lam0 = R.params("WS2")["lam"]
    vals = [R.kappa_lowdin("WS2", "TNN", "GGA", lam=l)["kappa"] for l in (0.0, lam0, 1.5 * lam0)]
    info("WS2 TNN kappa at lambda = 0, lam0, 1.5 lam0", np.round(vals, 4).tolist() + ["paper: +0.13, -0.02, -0.11"])
    paper_check("WS2 TNN kappa(lambda) = +0.13, -0.02, -0.11 (paper App. D) to rounding",
          max(abs(vals[0] - 0.13), abs(vals[1] + 0.02), abs(vals[2] + 0.11)) / 0.005, 1.0 + 1e-9)
    for m in MAT4:
        lda = R.kappa_lowdin(m, "TNN", "LDA")
        info(f"{m} TNN LDA kappa", f"{lda['kappa']:+.4f} eV A^2" + ("  (paper: +0.01)" if m == "MoS2" else ""))
    rr = R.kappa_lowdin("MoS2", "TNN", "LDA")
    paper_check("MoS2 TNN LDA kappa = +0.01 eV A^2 (paper Sec. III C) to rounding", abs(rr["kappa"] - 0.01) / 0.005, 1.0 + 1e-9)

    # ------------------------------------------------------------- grids
    sect("8. k-grids")
    kg, dAg = R.bz_grid(60)
    frac = np.linalg.solve(np.stack([R.B1, R.B2], 1), (-kg).T).T * 60
    ij = np.mod(np.rint(frac), 60).astype(int)
    keys = set(map(tuple, np.rint(np.linalg.solve(np.stack([R.B1, R.B2], 1), kg.T).T * 60).astype(int) % 60))
    check("BZ grid: k -> -k maps grid onto itself (mod G)", float(len(set(map(tuple, ij)) ^ keys)), 0.0)
    check("BZ grid: weights sum to BZ area", abs(dAg * len(kg) - R.BZ_AREA) / R.BZ_AREA, 1e-14)
    kd, dAd = R.disk_grid(0.6, 0.01)
    nh = len(kd) // 2
    check("disk grid: K' disk = -(K disk) exactly", float(np.max(np.abs(kd[nh:] + kd[:nh]))), 0.0)
    check("disk grid: area -> pi R^2 per disk", abs(dAd * nh - np.pi * 0.36) / (np.pi * 0.36), 2e-3)

    # ------------------------------------------------------------- eta
    sect("9. Injection coefficient eta^{a;bc} (tight binding)")
    p = R.params("MoS2", "TNN")
    Eg = R.direct_gap_K(p=p)
    info("MoS2 TNN GGA E_g (active, B=0, at K)", f"{Eg:.5f} eV")
    kd, dAd = R.disk_grid(0.3, 0.006)          # symmetry tests (exact on any paired grid)
    kf, dAf = R.disk_grid(0.3, 0.003)          # resolved grid for linearity tests
    w = np.array([Eg + 0.01, Eg + 0.05, Eg + 0.1])
    ep = R.eta_tensor(p=p, B=20.0, omega=w, nu=0.005, kpoints=kd, dA=dAd)
    em = R.eta_tensor(p=p, B=-20.0, omega=w, nu=0.005, kpoints=kd, dA=dAd)
    e0 = R.eta_tensor(p=p, B=0.0, omega=w, nu=0.005, kpoints=kd, dA=dAd)
    check("disk grid: eta(-B) = -eta(B) (all 8 components, B=20 T)", rel(ep + em, 0 * ep) if False else
          np.max(np.abs(ep + em)) / np.max(np.abs(ep)), 1e-9)
    check("disk grid: eta(B=0) = 0", np.max(np.abs(e0)) / np.max(np.abs(ep)), 1e-11)
    single = R.eta_tensor(p=p, B=0.0, omega=w, nu=0.005, kpoints=kd[:len(kd) // 2], dA=dAd)
    info("single valley K at B=0 vs K+K' at 20 T (eta^xxx)", f"{single[:, 0, 0, 0]} vs {ep[:, 0, 0, 0]}")
    fl_nospin = dict(spin_zeeman=False)
    ens = R.eta_tensor(p=p, B=20.0, omega=w, nu=0.005, kpoints=kd, dA=dAd, flags=fl_nospin)
    # tolerance: roundoff 1e-13 amplified by the K/K' cancellation (single valley ~20x the net)
    check("spin Zeeman (constant per spin block) leaves eta unchanged", np.max(np.abs(ens - ep)) / np.max(np.abs(ep)), 1e-10)
    e1 = R.eta_tensor(p=p, B=1.0, omega=w, nu=0.005, kpoints=kf, dA=dAf)
    e2 = R.eta_tensor(p=p, B=2.0, omega=w, nu=0.005, kpoints=kf, dA=dAf)
    check("linear in B at small field: eta(2T)/eta(1T) = 2 (rel dev)", np.max(np.abs(e2[:, 0, 0, 0] / e1[:, 0, 0, 0] - 2)) / 2, 1e-3)
    e10 = R.eta_tensor(p=p, B=10.0, omega=w[:1], nu=0.005, kpoints=kf, dA=dAf)
    e60 = R.eta_tensor(p=p, B=60.0, omega=w[:1], nu=0.005, kpoints=kf, dA=dAf)
    dl = abs(e60[0, 0, 0, 0] / (6 * e10[0, 0, 0, 0]) - 1)
    paper_check("E_g+0.01: deviation from linear law at 60 T < 1% (paper Sec. III C)", dl / 0.01, 1.0, f"deviation {dl:.4f}")
    ec3 = R.eta_tensor(p=p, B=20.0, omega=w, nu=0.005, kpoints=kd, dA=dAd, flags=dict(c3=True))
    a_ = ec3[:, 0, 0, 0]
    check("C3-symmetrized disk: alpha family xxx = -xyy = -yxy = -yyx",
          max(np.max(np.abs(a_ + ec3[:, 0, 1, 1])), np.max(np.abs(a_ + ec3[:, 1, 0, 1])),
              np.max(np.abs(a_ + ec3[:, 1, 1, 0]))) / np.max(np.abs(a_)), 1e-10)
    check("C3-symmetrized disk: beta family of eta (yyy, yxx, xxy) = 0 (sigma_v T)",
          max(np.max(np.abs(ec3[:, 1, 1, 1])), np.max(np.abs(ec3[:, 1, 0, 0])), np.max(np.abs(ec3[:, 0, 0, 1]))) / np.max(np.abs(a_)), 1e-10)
    check("symmetric in (bc): eta^{a;xy} = eta^{a;yx}", np.max(np.abs(ep[:, :, 0, 1] - ep[:, :, 1, 0])) / np.max(np.abs(ep)), 1e-13)
    # full BZ grid (C3- and mirror-invariant by construction)
    kg, dAg = R.bz_grid(90)
    wb = np.array([2.2, 2.8, 3.35])
    eb, el = R.eta_tensor(p=p, B=30.0, omega=wb, nu=0.01, kpoints=kg, dA=dAg, electric=True)
    ebm = R.eta_tensor(p=p, B=-30.0, omega=wb, nu=0.01, kpoints=kg, dA=dAg)
    sc = np.max(np.abs(eb))
    check("BZ grid N=90: eta(-B) = -eta(B)", np.max(np.abs(eb + ebm)) / sc, 1e-9)
    check("BZ grid: alpha family xxx = -xyy = -yxy = -yyx (C3, no symmetrization)",
          max(np.max(np.abs(eb[:, 0, 0, 0] + eb[:, 0, 1, 1])), np.max(np.abs(eb[:, 0, 0, 0] + eb[:, 1, 0, 1]))) / sc, 1e-9)
    check("BZ grid: beta family of magnetic injection vanishes", np.max(np.abs(eb[:, 1, 1, 1])) / sc, 1e-9)
    check("BZ grid: in-plane electric injection eta_el^{a;[xy]} = 0 (C3)",
          np.max(np.abs(el[:, :, 0, 1] - el[:, :, 1, 0])) / sc, 1e-9)
    eall = R.eta_tensor(p=p, B=30.0, omega=wb, nu=0.01, kpoints=kg, dA=dAg, flags=dict(pairs="all"))
    info("anti-resonant Lorentzian tails (pairs='all' minus resonant), rel. to eta^xxx",
         np.round((eall[:, 0, 0, 0] - eb[:, 0, 0, 0]) / eb[:, 0, 0, 0], 6).tolist())
    # flag decomposition
    flags_list = {"H_Z only": dict(moment_energy=False, phase_space=False, velocity_shift=False),
                  "+energy shift": dict(phase_space=False, velocity_shift=False),
                  "+velocity shift": dict(phase_space=False),
                  "all": {}}
    for nm, fl in flags_list.items():
        e_ = R.eta_tensor(p=p, B=30.0, omega=wb, nu=0.01, kpoints=kg, dA=dAg, flags=fl)
        info(f"MoS2 TNN N=90 B=30T eta^xxx [{nm}] at {wb.tolist()} eV", np.array2string(e_[:, 0, 0, 0], precision=4))
    # convergence of dM step inside eta
    ea = R.eta_tensor(p=p, B=30.0, omega=wb, nu=0.01, kpoints=kg, dA=dAg, dM_step=5e-4)
    check("eta: dM/dk step 1e-3 vs 5e-4", np.max(np.abs(ea - eb)) / sc, 1e-8)

    # ------------------------------------------------------------- shift
    sect("10. Shift current sigma^{a;(bc)} (tight binding, projector formalism)")
    kg, dAg = R.bz_grid(60)
    ws = np.array([2.0, 2.8])
    s0 = R.sigma_shift_tensor(p=p, B=0.0, omega=ws, nu=0.02, kpoints=kg, dA=dAg)
    sp_ = R.sigma_shift_tensor(p=p, B=30.0, omega=ws, nu=0.02, kpoints=kg, dA=dAg)
    sm_ = R.sigma_shift_tensor(p=p, B=-30.0, omega=ws, nu=0.02, kpoints=kg, dA=dAg)
    beta = s0[:, 1, 1, 1]
    sc = np.max(np.abs(beta))
    check("B=0: beta family yyy = -yxx = -xxy = -xyx", max(np.max(np.abs(beta + s0[:, 1, 0, 0])), np.max(np.abs(beta + s0[:, 0, 0, 1])),
                                                        np.max(np.abs(beta + s0[:, 0, 1, 0]))) / sc, 1e-6)
    check("B=0: mirror-forbidden alpha family of shift (xxx, xyy, yxy) = 0",
          max(np.max(np.abs(s0[:, 0, 0, 0])), np.max(np.abs(s0[:, 0, 1, 1])), np.max(np.abs(s0[:, 1, 0, 1]))) / sc, 1e-6)
    check("shift current even in B: sigma(30T) = sigma(-30T)", np.max(np.abs(sp_ - sm_)) / sc, 1e-6)
    s2 = R.sigma_shift_tensor(p=p, B=0.0, omega=ws, nu=0.02, kpoints=kg, dA=dAg, fd_step=5e-5)
    check("shift: FD step for d2P, 1e-4 vs 5e-5", np.max(np.abs(s2 - s0)) / sc, 1e-6)
    info("MoS2 TNN sigma^{y;yy} N=60 nu=20meV (nm muA V^-2)", f"B=0: {np.round(beta * 1e15, 4).tolist()} ; "
         f"B=30T change: {np.round((sp_[:, 1, 1, 1] - beta) / beta, 5).tolist()}")

    # ------------------------------------------------------------- informational
    sect("11. INFO: threshold field coefficient (coarse; compare signs with paper)")
    paper_chi = {"MoS2": 1.9e-4, "MoSe2": 0.2e-4, "WS2": 0.07e-4, "WSe2": -0.5e-4}
    for m in MAT4:
        pm = R.params(m, "TNN")
        Egm = R.direct_gap_K(p=pm)
        kd, dAd = R.disk_grid(0.3, 0.003)
        e = R.eta_tensor(p=pm, B=10.0, omega=[Egm + 0.01], nu=0.005, kpoints=kd, dA=dAd)
        info(f"{m} TNN GGA chi(E_g+0.01) = eta(10T)/10T, disk R=0.3/a h=0.003/a nu=5meV",
             f"{e[0, 0, 0, 0] / 10:+.3e} A m V^-2 s^-1 T^-1   (paper, R=0.6 h=0.001: {paper_chi[m]:+.1e})")

    npass = sum(ok for _, ok in RESULTS)
    nagree = sum(ok for _, ok in PAPER)
    print(f"\nSUMMARY: code checks {npass}/{len(RESULTS)} passed; paper comparisons {nagree}/{len(PAPER)} agree;"
          f" runtime {time.time() - t_start:.1f} s")
    for nm, ok in PAPER:
        if not ok:
            print("  DISCREPANCY:", nm)
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
