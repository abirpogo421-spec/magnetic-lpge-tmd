"""H: Hamiltonian, symmetry and analytic checks that need no trust in either code.

H1  eta(-B) = -eta(B), raw tensor, both from scratch (S2, 3 materials, F)          rel 1e-9
H2  eta(0) = 0: TB on S2 (F) and kp2 with K+K'                                       1e-10 x single-valley max
H3  C3 on the RAW tensor: x;xx = -x;yy = -y;xy on S2-C3 and S3-C3; mirror-forbidden (beta family) vanish
H4  Hermiticity (1000 k, B = +-100 T) and exact block structure vs the reference 6x6 H
H5  H_dn(-k,-B) = H_up(k,B)*; spin part of (up - dn) = 2 mu_B B; orbital part equal in both blocks
H6  L_z eigenvalues {0, +-2}; engine L_z == reference L_z; valence <L_z> = +2 at K, -2 at K'
H7  engine on a pure warped-Dirac H (lpge._H6 monkeypatched): per-k kernel vs ref.kp_kernels, ring average vs
    closed form, sign chi = -sign kappa for kappa in {+-0.3, +-1}
H8  local_kpts inversion symmetric as integer sets
H9  eta vs a brute-force double loop over 1e4 synthetic transitions (chunk_transitions monkeypatched)
H10 Hellmann-Feynman v_nn = dE_n/dk; sum_n Omega_n = 0 per block
"""
import numpy as np
from vcore import check, Result, judge, exact, rel, rel_scaled
import fx

MUB = 5.7883818060e-5


@check("H1", "H", "eta(-B) = -eta(B), raw tensor, S2, F, three materials")
def h1(ctx):
    out = []
    for mat in ("MoS2", "WS2", "WSe2"):
        ep = fx.eng_eta(ctx, mat, "TNN", "GGA", "F", 10.0, "S2")["raw"]
        em = fx.eng_eta(ctx, mat, "TNN", "GGA", "F", -10.0, "S2")["raw"]
        out.append(judge("H1", f"{mat} TNN F: |eta(B)+eta(-B)|/|eta(B)|", rel_scaled(ep + em, 0 * ep, ep), 1e-9))
    return out


@check("H2", "H", "eta(B=0) = 0 for T-paired disks (TB, F) and for kp2 K+K'")
def h2(ctx):
    out = []
    for mat in ("MoS2", "WSe2"):
        e0 = fx.eng_eta(ctx, mat, "TNN", "GGA", "F", 0.0, "S2")["raw"]
        eK = fx.eng_eta(ctx, mat, "TNN", "GGA", "F", 0.0, "S2K")["raw"]
        out.append(judge("H2", f"TB {mat} TNN F: max|eta(0)|/max|single valley|", rel_scaled(e0, 0 * e0, eK), 1e-10))
    kp2 = ctx.eng.kp2
    p = dict(v=2.76, kappa=-1.45, beta=3.485, mD=0.8715, lam=0.073)
    om = 2 * p["mD"] - p["lam"] + np.array([0.01, 0.1])
    tot = kp2.response("MoS2", 0.0, om, p=p, R=0.2, Nk=81, nu=0.005, want_shift=False, full=True)[0]
    one = kp2.response("MoS2", 0.0, om, p=p, R=0.2, Nk=81, nu=0.005, want_shift=False, pairs=((1, 1),), full=True)[0]
    out.append(judge("H2", "kp2 K+K' at b=0 relative to single valley (R=0.2/A, Nk=81)", rel_scaled(tot, 0 * tot, one), 1e-10))
    # the cancellation needs a mirror-symmetric disk mask; linspace rounding breaks it for some Nk (e.g. 61)
    asym = {}
    for Nk in (401, 101, 61):
        ks = np.linspace(-0.2, 0.2, Nk); KX, KY = np.meshgrid(ks, ks, indexing="ij")
        asym[Nk] = int(((KX ** 2 + KY ** 2 <= 0.2 * 0.2) != (KX[::-1] ** 2 + KY ** 2 <= 0.2 * 0.2)).sum())
    out.append(exact("H2", "kp2 disk mask mirror-symmetric at the production/smoke grids (Nk = 401, 101)",
                     asym[401] == 0 and asym[101] == 0, measured=str(asym),
                     detail="Nk=61 has %d boundary points without a mirror partner: K+K' then fails to cancel at b=0 by 6e-6 "
                            "of a single valley (latent kp2 fragility; not used in production)" % asym[61]))
    return out


ALPHA = [((0, 0, 0), 1), ((0, 1, 1), -1), ((1, 0, 1), -1), ((1, 1, 0), -1)]
BETA = [(1, 1, 1), (1, 0, 0), (0, 0, 1), (0, 1, 0)]


def _c3_dev(T):
    a = T[:, 0, 0, 0]
    sc = max(np.max(np.abs(a)), 1e-300)
    d_alpha = max(np.max(np.abs(T[:, i, j, k] * s - a)) for (i, j, k), s in ALPHA) / sc
    d_beta = max(np.max(np.abs(T[:, i, j, k])) for (i, j, k) in BETA) / sc
    return d_alpha, d_beta


H3_ALPHA_TOL = 1e-5      # feasible gate (measured FD truncation 2e-7 (Z) .. 9e-7 (F) at EPS = 1e-4)


@check("H3", "H", "C3 and mirror relations of the raw tensor on C3-closed k-sets (S2-C3, S3-C3)")
def h3(ctx):
    out = []
    for kn in ("S2C3", "S3C3"):
        for mat, model in (("MoS2", "TNN"), ("WSe2", "NN")):
            T = fx.eng_eta(ctx, mat, model, "GGA", "F", 10.0, kn)["raw"]
            da, db = _c3_dev(T)
            r = judge("H3", f"{kn} {mat}/{model}: alpha family x;xx=-x;yy=-y;xy=-y;yx", da, H3_ALPHA_TOL,
                      detail=f"protocol tol 1e-10 infeasible: deviation is FD truncation of the engine (scales as EPS^2: "
                             f"1.8e-5 at EPS=1e-3, 1.8e-7 at 1e-4 for Z; roundoff floor at 1e-5), so the gate is {H3_ALPHA_TOL:g}")
            out.append(r)
            out.append(judge("H3", f"{kn} {mat}/{model}: mirror-forbidden beta family of eta vanishes", db, H3_ALPHA_TOL if kn == "S2C3" else 1e-10))
    return out


@check("H4", "H", "Hermiticity (B=+-100 T) and block structure vs reference 6x6 H")
def h4(ctx):
    eng, ref = ctx.eng, ctx.ref
    rng = np.random.default_rng(4)
    k = rng.uniform(-6, 6, size=(1000, 2))
    dh = db = 0.0
    for mat, model, func in fx.CONFIGS:
        for B in (100.0, -100.0):
            up, dn = eng.H_blocks(mat, model, func, k[:, 0], k[:, 1], B)
            for H in (up, dn):
                dh = max(dh, np.max(np.abs(H - np.conj(np.swapaxes(H, -1, -2)))))
            H6 = ref.H6(k[:, 0], k[:, 1], B, p=ref.params(mat, model, func))
            Z = np.zeros_like(up)
            E6 = np.concatenate([np.concatenate([up, Z], -1), np.concatenate([Z, dn], -1)], -2)
            db = max(db, np.max(np.abs(E6 - H6)))
    return [judge("H4", "Hermiticity max|H - H^dag| (eV)", dh, 1e-14),
            judge("H4", "blockdiag(H_up, H_dn) == reference H6 (eV; mu_B 2014/2018 gives ~5e-12 at 100 T)", db, 1e-10)]


@check("H5", "H", "spin-block structure: time reversal, spin part, orbital part")
def h5(ctx):
    eng = ctx.eng
    kx, ky = fx.s1(ctx)
    dT = dS = dO = 0.0
    for mat, model, func in fx.CONFIGS:
        for B in (10.0, -37.0):
            up, dn = eng.H_blocks(mat, model, func, kx, ky, B)
            upm, dnm = eng.H_blocks(mat, model, func, -kx, -ky, -B)
            u0, d0 = eng.H_blocks(mat, model, func, kx, ky, 0.0)
            dT = max(dT, np.max(np.abs(dnm - np.conj(up))), np.max(np.abs(upm - np.conj(dn))))
            spin = (up - dn) - (u0 - d0)
            dS = max(dS, rel(spin, np.broadcast_to(2 * MUB * B * np.eye(3), spin.shape)))
            orb = 0.5 * ((up + dn) - (u0 + d0))
            dO = max(dO, rel(orb, np.broadcast_to(MUB * B * ctx.ref.LZ, orb.shape)))
    return [judge("H5", "H_dn(-k,-B) = H_up(k,B)^* and H_up(-k,-B) = H_dn(k,B)^* (eV)", dT, 1e-14),
            judge("H5", "spin part of (H_up - H_dn) = 2 mu_B B 1 (g_s = 2)", dS, 1e-8),
            judge("H5", "orbital part (H_up + H_dn)/2 field term = mu_B B L_z", dO, 1e-8)]


@check("H6", "H", "L_z matrix and valley orbital character")
def h6(ctx):
    eng, ref = ctx.eng, ctx.ref
    ev = np.sort(np.linalg.eigvalsh(eng.tb.LZ))
    out = [judge("H6", "eigenvalues of engine L_z = {-2, 0, +2}", float(np.max(np.abs(ev - [-2, 0, 2]))), 1e-12),
           judge("H6", "engine L_z == reference L_z (paper Eq. htot)", float(np.max(np.abs(eng.tb.LZ - ref.LZ))), 0.0)]
    worst = 0.0
    for mat, model, func in fx.CONFIGS:
        for tau, blk in ((1, 0), (-1, 1)):
            H = eng.H_blocks(mat, model, func, [tau * 4 * np.pi / 3], [0.0], 0.0)[blk][0]
            E, U = np.linalg.eigh(H)
            for LZ in (ref.LZ, eng.tb.LZ):
                worst = max(worst, abs(np.real(U[:, 0].conj() @ LZ @ U[:, 0]) - 2 * tau))
    out.append(judge("H6", "valence <L_z> = +2 at K, -2 at K' (reference and engine L_z)", worst, 1e-12))
    return out


# ------------------------------------------------------------------------------------------------- H7
def _dirac_H6(v, kappa, mD, lam, w=0.0, far=100.0):
    """A pure warped-Dirac model in the engine's (up, dn) 3x3 block format: valley tau = sign(kx), q = k - tau K,
    a third band at +far eV. Orbital Zeeman enters through the mass, m = mD - tau(b + s lam/2)."""
    KX = 4 * np.pi / 3

    def H6(kx, ky, model, B, spin=True):
        tau = np.where(kx > 0, 1.0, -1.0)
        qx, qy = kx - tau * KX, ky
        b = MUB * B
        blocks = []
        for s in (1, -1):
            dx = v * tau * qx + kappa * (qx ** 2 - qy ** 2)
            dy = v * qy - 2 * kappa * tau * qx * qy
            dz = mD + w * (qx ** 2 + qy ** 2) - tau * (b + s * lam / 2)
            H = np.zeros(kx.shape + (3, 3), complex)
            H[:, 0, 0] = -dz; H[:, 1, 1] = dz            # band 0 = valence (lower), band 1 = conduction
            H[:, 0, 1] = dx + 1j * dy; H[:, 1, 0] = dx - 1j * dy
            H[:, 2, 2] = far
            blocks.append(H)
        return blocks[0], blocks[1]
    return H6


@check("H7", "H", "engine on a pure warped-Dirac Hamiltonian: kernel, ring average, sign rule")
def h7(ctx):
    eng, ref = ctx.eng, ctx.ref
    L = eng.L
    v, mD, lam = 1.2, 0.85, 0.07
    out = []
    orig = L._H6
    try:
        # (a) per-k kernel Delta^x g_xx vs reference kp kernels (tau = +1, s = +1 block)
        kappa = 0.3
        L._H6 = _dirac_H6(v, kappa, mD, lam)
        rng = np.random.default_rng(3)
        q = rng.uniform(-0.15, 0.15, size=(300, 2))
        kx = 4 * np.pi / 3 + q[:, 0]; ky = q[:, 1]
        tr = L.chunk_transitions(kx, ky, "TNN", 0.0, dict(spin=True), 1.0, 0.0)
        ecv, W, Da, g = tr[0]                                 # up block, c = band 1
        m = mD - lam / 2
        # engine H = [[-dz, dx+i dy],[dx-i dy, dz]] = -(d.sigma)^T-like; the reference d-vector convention:
        kk = ref.kp_kernels(q[:, 0], q[:, 1], v, kappa, m, +1)
        e1 = rel(Da[:, 0] * g[:, 0, 0], kk["Delta"]["x"] * kk["g"]["xx"])
        e2 = rel(ecv, 2 * kk["dabs"])
        out.append(judge("H7", "per-k Delta^x_cv g_xx and e_cv vs ref.kp_kernels", max(e1, e2), 1e-6))
        # (b) ring average at small k vs the closed form (paper Eq. app_injfull)
        dev = 0.0
        for kr in (0.005, 0.01, 0.02):
            ph = 2 * np.pi * np.arange(2048) / 2048
            t = L.chunk_transitions(4 * np.pi / 3 + kr * np.cos(ph), kr * np.sin(ph), "TNN", 0.0, dict(spin=True), 1.0, 0.0)[0]
            avg = np.mean(t[2][:, 0] * t[3][:, 0, 0])
            cf = ref.kp_inj_closed_next(kr, v, kappa, m)
            dev = max(dev, abs(avg - cf) / abs(cf))
        out.append(judge("H7", "ring average <Delta^x g_xx> vs closed form, k <= 0.02/a", dev, 1e-3))
        # (c) sign rule chi = -sign(kappa) at E_g + 0.01 eV on T-paired disks (physical sign)
        signs = []
        kp = L.local_kpts(0.25, 0.0025)
        for kappa in (0.3, -0.3, 1.0, -1.0):
            L._H6 = _dirac_H6(v, kappa, mD, lam)
            om = np.array([2 * m + 0.01])
            rp = L.eta("MoS2", "TNN", None, 10.0, om, 0.005, dict(spin=True), kpts=kp)
            rm = L.eta("MoS2", "TNN", None, -10.0, om, 0.005, dict(spin=True), kpts=kp)
            chi = -2 * rp["si_factor"] * (rp["sym"][0, 0, 0, 0] - rm["sym"][0, 0, 0, 0]) / 20
            signs.append((kappa, np.sign(chi)))
        ok = all(s == -np.sign(k) for k, s in signs)
        out.append(exact("H7", "sign chi = -sign kappa (kappa = +-0.3, +-1)", ok, measured=str([(k, int(s)) for k, s in signs])))
    finally:
        L._H6 = orig
    return out


@check("H8", "H", "local_kpts: K' disk = -(K disk) as integer sets; areas")
def h8(ctx):
    L = ctx.eng.L
    out = []
    for R, h in ((0.3, 0.01), (0.9, 0.0007), (0.9, 0.002), (0.3, 0.03)):
        kx, ky, dA = L.local_kpts(R, h)
        KX = 4 * np.pi / 3
        m = kx > 0
        iA = np.rint(np.stack([(kx[m] - KX) / h, ky[m] / h], 1)).astype(np.int64)
        iB = np.rint(np.stack([(kx[~m] + KX) / h, ky[~m] / h], 1)).astype(np.int64)
        key = lambda a: np.sort(a[:, 0] * 10 ** 7 + a[:, 1])
        ok = (len(iA) == len(iB) > 0 and np.array_equal(key(iA), key(-iB)) and len(np.unique(key(iA))) == len(iA)
              and np.allclose(kx[m] - KX, iA[:, 0] * h, atol=1e-12) and np.allclose(kx[~m] + KX, iB[:, 0] * h, atol=1e-12))
        out.append(exact("H8", f"R={R} h={h}: inversion-paired integer sets", ok, measured=f"{len(iA)}+{len(iB)} pts"))
    return out


@check("H9", "H", "eta vs brute-force double loop over 1e4 synthetic transitions")
def h9(ctx):
    L = ctx.eng.L
    rng = np.random.default_rng(9)
    nu = 0.005; c = 1.8; n = 10000
    ecv = c + rng.uniform(-200 * nu, 200 * nu, n)
    W = 1 + 0.01 * rng.standard_normal(n); Da = rng.standard_normal((n, 2))
    g = rng.standard_normal((n, 2, 2)); g = g + np.swapaxes(g, 1, 2)
    orig = L.chunk_transitions
    om = np.array([c - 0.05, c, c + 0.0301, c + 0.7])
    dA = 1e-4
    try:
        L.chunk_transitions = lambda *a, **k: [(ecv, W, Da, g)]
        r = L.eta("MoS2", "TNN", None, 0.0, om, nu, dict(spin=True), kpts=(np.zeros(1), np.zeros(1), dA))
    finally:
        L.chunk_transitions = orig
    bf = np.zeros((len(om), 2, 2, 2))
    for iw, w in enumerate(om):
        Lw = (nu / np.pi) / ((w - ecv) ** 2 + nu ** 2)
        for a in range(2):
            for b in range(2):
                for cc in range(2):
                    bf[iw, a, b, cc] = np.sum(Lw * W * Da[:, a] * g[:, b, cc])
    bf = bf * np.pi * dA / (2 * np.pi) ** 2
    bf = 0.5 * (bf + np.swapaxes(bf, -1, -2))
    return judge("H9", "eta(raw) vs brute force (no binning, no window, exact Lorentzian)", rel(r["raw"], bf), 1e-12)


@check("H10", "H", "Hellmann-Feynman velocities and Berry sum rule (engine internals)")
def h10(ctx):
    eng = ctx.eng
    kx, ky = fx.s1(ctx)
    hh = 1e-5; dv = ds = 0.0
    for mat, model, func in fx.CONFIGS[:4]:
        bq = eng.band_quantities(mat, model, func, kx, ky, 10.0)
        bp = eng.H_blocks(mat, model, func, kx + hh, ky, 10.0); bm = eng.H_blocks(mat, model, func, kx - hh, ky, 10.0)
        for blk in (0, 1):
            vfd = (np.linalg.eigvalsh(bp[blk]) - np.linalg.eigvalsh(bm[blk])) / (2 * hh)
            dv = max(dv, rel(np.einsum("knn->kn", bq[blk]["vx"]).real, vfd))
            Om = bq[blk]["Omega"]
            ds = max(ds, np.max(np.abs(Om.sum(1))) / np.max(np.abs(Om)))
    return [judge("H10", "v_nn (engine) vs dE_n/dk_x (FD 1e-5)", dv, 1e-7),
            judge("H10", "sum_n Omega_n = 0 per block (rel)", ds, 1e-10)]
