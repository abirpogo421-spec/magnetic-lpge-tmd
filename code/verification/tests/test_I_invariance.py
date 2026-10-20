"""I: invariance checks (results must not depend on how the question is asked).

I1 eta(omega0) does not depend on the other requested omegas (add omega0 + 2 eV; add omega0 - 0.3 eV); engine and kp2
I2 CHUNK 997 vs 20000; shuffled k order; additivity eta(A u B) = eta(A) + eta(B)
I3 integral of eta over omega0 +- 200 nu is independent of nu (narrow transition set: tiny disk at K; engine and kp2)
I4 eta(lam=0.2) followed by eta() gives the fresh-process value
"""
import os, sys, json, subprocess
import numpy as np
from vcore import check, Result, judge, rel, rel_scaled, SUITE_DIR
import fx


@check("I1", "I", "query independence: eta at omega0 independent of the other requested omegas")
def i1(ctx):
    eng = ctx.eng
    kp = fx.kset(ctx, "S2")
    out = []
    for var in ("Z", "F"):
        w0 = fx.eg(ctx, "MoS2", "TNN", "GGA") + 0.01
        a = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, [w0], 0.005, var, kp)["raw"][0]
        b = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, [w0, w0 + 2.0], 0.005, var, kp)["raw"][0]
        c = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, [w0 - 0.3, w0], 0.005, var, kp)["raw"][1]
        sK = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, [w0], 0.005, var, fx.kset(ctx, "S2K"))["raw"][0]
        d = max(rel_scaled(a, b, sK), rel_scaled(a, c, sK))
        out.append(judge("I1", f"engine MoS2 TNN {var} (rel to single-valley max)", d, 1e-12))
    kp2 = eng.kp2
    p = dict(v=2.76, kappa=-1.45, beta=3.485, mD=0.8715, lam=0.073); bb = 5.7883818060e-5 * 30
    w0 = 2 * p["mD"] - p["lam"] + 0.01
    f = lambda om: kp2.response("MoS2", bb, np.array(om), p=p, R=0.2, Nk=81, nu=0.005, want_shift=False, full=True)[0]
    one = kp2.response("MoS2", bb, np.array([w0]), p=p, R=0.2, Nk=81, nu=0.005, want_shift=False, pairs=((1, 1),), full=True)[0]
    a, b, c = f([w0])[0], f([w0, w0 + 2.0])[0], f([w0 - 0.3, w0])[1]
    out.append(judge("I1", "kp2 response (rel to single-valley max)", max(rel_scaled(a, b, one), rel_scaled(a, c, one)), 1e-12))
    return out


@check("I5", "I", "production-shaped calls: 1001 omega on a 51k-point disk set (lchunk < CHUNK, several k chunks)")
def i5(ctx):
    """Red team RT22: an indexing/accumulation error that appears only when len(omega) > 200 AND a k chunk holds more than
    lchunk transitions. eta at omega0 from a 1001-omega request must equal eta from [omega0] alone; CHUNK 997 vs 20000 at
    1001 omega must agree; and the 1001-omega spectrum must equal the sum of its two disk halves."""
    eng = ctx.eng; L = eng.L
    kp = eng.disks(0.9, 0.01)                         # 2 x 25.4k = 50.9k points > CHUNK
    w0 = fx.eg(ctx, "MoS2", "TNN", "GGA") + 0.01
    om = np.linspace(w0 - 0.2, w0 + 2.0, 1001); j = 91; om[j] = w0
    kK = eng.split_valleys(kp)[0]
    one = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, [w0], 0.005, "Z", kp)["raw"][0]
    sK = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, [w0], 0.005, "Z", kK)["raw"][0]
    big = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, om, 0.005, "Z", kp)["raw"]
    old = L.CHUNK
    try:
        L.CHUNK = 997
        small = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, om, 0.005, "Z", kp)["raw"]
    finally:
        L.CHUNK = old
    bigK = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, om, 0.005, "Z", kK)["raw"]
    bigKp = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, om, 0.005, "Z", eng.split_valleys(kp)[1])["raw"]
    return [judge("I5", "eta(omega0) from a 1001-omega request == eta from [omega0] (51k k-points)", rel_scaled(big[j], one, sK), 1e-12),
            judge("I5", "1001 omega: CHUNK 997 vs 20000", rel_scaled(small, big, bigK), 1e-12),
            judge("I5", "1001 omega: eta(K u K') == eta(K) + eta(K')", rel_scaled(bigK + bigKp, big, bigK), 1e-12)]


@check("I2", "I", "CHUNK size, k order, and additivity")
def i2(ctx):
    eng = ctx.eng; L = eng.L
    kx, ky, dA = fx.kset(ctx, "S2")
    om = fx.omega(ctx, "S2", "MoS2", "TNN", "GGA")
    base = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, om, 0.005, "F", (kx, ky, dA))["raw"]
    sK = fx.eng_eta(ctx, "MoS2", "TNN", "GGA", "F", 10.0, "S2K")["raw"]
    old = L.CHUNK
    try:
        L.CHUNK = 997
        small = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, om, 0.005, "F", (kx, ky, dA))["raw"]
    finally:
        L.CHUNK = old
    perm = np.random.default_rng(1).permutation(len(kx))
    shuf = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, om, 0.005, "F", (kx[perm], ky[perm], dA))["raw"]
    half = len(kx) // 3
    A = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, om, 0.005, "F", (kx[perm[:half]], ky[perm[:half]], dA))["raw"]
    Bp = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, om, 0.005, "F", (kx[perm[half:]], ky[perm[half:]], dA))["raw"]
    return [judge("I2", "CHUNK 997 vs 20000", rel_scaled(small, base, sK), 1e-12),
            judge("I2", "shuffled k order", rel_scaled(shuf, base, sK), 1e-12),
            judge("I2", "additivity eta(A u B) = eta(A) + eta(B)", rel_scaled(A + Bp, base, sK), 1e-12)]


def _integral(f_eta, center, nu):
    om = center + np.linspace(-200 * nu, 200 * nu, 8001)
    y = f_eta(om, nu)
    return np.trapezoid(y, om, axis=0) if hasattr(np, "trapezoid") else np.trapz(y, om, axis=0)


@check("I3", "I", "integral of eta over +-200 nu independent of nu (narrow transition set)")
def i3(ctx):
    eng = ctx.eng; L = eng.L
    out = []
    # engine on a pure warped-Dirac H (remote band decoupled), single valley, tiny disk |q| <= 0.004/a:
    # all transitions lie within ~1 meV, so the in-window Lorentzian fraction (2/pi) atan(200) is nu-independent.
    from test_H_symmetry import _dirac_H6
    kx, ky, dA = eng.disks(0.004, 0.0005)
    m = kx > 0
    kp = (kx[m], ky[m], dA)
    orig = L._H6
    try:
        L._H6 = _dirac_H6(1.2, 0.3, 0.85, 0.07)
        c = 2 * (0.85 - 0.035)
        f = lambda om, nu: L.eta("MoS2", "TNN", None, 10.0, om, nu, dict(spin=True), kpts=kp)["raw"]
        I5, I10 = _integral(f, c, 0.005), _integral(f, c, 0.010)
    finally:
        L._H6 = orig
    out.append(judge("I3", "engine (warped-Dirac H): |I(nu=5 meV) - I(10 meV)|/|I|", rel(I5, I10), 1e-4))
    kp2 = eng.kp2
    p = dict(v=2.76, kappa=-1.45, beta=3.485, mD=0.8715, lam=0.073)
    c2 = 2 * p["mD"] - p["lam"]
    g = lambda om, nu: kp2.response("MoS2", 0.0, om, p=p, R=0.004, Nk=9, nu=nu, want_shift=False, pairs=((1, 1),), full=True)[0]
    J5, J10 = _integral(g, c2, 0.005), _integral(g, c2, 0.010)
    out.append(judge("I3", "kp2: |I(nu=5 meV) - I(10 meV)|/|I|", rel(J5, J10), 1e-4))
    return out


@check("I4", "I", "no state leaks: eta(lam=0.2) then eta() equals a fresh process")
def i4(ctx):
    eng = ctx.eng
    kp = eng.disks(0.1, 0.01)
    om = [1.9, 2.0]
    eng.eta_phys("MoS2", "TNN", "GGA", 10.0, om, 0.005, "F", kp, lam=0.2)
    a = eng.eta_phys("MoS2", "TNN", "GGA", 10.0, om, 0.005, "F", kp)["raw"]
    code = (f"import sys; sys.path.insert(0, {os.path.join(SUITE_DIR, 'adapters')!r}); import engine as E, numpy as np;"
            f"E.load({ctx.code_root!r}); kp = E.disks(0.1, 0.01);"
            f"r = E.eta_phys('MoS2','TNN','GGA',10.0,{om!r},0.005,'F',kp)['raw']; np.save(sys.argv[1], r)")
    fn = os.path.join(ctx.scratch, "i4_fresh.npy")
    subprocess.run([sys.executable, "-c", code, fn], check=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    b = np.load(fn)
    return judge("I4", "in-process after lambda override vs fresh process", rel(a, b), 1e-15)
