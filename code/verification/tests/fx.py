"""Shared test sets and cached engine/reference evaluations (protocol sec. 4, "Test sets").

S1  400 seeded random k (box |kx|,|ky| <= 6/a) with every intra-block gap > 1 meV (all materials, B = 0)
S2  engine disks R = 0.3/a, h = 0.01/a; nu = 5 meV; B = +-10 T; omega = E_g + {7.5, 8.75, 10, 11.25, 12.5 meV, 0.1, 1.5 eV}
S3  engine full-BZ grid N = 60, nu = 30 meV, 41 omega from E_g - 0.1 to E_g + 2.9 eV
S2-C3 = ref.c3_expand(S2);  S3-C3 = ref.bz_grid(60) (Gamma-centred, C3 and inversion invariant)
E_g is the reference direct gap at K (ref.direct_gap_K), so both codes get identical omega arrays.
"""
import numpy as np

S2_OFF = np.array([0.0075, 0.00875, 0.01, 0.01125, 0.0125, 0.1, 1.5])
S2_NU = 0.005
S3_NU = 0.030
S3_OFF = np.linspace(-0.1, 2.9, 41)
CONFIGS = [("MoS2", "TNN", "GGA"), ("MoSe2", "TNN", "GGA"), ("WS2", "TNN", "GGA"), ("WSe2", "TNN", "GGA"),
           ("MoS2", "NN", "GGA"), ("MoSe2", "NN", "GGA"), ("WS2", "NN", "GGA"), ("WSe2", "NN", "GGA"),
           ("MoS2", "TNN", "LDA")]


def eg(ctx, mat, model, func, lam=None):
    from refside import eg as _eg
    return ctx.cached(("eg", mat, model, func, lam), lambda: _eg(mat, model, func, lam=lam))


def s1(ctx):
    def build():
        rng = np.random.default_rng(20261014)
        ref = ctx.ref
        pts = []
        while len(pts) < 400:
            k = rng.uniform(-6, 6, size=(200, 2))
            ok = np.ones(len(k), bool)
            for mat, model, func in CONFIGS:
                p = ref.params(mat, model, func)
                for s in (1, -1):
                    E = np.linalg.eigvalsh(ref.H_block(k[:, 0], k[:, 1], s, 0.0, p=p))
                    ok &= np.min(np.diff(E, axis=1), axis=1) > 1e-3
            pts += list(k[ok])
        k = np.array(pts[:400])
        return k[:, 0].copy(), k[:, 1].copy()
    return ctx.cached("S1", build)


def s1_wide_gap(ctx, gmin=0.1):
    """S1 points whose intra-block gaps all exceed gmin (for derivatives of the moment, which carry 1/gap^3)."""
    def build():
        kx, ky = s1(ctx)
        ok = np.ones(len(kx), bool)
        for mat, model, func in CONFIGS:
            p = ctx.ref.params(mat, model, func)
            for s in (1, -1):
                E = np.linalg.eigvalsh(ctx.ref.H_block(kx, ky, s, 0.0, p=p))
                ok &= np.min(np.diff(E, axis=1), axis=1) > gmin
        return kx[ok], ky[ok]
    return ctx.cached(("S1g", gmin), build)


def kset(ctx, name):
    """Named k-sets as engine kpts tuples (kx, ky, dA)."""
    def build():
        eng = ctx.eng
        if name == "S2":
            return eng.disks(0.3, 0.01)
        if name == "S2K":
            return eng.split_valleys(kset(ctx, "S2"))[0]
        if name == "S2Kp":
            return eng.split_valleys(kset(ctx, "S2"))[1]
        if name == "S3":
            return eng.grid_cart(60)
        if name == "S2C3":
            from refside import K
            k, dA = K(kset(ctx, "S2"))
            kk, ww = ctx.ref.c3_expand(k, dA)
            return kk[:, 0].copy(), kk[:, 1].copy(), float(ww[0])
        if name == "S3C3":
            k, dA = ctx.ref.bz_grid(60)
            return k[:, 0].copy(), k[:, 1].copy(), float(dA)
        raise KeyError(name)
    return ctx.cached(("kset", name), build)


def omega(ctx, kname, mat, model, func, lam=None):
    g = eg(ctx, mat, model, func, lam)
    return g + (S3_OFF if kname.startswith("S3") else S2_OFF)


def nu_of(kname):
    return S3_NU if kname.startswith("S3") else S2_NU


def eng_eta(ctx, mat, model, func, var, B, kname, lam=None):
    key = ("eng", mat, model, func, var, float(B), kname, lam)

    def run():
        r = ctx.eng.eta_phys(mat, model, func, B, omega(ctx, kname, mat, model, func, lam), nu_of(kname), var,
                             kset(ctx, kname), lam=lam)
        return dict(raw=r["raw"], sym=r["sym"])
    out = ctx.cached(key, run)
    ctx.fixtures["eng|" + "|".join(map(str, key[1:]))] = out["raw"]
    return out


REF_DA = {"S2": 0.01 ** 2, "S2K": 0.01 ** 2, "S2Kp": 0.01 ** 2, "S2C3": 0.01 ** 2 / 3}


def ref_dA(ctx, kname):
    """The reference's own area element for each named k-set (never the engine's dA)."""
    if kname in REF_DA:
        return REF_DA[kname]
    if kname == "S3":
        return ctx.ref.BZ_AREA / 60 ** 2
    if kname == "S3C3":
        return float(ctx.ref.bz_grid(60)[1])
    raise KeyError(kname)


def ref_eta(ctx, mat, model, func, var, B, kname, lam=None, c3=False):
    key = ("ref", mat, model, func, var, float(B), kname, lam, c3)

    def run():
        from refside import ref_eta as _r
        if kname == "S2":          # additivity: K + K' halves (also gives the single-valley scale)
            return ref_eta(ctx, mat, model, func, var, B, "S2K", lam, c3) + ref_eta(ctx, mat, model, func, var, B, "S2Kp", lam, c3)
        return _r(mat, model, func, B, omega(ctx, kname, mat, model, func, lam), nu_of(kname), var, kset(ctx, kname),
                  c3=c3, lam=lam, dA=ref_dA(ctx, kname))
    return ctx.cached(key, run)
