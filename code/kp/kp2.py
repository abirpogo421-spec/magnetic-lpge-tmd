#!/usr/bin/env python3
"""
Physical-units two-band k.p engine for the six group-VIB TMDs (Kormanyos 2015 params).

  Kormanyos et al., 2D Mater. 2, 022001 (2015) -- the only PUBLISHED source that
  tabulates k.p parameters (gap, Fermi velocity gamma, trigonal-warping kappa, effective
  masses) for ALL six monolayer TMDs. (Liu 2013 fits the full 3rd-order set for MoS2 only.)

Warped-Dirac d-vector (paper convention), PHYSICAL k in 1/Angstrom, valley tau, spin s:
    dx = v*tau*kx + kappa*(kx^2 - ky^2)
    dy = v*ky     - 2*kappa*tau*kx*ky
    dz = mD + beta*k^2 - tau*(b + s*lambda/2)
  v   = gamma          [eV*A]      (Dirac velocity)
  kappa                [eV*A^2]    (trigonal warping; NEGATIVE for all six)
  mD  = Delta/2        [eV]        (Dirac mass; can be anchored to the TB gap)
  beta = (hbar^2/4)(1/m_c + 1/|m_v|)  [eV*A^2]   (electron-hole-asymmetric k^2 term)
  b   = mu_B*B         [eV] ;  lambda = valence Ising-SOC splitting (from tmd_params)

LPGE eta^{x;xx} and shift sigma^{y;yy} use the SAME kernel/measure/prefactor as the TB
engine (Q = tr[Pv db Pc dc Pv], Delta^a = -d(gap)/dk_a, +pi LPGE / -2pi shift, C3-symmetrized),
summed over the spin-valley-locked active pair (tau,s) = (+,+) and (-,-).

IMPORTANT (see PHYSICS_VERIFICATION.md): this minimal 2-band k.p is QUALITATIVE. Because
kappa<0 for every material, the leading closed form eta ~ 6 kappa v^3 b / mD^4 gives the SAME
(negative) LPGE sign for all six, so it reproduces the full-TB band-edge sign only for
MoS2 / MoTe2 / WTe2 and gets MoSe2 / WS2 / WSe2 wrong. Use it as an illustrative closed form
(validated quantitatively for MoS2); use the TB survey for the per-material results.
"""
import numpy as np

HBAR2_2M = 3.80998    # hbar^2/(2 m_e)  [eV * Angstrom^2]

# Kormanyos 2015: Delta(eV), gamma(eV*A), kappa(eV*A^2), m_c(m_e), m_v(m_e, <0)
KORM = {
    'MoS2':  (1.67, 2.76, -1.48, 0.46, -0.54),
    'MoSe2': (1.40, 2.53, -1.31, 0.56, -0.59),
    'WS2':   (1.60, 3.34, -1.24, 0.26, -0.35),
    'WSe2':  (1.30, 3.17, -1.11, 0.28, -0.36),
    'MoTe2': (0.997, 2.33, -1.19, 0.62, -0.66),
    'WTe2':  (0.792, 3.04, -1.01, 0.26, -0.34),
}
# valence Ising-SOC splitting lambda (eV) -- same table as the TB model (tmd_params.LAMBDA)
LAMBDA = {'MoS2': 0.073, 'WS2': 0.211, 'MoSe2': 0.091, 'WSe2': 0.228, 'MoTe2': 0.107, 'WTe2': 0.237}

# window/broadening: match the TB engine if importable, else sane defaults
try:
    raise ImportError  # standalone: no window cut (WIN) so results do not depend on the TB engine
    NU = _tb.NU; WIN = _tb.WIN; C3 = _tb.C3_sym3
except Exception:
    NU = 0.01; WIN = 1e9
    def C3(T):  # 3-fold average of a [...,2,2,2] tensor (x,y indices)
        R = np.array([[np.cos(2*np.pi/3), -np.sin(2*np.pi/3)],
                      [np.sin(2*np.pi/3),  np.cos(2*np.pi/3)]])
        out = np.array(T, float)
        acc = np.array(T, float)
        Rc = R.copy()
        for _ in range(2):
            acc = acc + np.einsum('ia,jb,kc,...abc->...ijk', Rc, Rc, Rc, np.array(T, float))
            Rc = R @ Rc
        return acc/3.0


def params(mat, mD_from_gap=None):
    D, g, kap, mc, mv = KORM[mat]
    beta = 0.5*HBAR2_2M*(1.0/mc + 1.0/abs(mv))          # dz = mD + beta k^2
    mD = D/2 if mD_from_gap is None else mD_from_gap
    return dict(v=g, kappa=kap, beta=beta, mD=mD, lam=float(LAMBDA[mat]), Delta=D)


def _dv(KX, KY, tau, s, b, p):
    dx = p['v']*tau*KX + p['kappa']*(KX*KX - KY*KY)
    dy = p['v']*KY     - 2*p['kappa']*tau*KX*KY
    dz = p['mD'] + p['beta']*(KX*KX + KY*KY) - tau*(b + s*p['lam']/2)
    return np.stack([dx, dy, dz], -1)


def _P(dv):
    dd = np.linalg.norm(dv, axis=-1); n = dv/dd[..., None]
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    ns = np.zeros(dv.shape[:-1]+(2, 2), complex)
    ns[..., 0, 0] = nz; ns[..., 0, 1] = nx-1j*ny; ns[..., 1, 0] = nx+1j*ny; ns[..., 1, 1] = -nz
    I = np.broadcast_to(np.eye(2, dtype=complex), ns.shape)
    return 0.5*(I-ns), 0.5*(I+ns), dd


def response(mat, b, om, p=None, R=0.35, Nk=131, nu=None, want_shift=True, pairs=((+1, +1), (-1, -1)), full=False):
    """Return (LPGE eta^{x;xx}(om), shift sigma^{y;yy}(om)) for the K+K' active pair. Arb. units."""
    if p is None:
        p = params(mat)
    nu = NU if nu is None else nu; half = WIN*nu
    ks = np.linspace(-R, R, Nk); KX, KY = np.meshgrid(ks, ks, indexing='ij')
    m = (KX*KX+KY*KY) <= R*R; dA = (ks[1]-ks[0])**2; MEAS = dA/(2*np.pi)**2; e = 1e-3
    mm = lambda A, B: np.einsum('...ij,...jk->...ik', A, B); tr = lambda M: np.einsum('...ii->...', M)
    inj = np.zeros((len(om), 2, 2, 2)); lin = np.zeros((len(om), 2, 2, 2))
    for (tau, s) in pairs:
        def PP(ox, oy): return _P(_dv(KX+ox*e, KY+oy*e, tau, s, b, p))
        Pv, Pc, dd = PP(0, 0)
        Pvxp, Pcxp, ddxp = PP(1, 0); Pvxm, Pcxm, ddxm = PP(-1, 0)
        Pvyp, Pcyp, ddyp = PP(0, 1); Pvym, Pcym, ddym = PP(0, -1)
        dPc = [(Pcxp-Pcxm)/(2*e), (Pcyp-Pcym)/(2*e)]; dPv = [(Pvxp-Pvxm)/(2*e), (Pvyp-Pvym)/(2*e)]
        gap = 2*dd; Da = [-2*(ddxp-ddxm)/(2*e), -2*(ddyp-ddym)/(2*e)]
        Q = np.zeros(KX.shape+(2, 2), complex)
        for bb in range(2):
            for cc in range(2):
                Q[..., bb, cc] = tr(mm(mm(Pv, dPc[bb]), dPv[cc]))
        injpl = np.stack([Da[0][..., None, None]*Q, Da[1][..., None, None]*Q], axis=-3).real
        ec = gap[m]; ij = injpl[m]
        Lor = (nu/np.pi)/((om[:, None]-ec[None, :])**2 + nu**2)
        Lor = np.where(np.abs(om[:, None]-ec[None, :]) < half, Lor, 0.0)
        inj += np.einsum('ok,kabc->oabc', Lor, ij)  # measure applied once below (MEAS); kp_phys.py applied dA twice
        if want_shift:
            Pvpp, _, _ = PP(1, 1); Pvpm, _, _ = PP(1, -1); Pvmp, _, _ = PP(-1, 1); Pvmm, _, _ = PP(-1, -1)
            Pcpp = PP(1, 1)[1]; Pcpm = PP(1, -1)[1]; Pcmp = PP(-1, 1)[1]; Pcmm = PP(-1, -1)[1]
            ddPv = [[(Pvxp-2*Pv+Pvxm)/e**2, (Pvpp-Pvpm-Pvmp+Pvmm)/(4*e**2)],
                    [(Pvpp-Pvpm-Pvmp+Pvmm)/(4*e**2), (Pvyp-2*Pv+Pvym)/e**2]]
            ddPc = [[(Pcxp-2*Pc+Pcxm)/e**2, (Pcpp-Pcpm-Pcmp+Pcmm)/(4*e**2)],
                    [(Pcpp-Pcpm-Pcmp+Pcmm)/(4*e**2), (Pcyp-2*Pc+Pcym)/e**2]]
            C = np.zeros(KX.shape+(2, 2, 2), complex)
            for a in range(2):
                for bb in range(2):
                    for cc in range(2):
                        Cmn = tr(mm(mm(Pv, dPc[bb]), ddPv[a][cc]+mm(dPc[a], dPv[cc])))
                        Cnm = tr(mm(mm(Pc, dPv[bb]), ddPc[a][cc]+mm(dPv[a], dPc[cc])))
                        C[..., a, bb, cc] = 0.5*(Cmn-Cnm)
            Csym = 0.5*(C + np.swapaxes(C, -1, -2))
            li = Csym.imag[m]
            lin += np.einsum('ok,kabc->oabc', Lor, li)
    LPGE = C3(inj*np.pi*MEAS); LPGE = 0.5*(LPGE+np.swapaxes(LPGE, -1, -2))
    SHIFT = C3(lin*(-2*np.pi)*MEAS)
    if full:
        return LPGE, SHIFT
    return LPGE[:, 0, 0, 0], SHIFT[:, 1, 1, 1]


def save_all(outroot, materials=None, om=None, fields=(0.0, 0.02, 0.08), mD_from_gap=None):
    """Write kp/<MAT>_kp_spectra.npz for each material: LPGE & shift vs omega at a few fields,
    plus the k.p params. mD_from_gap: optional dict {mat: TB_gap} to anchor the onset to the TB gap."""
    import os
    materials = materials or list(KORM)
    od = os.path.join(outroot, "kp"); os.makedirs(od, exist_ok=True)
    for mat in materials:
        g = None if mD_from_gap is None else mD_from_gap.get(mat)
        p = params(mat, mD_from_gap=(g/2 if g else None))
        if om is None:
            lo = (g or p['Delta']) - 0.05; hi = (g or p['Delta']) + 1.2
            omg = np.linspace(lo, hi, 200)
        else:
            omg = om
        L = {}; S = {}
        for b in fields:
            L[b], S[b] = response(mat, float(b), omg, p)
        np.savez_compressed(os.path.join(od, f"{mat}_kp_spectra.npz"),
                            omega=omg, fields=np.array(fields),
                            LPGE=np.array([L[b] for b in fields]),
                            SHIFT=np.array([S[b] for b in fields]),
                            v=p['v'], kappa=p['kappa'], beta=p['beta'], mD=p['mD'],
                            lam=p['lam'], Delta=p['Delta'],
                            README=("two-band k.p (Kormanyos 2015), arb units. LPGE[i]/SHIFT[i] vs omega "
                                    "at fields[i]=mu_B B (eV). QUALITATIVE: kappa<0 for all -> LPGE sign "
                                    "matches full TB only for MoS2/MoTe2/WTe2. See PHYSICS_VERIFICATION.md."))
        print(f"  wrote kp/{mat}_kp_spectra.npz  (v={p['v']}, kappa={p['kappa']}, mD={p['mD']:.3f})", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Two-band k.p spectra (Kormanyos) for the six TMDs.")
    ap.add_argument("--outroot", default=".")
    ap.add_argument("--materials", nargs="*", default=None)
    a = ap.parse_args()
    print("Two-band k.p (Kormanyos 2015) spectra -> <outroot>/kp/*_kp_spectra.npz")
    save_all(a.outroot, materials=a.materials)
    print("done.")
