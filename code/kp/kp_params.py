"""Two-band k.p parameters derived from the Liu TB by Lowdin reduction at K (tau=+1, lambda=0, B=0).
Lengths in units of a (k in 1/a), energies in eV.  d-vector convention of kp2.py:
  dx = v tau kx + kappa (kx^2-ky^2),  dy = v ky - 2 kappa tau kx ky,  dz = mD + w k^2 - tau (b + s lambda/2)
Mapping from the Liu third-order form (c, v diagonal; t, gamma3 off-diagonal):
  v = t,  kappa = gamma3/2,  w = (g1 - g2)/4 (bare diagonal curvatures, Dirac part NOT included),  mD = (Ec - Ev)/2
Source fit: /home/claude/eng/tests/kp_vs_tb.py fit() (validated: 'D+k+m' reproduces the TB chi of MoS2 to ~5%)."""
import json, sys
sys.path.insert(0, "/home/claude/eng/tmdsurvey2"); import tb_core as tb

def tb_kp(mat, model="TNN", soc=False):
    """Exact small-q Taylor coefficients of the Lowdin-reduced 2x2 H at K (fit on rings r = 0.002-0.01 /a,
    where the O(q^3) terms are < 1e-5 of the retained ones). Reproduces Table II (v, kappa) of the paper."""
    sys.path.insert(0, "/home/claude/eng/tests")
    from kp_vs_tb import lowdinK
    import numpy as np
    tb.set_config(mat, "GGA")
    r = np.repeat([0.002, 0.004, 0.006, 0.01], 64); ph = np.tile(np.linspace(0, 2*np.pi, 65)[:-1], 4)
    qx, qy = r*np.cos(ph), r*np.sin(ph); zp, zm = qx+1j*qy, qx-1j*qy; q2 = r*r; w3 = qx*(qx**2-3*qy**2)
    lam0 = float(tb.LAM) if soc else 0.0
    H = lowdinK(qx[None], qy[None], model, 1, 1 if soc else 0, lam0, 0.0)[0]
    Xo = np.stack([zm, zp**2/2, q2*zm, zp, zm**2], 1); co = np.linalg.lstsq(Xo, H[:, 0, 1], rcond=None)[0]
    Xd = np.stack([np.ones_like(qx), q2/2, w3, q2**2], 1)
    cc = np.linalg.lstsq(Xd, H[:, 0, 0].real, rcond=None)[0]; cv = np.linalg.lstsq(Xd, H[:, 1, 1].real, rcond=None)[0]
    return dict(v=float(abs(co[0])), kappa=float(co[1].real)/2, beta=float(cc[1]-cv[1])/4,
                mD=float(cc[0]-cv[0])/2, lam=0.0 if soc else float(tb.LAM), Delta=float(cc[0]-cv[0]), a_A=float(tb.A_PHYS*1e10),
                mc_bare=float(cc[1]), mv_bare=float(cv[1]))

if __name__ == "__main__":
    for soc, fn in ((False, "kp_params_TNN.json"), (True, "kp_params_TNN_soc.json")):
        out = {m: tb_kp(m, soc=soc) for m in ("MoS2", "MoSe2", "WS2", "WSe2")}
        for m, p in out.items():
            print("SOC" if soc else "noSOC", m, {k: round(x, 4) for k, x in p.items()}, " kappa eV A^2 = %.3f" % (p['kappa']*p['a_A']**2))
        json.dump(out, open(fn, "w"), indent=1)
