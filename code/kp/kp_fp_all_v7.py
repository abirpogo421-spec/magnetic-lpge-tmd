"""v7 (2026-10-14): threshold chi from the two-band k.p with first-principles (Kormanyos et al., 2D Mater. 2, 022001 (2015))
parameters for the BAND-EDGE (A) spin pair. Fixes verification report V-01:
  - kappa from the kappa_down column of Table 6 (DFT gap): Kormanyos p.20 "in the VB the upper index (1) ((2)) is equivalent to down (up)".
  - masses of the A pair (Table 3/4 band-edge values): m_c = 0.43, 0.49, 0.26, 0.28; m_v = -0.54, -0.59, -0.35, -0.36.
  - gap: E_A is the A-transition gap (Ebg,K for MoX2; Ebg,K + |2 Delta_cb| for WX2), already including SOC, so
    m_D = (E_A + lambda)/2 and the active mass m = m_D - lambda/2 = E_A/2 (paper Sec. II C definitions).
  - w = (hbar^2/4m_e)(1/m_c + 1/|m_v|) - v^2/(2m) = ... - v^2/E_A (paper relation, m = active mass).
Units: k in 1/A; SI = code * 1e-10 * e^3/hbar^2. chi_phys = -2 x [eta_code(+B) - eta_code(-B)]/(2B) x SI at B = 30 T (physical sign, same -2 convention as the TB engine and fig_main.py PHYS).
Old (wrong) inputs kept in kp_fp_all.py for comparison."""
import sys, os, json, datetime, hashlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import kp2
MUB = 5.7883818060e-5; HB2M = 3.80998; SI = 1e-10*1.602176634e-19**3/1.054571817e-34**2
#            E_A    gamma  kappa_dn  m_c   m_v    lambda(Liu)
KORM = {'MoS2': (1.670, 2.76, -1.45, 0.43, -0.54, 0.073), 'MoSe2': (1.400, 2.53, -1.23, 0.49, -0.59, 0.091),
        'WS2': (1.632, 3.34, -1.09, 0.26, -0.35, 0.211), 'WSe2': (1.337, 3.17, -0.93, 0.28, -0.36, 0.228)}
OFF = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.30]); out = {}
for m, (EA, g, k, mc, mv, lam) in KORM.items():
    mD = (EA + lam)/2; w = 0.5*HB2M*(1/mc + 1/abs(mv)) - g*g/EA
    p = dict(v=g, kappa=k, beta=w, mD=mD, lam=lam)
    Eg = 2*mD - lam; om = Eg + OFF; b = MUB*30
    c = [(kp2.response(m, s*b, om, p=p, R=0.2, Nk=401, nu=0.005, want_shift=False)[0]) for s in (1, -1)]
    chi = -2.0*(c[0] - c[1])/60*SI
    out[m] = dict(params=p, E_A=EA, offsets_eV=OFF.tolist(), chi_phys_SI_per_T=chi.tolist())
    print(m, "Eg=%.3f w=%.3f eV A^2" % (Eg, w), "chi(Eg+" + ",".join(f"{o:g}" for o in OFF) + ") =", np.array2string(chi*1e3, precision=3), 'e-3', flush=True)
out['meta'] = dict(script='kp_fp_all_v7.py', date=datetime.datetime.now().isoformat(), grid='R=0.2/A Nk=401 nu=5 meV', B_T=30,
                   kp2_sha256=hashlib.sha256(open(kp2.__file__, 'rb').read()).hexdigest(), units='A m V^-2 s^-1 T^-1')
json.dump(out, open("kp_fp_all_v7.json", "w"), indent=1)
