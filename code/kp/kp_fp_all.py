"""chi from the two-band k.p with first-principles (Kormanyos 2015, DFT) parameters, all four materials.
Units: k in 1/A -> code units are Angstrom; SI = code * 1e-10 * e^3/hbar^2."""
import os, sys, json, numpy as np; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import kp2
MUB = 5.7883818060e-5; HB2M = 3.80998; SI = 1e-10*1.602176634e-19**3/1.054571817e-34**2
KORM = {'MoS2': (1.67, 2.76, -1.48, 0.46, -0.54, 0.073), 'MoSe2': (1.40, 2.53, -1.31, 0.56, -0.59, 0.091),
        'WS2': (1.60, 3.34, -1.24, 0.26, -0.35, 0.211), 'WSe2': (1.30, 3.17, -1.11, 0.28, -0.36, 0.228)}
out = {}
for m, (D, g, k, mc, mv, lam) in KORM.items():
    mD = D/2; p = dict(v=g, kappa=k, beta=0.5*HB2M*(1/mc+1/abs(mv))-g*g/(2*mD), mD=mD, lam=lam)
    Eg = 2*mD-lam; om = Eg+np.array([0.02, 0.05, 0.10, 0.20]); b = MUB*30
    c = [(kp2.response(m, s*b, om, p=p, R=0.2, Nk=401, nu=0.005, want_shift=False)[0]) for s in (1, -1)]
    chi = (c[0]-c[1])/60*SI; out[m] = dict(params=p, chi_SI_per_T=chi.tolist())
    print(m, "w=%.2f eV A^2" % p['beta'], "chi(Eg+.02,.05,.1,.2) =", np.array2string(chi, precision=2))
json.dump(out, open("kp_fp_all.json", "w"), indent=1)
