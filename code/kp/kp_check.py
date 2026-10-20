"""Cross-check: two-band k.p (TB-derived exact small-q parameters, kp2.response) vs the full TB (lpge.eta, Z field model)
for chi = d eta^{x;xx}/dB at hbar*omega = Eg + {0.02, 0.05, 0.10, 0.20} eV.  Same code units (k in 1/a)."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "pub_pipeline")); sys.path.insert(0, HERE)
import lpge as L, tb_core as tb, kp2
PF = sys.argv[1] if len(sys.argv) > 1 else "kp_params_TNN.json"; P = json.load(open(os.path.join(HERE, PF)))
B = 30.0; OFF = np.array([0.02, 0.05, 0.10, 0.20]); NU = 0.005; out = {}
for mat, p in P.items():
    b = tb.MU_B*B
    Eg = 2*p['mD'] - p['lam']                       # active-pair gap of the k.p model
    om = Eg + OFF
    kp = [kp2.response(mat, s*b, om, p=p, R=0.6, Nk=601, nu=NU, want_shift=False)[0] for s in (1, -1)]
    chi_kp = (kp[0]-kp[1])/(2*B)
    tb.set_config(mat, "GGA")
    q = np.linspace(-0.05, 0.05, 201); g = []
    for blk in L._H6(4*np.pi/3+q, 0*q, "TNN", 0.0):
        e = np.linalg.eigvalsh(blk); g.append((e[:, 1]-e[:, 0]).min())
    EgTB = min(g); kpts = L.local_kpts(0.6, 0.002)
    r = [L.eta(mat, "TNN", None, s*B, EgTB+OFF, NU, dict(spin=True), kpts=kpts)['sym'][:, 0, 0, 0] for s in (1, -1)]
    chi_tb = (r[0]-r[1])/(2*B)
    out[mat] = dict(Eg_kp=Eg, Eg_tb=EgTB, chi_kp=chi_kp.tolist(), chi_tb=chi_tb.tolist())
    print(f"{mat}: Eg kp={Eg:.4f} TB={EgTB:.4f}")
    for o, a_, c_ in zip(OFF, chi_kp, chi_tb):
        print(f"   Eg+{o:.2f}: chi_kp={a_:+.3e}  chi_TB={c_:+.3e}  ratio={a_/c_:+.2f}")
json.dump(out, open(os.path.join(HERE, "kp_check_" + PF), "w"), indent=1)
