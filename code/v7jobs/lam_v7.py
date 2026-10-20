"""v7 (2026-10-14): lambda sweep for Fig. 12 with the fixed engine and disks of radius 0.9/a (same as Figs. 9-11).
Original docstring: Band-edge runs on a fine local grid around K, K' (radius R=0.6/a).
usage: python3 run_edge.py MAT VARIANT MODE
MODE B   : eta(omega = Eg + offsets, B = 0..60 T step 3) at lambda = Liu value       -> edge_{MAT}_{VAR}.npz
MODE lam : chi = eta/B from B = +-10 T (odd part) vs SOC lambda in [0, 0.4] eV     -> lam_{MAT}_{VAR}.npz
Grid h = 0.002/a with nu = 5 meV (converged to a few % against h=0.001, nu=2 meV in check A8)."""
import sys, os, json, time, hashlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../pub_pipeline")
import lpge as L, tb_core as tb

mat, var, mode = sys.argv[1], sys.argv[2], sys.argv[3]
FLAGS = {'Z': dict(spin=True, mom=False, dos='none', vel=False),
         'F': dict(spin=True, mom=True, dos='v', vel=True)}[var]
H, NU, R = 0.002, 0.005, 0.9
OFF = np.array([0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30])
kp = L.local_kpts(R, H)
sha = hashlib.sha256(open(L.__file__, 'rb').read()).hexdigest()

def gap(lam=None):
    tb.set_config(mat, "GGA")
    if lam is not None: tb.LAM = lam
    q = np.linspace(-0.05, 0.05, 201); g = []
    for blk in L._H6(4*np.pi/3+q, 0*q, "TNN", 0.0):
        e = np.linalg.eigvalsh(blk); g.append((e[:, 1]-e[:, 0]).min())
    return min(g)

meta = dict(material=mat, variant=var, flags=FLAGS, grid=f"local disks R={R}/a, h={H}/a", nu_eV=NU,
            model="Liu TNN GGA", offsets_eV=OFF.tolist(), lpge_sha256=sha)
if mode == "B":
    Bs = np.arange(0, 60.01, 3.0); Eg = gap(); om = Eg+OFF
    eta = np.zeros((len(Bs), len(OFF), 2, 2, 2)); viol = np.zeros((len(Bs), len(OFF)))
    for i, B in enumerate(Bs):
        t = time.time(); r = L.eta(mat, "TNN", None, B, om, NU, FLAGS, kpts=kp)
        eta[i], viol[i] = r['sym'], r['c3_violation']
        np.savez(f"edgeB7_{mat}_{var}.npz", B_tesla=Bs, Eg=Eg, omega=om, eta=eta, c3_violation=viol,
                 meta=json.dumps(dict(meta, Eg=Eg, si_factor=r['si_factor'], done=i+1)))
        print(f"{mat} {var} edge B={B:4.0f}T {time.time()-t:.0f}s chi_xxx(Eg+.1)={r['sym'][4,0,0,0]:+.3e}", flush=True)
else:
    tb.set_config(mat, "GGA"); lam0 = tb.LAM
    lams = np.unique(np.round(np.concatenate([np.linspace(0.5*lam0, 1.5*lam0, 21), np.linspace(0, 0.4, 21)]), 6)); chi = np.zeros((len(lams), len(OFF))); Egs = np.zeros(len(lams))
    for i, lam in enumerate(lams):
        t = time.time(); Eg = gap(lam); Egs[i] = Eg; om = Eg+OFF
        p = L.eta(mat, "TNN", None, 10.0, om, NU, FLAGS, kpts=kp, lam=lam)
        m = L.eta(mat, "TNN", None, -10.0, om, NU, FLAGS, kpts=kp, lam=lam)
        chi[i] = 0.5*(p['sym'][:, 0, 0, 0]-m['sym'][:, 0, 0, 0])/10.0*p['si_factor']
        np.savez(f"lam7_{mat}_{var}.npz", lam=lams, Eg=Egs, chi_SI_per_T=chi,
                 meta=json.dumps(dict(meta, Bfit="odd part of +-10 T", done=i+1, lam_liu=lam0)))
        print(f"{mat} {var} lam={lam:.2f} Eg={Eg:.3f} {time.time()-t:.0f}s chi(Eg+.05)={chi[i,2]:+.3e}", flush=True)
