"""Fig. 8 rerun: signed eta^{x;xx}(hbar omega, B), TNN, GGA, full field model F.
Window: 0.95*E_g to 1.05*(largest interband energy over the BZ). More field points (dB = 1 T) instead of more k/omega points.
usage: python3 maps_v7.py MAT [N=600] [NOM=900] [DB=1.0] [NU=0.010]
Resumes from prod7_{MAT}_F.npz if present (one B value per save). Only B >= 0 is computed; eta(-B) = -eta(B) by time reversal."""
import sys, os, json, time, hashlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../pub_pipeline")
import lpge as L, tb_core as tb
mat = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
NOM = int(sys.argv[3]) if len(sys.argv) > 3 else 900; DB = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
nu = float(sys.argv[5]) if len(sys.argv) > 5 else 0.010
FL = dict(spin=True, mom=True, dos='v', vel=True)
tb.set_config(mat, "GGA")
q = np.linspace(-0.05, 0.05, 401); g = []
for blk in L._H6(4*np.pi/3 + q, 0*q, "TNN", 0.0):
    e = np.linalg.eigvalsh(blk); g.append((e[:, 1] - e[:, 0]).min())
Eg = float(min(g))
kx, ky = np.meshgrid(np.linspace(-2*np.pi, 2*np.pi, 401), np.linspace(-2*np.pi, 2*np.pi, 401))
Emax = max(float((lambda e: (e[:, 2] - e[:, 0]).max())(np.linalg.eigvalsh(blk))) for blk in L._H6(kx.ravel(), ky.ravel(), "TNN", 0.0))
om = np.linspace(0.95*Eg, 1.05*Emax, NOM); Bs = np.arange(0, 60 + 1e-9, DB)
fn = f"prod7_{mat}_F.npz"; sha = hashlib.sha256(open(L.__file__, 'rb').read()).hexdigest()
eta = np.zeros((len(Bs), NOM)); viol = np.zeros((len(Bs), NOM)); done = 0
if os.path.exists(fn):
    d = np.load(fn, allow_pickle=True); m0 = json.loads(str(d['meta']))
    if m0['N'] == N and len(d['omega']) == NOM and np.allclose(d['B_tesla'], Bs): eta, viol, done = d['eta'], d['c3_violation'], m0['done']
for i in range(done, len(Bs)):
    t = time.time(); r = L.eta(mat, "TNN", N, float(Bs[i]), om, nu, FL)
    eta[i] = -2*r['si_factor']*r['sym'][:, 0, 0, 0]; viol[i] = r['c3_violation']
    meta = dict(material=mat, flags=FL, N=N, nu_eV=nu, model="Liu TNN GGA", Eg=Eg, Emax_interband=Emax, done=i+1,
                units="eta in A m V^-2 s^-1 (physical sign)", lpge_sha256=sha, Bsym="eta(-B) = -eta(B)")
    np.savez(fn, B_tesla=Bs, omega=om, eta=eta, c3_violation=viol, meta=json.dumps(meta))
    print(f"{mat} B={Bs[i]:5.1f} T  {time.time()-t:.0f} s", flush=True)
print("done", mat)
