"""Fig. 9 rerun (and threshold chi for Fig. 11): eta^{x;xx}(B) at hbar omega = E_g + 0.01 eV.
Finer local grid (spacing 0.0007/a, converged to <1% vs 0.001/a), disks of radius 0.9/a (same as Fig. 10),
eta averaged over 5 omega points in [E_g+0.0075, E_g+0.0125] eV to remove grid wiggle, B = 0..60 T in 2 T steps.
usage: python3 edge_v7.py MAT MODEL [H=0.0007] [DB=2] [FUNC=GGA]   (MODEL = TNN or NN; FUNC = GGA or LDA)
Resumes from edge7_{MAT}_{MODEL}.json. chi = slope of eta = s B + c B^3 fitted over |B| <= 60 T."""
import sys, os, json, time, hashlib, datetime, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../pub_pipeline")
import lpge as L, tb_core as tb
mat, model = sys.argv[1], sys.argv[2]; h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0007
DB = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
FUNC = sys.argv[5] if len(sys.argv) > 5 else 'GGA'
SHA = hashlib.sha256(open(L.__file__, 'rb').read()).hexdigest()
FL = dict(spin=True, mom=True, dos='v', vel=True); nu = 0.005
tb.set_config(mat, FUNC)
q = np.linspace(-0.05, 0.05, 401); g = []
for blk in L._H6(4*np.pi/3 + q, 0*q, model, 0.0):
    e = np.linalg.eigvalsh(blk); g.append((e[:, 1] - e[:, 0]).min())
Eg = float(min(g)); om = Eg + 0.01 + np.linspace(-0.0025, 0.0025, 5); kl = L.local_kpts(0.9, h)
fn = f"edge7_{mat}_{model}.json" if FUNC == "GGA" else f"edge7_{mat}_{model}_{FUNC}.json"
out = json.load(open(fn)) if os.path.exists(fn) else dict(Eg=Eg, h=h, R=0.9, nu=nu, omega=om.tolist(), eta={}, functional=FUNC, model=model, flags=FL, lpge_sha256=SHA, script='edge_v7.py', started=datetime.datetime.now().isoformat())
for B in np.arange(0, 60 + 1e-9, DB):
    k = f"{B:g}"
    if k in out['eta']: continue
    assert out.get('lpge_sha256') == SHA, 'engine changed since this file was started'
    t = time.time(); r = L.eta(mat, model, None, float(B), om, nu, FL, functional=FUNC, kpts=kl)
    out['eta'][k] = (-2*r['si_factor']*r['sym'][:, 0, 0, 0]).tolist(); json.dump(out, open(fn, 'w'))
    print(mat, model, k, float(np.mean(out['eta'][k])), round(time.time() - t, 1), flush=True)
B = np.array(sorted(map(float, out['eta']))); Y = np.array([np.mean(out['eta'][f"{b:g}"]) for b in B])
Bf = np.concatenate([-B[:0:-1], B]); Yf = np.concatenate([-Y[:0:-1], Y])
s, c = np.linalg.lstsq(np.vstack([Bf, Bf**3]).T, Yf, rcond=None)[0]
out['chi_SI_per_T'] = float(s); out['finished'] = datetime.datetime.now().isoformat(); out['dev_60T_pct'] = float(100*c*60**2/s); json.dump(out, open(fn, 'w'))
print("chi", mat, model, s, "deviation at 60 T (%)", out['dev_60T_pct'])
