"""v7 check: full-BZ mesh convergence of the small-field slope chi = eta(1 T)/(1 T) at the van Hove energy (from prod7 maps),
N = 600 (production) vs N = 900, TNN GGA, model F, nu = 10 meV. Fixed engine. Output c2_N_convergence.json."""
import sys, os, json, hashlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../pub_pipeline")
import lpge as L
FL = dict(spin=True, mom=True, dos='v', vel=True); out = {}
for m in ['MoS2', 'MoSe2', 'WS2', 'WSe2']:
    d = np.load(os.path.dirname(os.path.abspath(__file__)) + f"/../prod7_{m}_F.npz", allow_pickle=True)
    w = float(d['omega'][np.argmax(np.abs(d['eta'][-1]))])
    r = {}
    for N in (600, 900):
        q = L.eta(m, "TNN", N, 1.0, np.array([w]), 0.010, FL); r[N] = -2*q['si_factor']*q['sym'][0, 0, 0, 0]
    out[m] = dict(vH_eV=w, chi600=r[600], chi900=r[900], rel=(r[600]-r[900])/r[900]); print(m, out[m], flush=True)
out['meta'] = dict(lpge_sha256=hashlib.sha256(open(L.__file__, 'rb').read()).hexdigest(), B_T=1.0, nu=0.01)
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/c2_N_convergence.json", "w"), indent=1)
