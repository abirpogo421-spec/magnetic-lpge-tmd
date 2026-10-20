"""v7 check: local-spacing convergence at threshold. eta^{x;xx} at B = 30 T, 5-point average over Eg+0.0075..0.0125 eV,
R = 0.9/a, nu = 5 meV, spacing h = 0.001/a, compared with the production h = 0.0007/a values stored in edge7_{m}_{model}.json
(key "30"). Fixed engine. Output c3_h_convergence.json."""
import sys, os, json, hashlib, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here + "/../../pub_pipeline")
import lpge as L, tb_core as tb
FL = dict(spin=True, mom=True, dos='v', vel=True); out = {}; kl = L.local_kpts(0.9, 0.001)
for m in ['MoS2', 'MoSe2', 'WS2', 'WSe2']:
    for model in ['TNN', 'NN']:
        e7 = json.load(open(here + f"/../edge7_{m}_{model}.json")); om = np.array(e7['omega'])
        r = L.eta(m, model, None, 30.0, om, 0.005, FL, kpts=kl)
        a = float(np.mean(-2*r['si_factor']*r['sym'][:, 0, 0, 0])); b = float(np.mean(e7['eta']['30']))
        out[f"{m}|{model}"] = dict(h001=a, h0007=b, rel=(a-b)/b); print(m, model, out[f"{m}|{model}"], flush=True)
        json.dump(out, open(here + "/c3_h_convergence.json", "w"), indent=1)
out['meta'] = dict(lpge_sha256=hashlib.sha256(open(L.__file__, 'rb').read()).hexdigest(), B_T=30, R=0.9, nu=0.005)
json.dump(out, open(here + "/c3_h_convergence.json", "w"), indent=1)
