"""v7 check: grid convergence of the full-BZ maps at the points where the raw C3 violation of prod7 is largest
(WSe2 3.04 eV / 60 T and the vH energy 2.986 eV / 20, 60 T; WS2 2.622 eV / 57 T; MoS2 2.423 eV / 35 T).
eta (C3-symmetrized, physical) and raw C3 violation at N = 600, 900, 1200, TNN GGA, model F, nu = 10 meV.
Also the same at model ED (no velocity shift) and Z (Zeeman only) for WSe2 3.04 eV / 60 T, to locate the cause."""
import sys, os, json, hashlib, time, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here + "/../../pub_pipeline")
import lpge as L
F = dict(spin=True, mom=True, dos='v', vel=True); ED = dict(spin=True, mom=True, dos='v'); Z = dict(spin=True)
pts = [('WSe2', 3.039, 60.0, F), ('WSe2', 2.986, 60.0, F), ('WSe2', 2.986, 20.0, F), ('WS2', 2.622, 57.0, F), ('MoS2', 2.423, 35.0, F),
       ('WSe2', 3.039, 60.0, ED), ('WSe2', 3.039, 60.0, Z)]
out = []
for m, w, B, fl in pts:
    for N in (600, 900, 1200):
        t = time.time(); r = L.eta(m, "TNN", N, B, np.array([w]), 0.010, fl)
        rec = dict(m=m, omega=w, B=B, flags=fl, N=N, eta=float(-2*r['si_factor']*r['sym'][0, 0, 0, 0]),
                   eta_raw_xxx=float(-2*r['si_factor']*r['raw'][0, 0, 0, 0]), c3viol=float(r['c3_violation'][0]), sec=round(time.time()-t, 1))
        out.append(rec); print(rec, flush=True)
        json.dump(dict(points=out, lpge_sha256=hashlib.sha256(open(L.__file__, 'rb').read()).hexdigest()), open(here + "/c4_hotspots.json", "w"), indent=1)
