"""chi(omega) near the absorption edge: eta^{x;xx}(B=30 T)/30 T on a dense omega grid.
eta is exactly odd in B (T-paired disks), so chi = eta(B)/B; the response is linear within 2% to 60 T near the edge.
Usage: python3 chiw9_v7.py MAT MODEL VARIANT H   (v7: fixed engine, writes engine hash)"""
import sys, os, json, hashlib, datetime, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../pub_pipeline")
import lpge as L, tb_core as tb
m, model, var, h = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
FL = {'Z': dict(spin=True), 'F': dict(spin=True, mom=True, dos='v', vel=True)}[var]
tb.set_config(m, 'GGA'); q = np.linspace(-0.05, 0.05, 401); g = []
for blk in L._H6(4*np.pi/3 + q, 0*q, model, 0.0):
    e = np.linalg.eigvalsh(blk); g.append((e[:, 1] - e[:, 0]).min())
Eg = float(min(g)); off = np.arange(-0.03, 0.4501, 0.001); om = Eg + off
B = 30.0; r = L.eta(m, model, None, B, om, 0.005, FL, kpts=L.local_kpts(0.9, h))
chi = (-2.0*r['si_factor']*r['sym'][:, 0, 0, 0]/B).tolist()
json.dump(dict(Eg=Eg, off=off.tolist(), chi=chi, h=h, R=0.9, nu=0.005, B=B, model=model, flags=FL, lpge_sha256=hashlib.sha256(open(L.__file__, "rb").read()).hexdigest(), script="chiw9_v7.py", date=datetime.datetime.now().isoformat()), open(f"chiwR9v7_{m}_{model}_{var}_{h}.json", "w"))
print("done", m, model, var, Eg, flush=True)
