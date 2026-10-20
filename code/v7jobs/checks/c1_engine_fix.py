"""Check the v7 engine (exact Lorentzian sum) against X3's independent exact values (verification 2026-10-13).
Expect: MoSe2 TNN, lambda0, Eg+0.1 eV, R=0.6/a, h=0.002/a, nu=5 meV, chi from +-10 T: -2.250e-5 A m V^-2 s^-1 T^-1.
Expect: MoSe2 TNN threshold Eg+0.01, R=0.6/a, h=0.001/a, 30 T: chi = +1.904e-5 (no cut)."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../pub_pipeline")
import lpge as L, tb_core as tb
FL = dict(spin=True, mom=True, dos='v', vel=True)
def gap(mat, model="TNN"):
    tb.set_config(mat, "GGA"); q = np.linspace(-0.05, 0.05, 401); g = []
    for blk in L._H6(4*np.pi/3+q, 0*q, model, 0.0):
        e = np.linalg.eigvalsh(blk); g.append((e[:, 1]-e[:, 0]).min())
    return float(min(g))
Eg = gap("MoSe2")
t = time.time(); kp = L.local_kpts(0.6, 0.002)
p = L.eta("MoSe2", "TNN", None, 10.0, np.array([Eg+0.1]), 0.005, FL, kpts=kp)
m = L.eta("MoSe2", "TNN", None, -10.0, np.array([Eg+0.1]), 0.005, FL, kpts=kp)
chi = -2*p['si_factor']*0.5*(p['sym'][0,0,0,0]-m['sym'][0,0,0,0])/10
print("lam0 Eg+0.1 chi", chi, "expect -2.250e-5", "time", round(time.time()-t,1), "npts", len(kp[0]), flush=True)
t = time.time(); kp = L.local_kpts(0.6, 0.001)
r = L.eta("MoSe2", "TNN", None, 30.0, np.array([Eg+0.01]), 0.005, FL, kpts=kp)
print("thr chi", -2*r['si_factor']*r['sym'][0,0,0,0]/30, "expect +1.904e-5", "time", round(time.time()-t,1), "npts", len(kp[0]), flush=True)
