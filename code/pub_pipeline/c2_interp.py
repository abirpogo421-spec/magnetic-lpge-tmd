"""Causal test C2: interpolate H(s) = (1-s) H_NN + s H_TNN (same material) and track kappa(s) and the edge chi(s).
Pre-registered: if the near-edge sign is set by kappa, sign changes of kappa(s) and chi_edge(s) coincide."""
import sys, os, json, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_core as tb, lpge
S = None
def H3mix(model, al, be):
    return (1-S)*tb.H_NN_np(al, be, tb.P_NN) + S*tb.H_TNN_np(al, be, tb.P_TNN)
tb.make_H3 = H3mix
def gamma3():
    K = np.array([4*np.pi/3, 0.0])
    Hf = lambda k: H3mix(None, 0.5*k[:, 0][:, None], (np.sqrt(3)/2)*k[:, 1][:, None])[:, 0]
    E, V = np.linalg.eigh(Hf(K[None])[0])
    c = int(np.argmax(abs(V[0, :])**2)); v = [i for i in range(3) if i != c and np.vdot(V[:, i], tb.LZ@V[:, i]).real > 1][0]
    r = np.linspace(0.01, 0.08, 8); ph = np.linspace(0, 2*np.pi, 25)[:-1]; R, P = np.meshgrid(r, ph)
    qx = (R*np.cos(P)).ravel(); qy = (R*np.sin(P)).ravel()
    Hb = np.einsum('ia,kij,jb->kab', V.conj(), Hf(K[None]+np.stack([qx, qy], 1)), V)
    u = [i for i in range(3) if i not in (c, v)][0]
    Heff = Hb[:, c, v] + Hb[:, c, u]*Hb[:, u, v]*0.5*(1/(E[c]-E[u])+1/(E[v]-E[u]))
    zp = qx+1j*qy; zm = qx-1j*qy; q2 = qx**2+qy**2
    X = np.stack([zm, zp**2/2, q2*zm/3, zp, zm**2], 1); co, *_ = np.linalg.lstsq(X, Heff, rcond=None)
    co = co*np.conj(co[0])/abs(co[0]); return co[1].real
mat = sys.argv[1]; out = {}
tb.set_config(mat, 'GGA')
for s in np.linspace(0, 1, 21):
    S = float(s)
    kx, ky, dA = lpge.local_kpts(0.05, 0.002); up, dn = lpge._H6(kx, ky, None, 0.0)
    Eg = min((np.linalg.eigvalsh(up)[:, 1]-np.linalg.eigvalsh(up)[:, 0]).min(), (np.linalg.eigvalsh(dn)[:, 1]-np.linalg.eigvalsh(dn)[:, 0]).min())
    om = np.array([Eg+0.05, Eg+0.10]); kp = lpge.local_kpts(0.6, 0.004)
    a = lpge.eta(mat, None, 0, 30.0, om, 0.01, dict(spin=True), kpts=kp); b = lpge.eta(mat, None, 0, -30.0, om, 0.01, dict(spin=True), kpts=kp)
    chi = ((a['sym']-b['sym'])[:, 0, 0, 0]/60).tolist(); g3 = gamma3()
    out[f"{s:.2f}"] = dict(gamma3=g3, chi=chi, Eg=float(Eg))
    print(f"{mat} s={s:.2f}  gamma3={g3:+.4f}  chi(Eg+.05,Eg+.1)={chi[0]*1e7:+.3f},{chi[1]*1e7:+.3f} e-7  Eg={Eg:.3f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"c2_{mat}.json"), "w"), indent=1)
