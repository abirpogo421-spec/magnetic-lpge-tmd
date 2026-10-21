import sys, time, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_core as tb, response as resp, field_scans as fs
m = sys.argv[1]; N = int(sys.argv[2]); ext = float(sys.argv[3])
be = tb.NumpyBackend()
tb.set_config(m, "GGA"); al, beta, JinvT, dArea = tb.build_grid(N); ka, psm = tb.material_constants()
old = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figure_data", "f5_base_omega.npz"))[m]   # omega grid of the zero-field survey run
dw = old[1]-old[0]; om = np.arange(old[0], old[-1]*ext, dw)
H3 = tb.make_H3('TNN', al, beta); out = {}
for b in (0.0, 0.08):
    t = time.time()
    G = tb.geometry(be, fs._orbital_H(H3, b), N, JinvT, "external", 0.0, ka, psm, orbital_corr=False)
    L, S = resp.nonlinear(G, om, dArea, be=be); L = L.real
    out[f"L{b}"] = np.sqrt((L**2).sum((-1, -2, -3))); out[f"S{b}"] = np.sqrt((S**2).sum((-1, -2, -3)))
    print(m, b, round(time.time()-t), flush=True)
np.savez(f"f5_{m}_{N}.npz", omega=om, **out)
