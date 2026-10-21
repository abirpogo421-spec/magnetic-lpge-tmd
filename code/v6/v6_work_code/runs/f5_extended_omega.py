import os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "v6jobs", "f5"))   # v6 engine (tb_core, response, field_scans)
import tb_core as tb, response as resp, field_scans as fs
m = sys.argv[1]; N = int(sys.argv[2]); ext = float(sys.argv[3])
be = tb.NumpyBackend()
tb.set_config(m, "GGA"); al, beta, JinvT, dArea = tb.build_grid(N); ka, psm = tb.material_constants()
old = np.load(os.path.join(HERE, "..", "..", "figure_data", "f5_base_omega.npz"))[m]   # omega grid of the zero-field survey run
dw = old[1]-old[0]; om = np.arange(old[0], old[-1]*ext, dw)
H3 = tb.make_H3('TNN', al, beta); out = {}
for b in (0.0, 0.08):
    t = time.time()
    G = tb.geometry(be, tb.H_external(H3, b/tb.MU_B), N, JinvT, "external", b/tb.MU_B, ka, psm, orbital_corr=True)
    Ls, Ss = [], []
    for c in np.array_split(om, max(1, len(om)//100)):
        L, S = resp.nonlinear(G, c, dArea, be=be); L = L.real
        Ls.append(np.sqrt((L**2).sum((-1, -2, -3)))); Ss.append(np.sqrt((S**2).sum((-1, -2, -3)))); del L, S
    out[f"L{b}"] = np.concatenate(Ls); out[f"S{b}"] = np.concatenate(Ss); del G
    print(m, b, round(time.time()-t), flush=True)
np.savez(f"f5_{m}_{N}.npz", omega=om, **out)
