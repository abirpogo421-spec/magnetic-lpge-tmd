"""v7 (2026-10-14): moment-term decomposition of chi, fixed engine. Replaces paper/v6_work/code/runs/decomp.py (cloud paths).
chi = -2 x si_factor x [eta(+B) - eta(-B)]/(2B), B = 10 T, A m V^-2 s^-1 T^-1.
 vH  : full BZ, N=600, nu=10 meV, at the van Hove energies of the paper (Sec. III B, from prod_*_F.npz).
 edge: local disks R=0.9/a, h=0.002/a, nu=5 meV, at Eg+0.1 eV (same grid as Fig. 12).
Variants: Z (Zeeman only), D (+phase-space), E (+energy shift), V (+velocity shift), ED, EV, F (all).
usage: python3 decomp_v7.py MAT   -> decomp7_{MAT}.json"""
import sys, os, json, hashlib, datetime, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../pub_pipeline")
import lpge as L, tb_core as tb
m = sys.argv[1]; PHYS = -2.0; B = 10.0
VH = {'MoS2': 3.350, 'MoSe2': 3.043, 'WS2': 3.600, 'WSe2': 2.991}   # v6 values (fallback only)
if os.path.exists(f"prod7_{m}_F.npz"):   # v7: van Hove energy = omega of max |eta| at the largest field in the Fig. 8 maps
    _d = np.load(f"prod7_{m}_F.npz", allow_pickle=True); _meta = json.loads(str(_d['meta']))
    assert _meta['done'] == len(_d['B_tesla']), 'prod7 map not finished; refusing to take vH from a partial map'
    VH[m] = float(_d['omega'][np.argmax(np.abs(_d['eta'][-1]))])
tb.set_config(m, 'GGA'); q = np.linspace(-0.05, 0.05, 401); g = []
for blk in L._H6(4*np.pi/3 + q, 0*q, 'TNN', 0.0):
    e = np.linalg.eigvalsh(blk); g.append((e[:, 1] - e[:, 0]).min())
Eg = float(min(g))
V = {'Z': dict(spin=True), 'D': dict(spin=True, dos='v'), 'E': dict(spin=True, mom=True), 'V': dict(spin=True, vel=True),
     'ED': dict(spin=True, mom=True, dos='v'), 'EV': dict(spin=True, mom=True, vel=True), 'F': dict(spin=True, mom=True, dos='v', vel=True)}
kl = L.local_kpts(0.9, 0.002)
out = dict(meta=dict(material=m, Eg=Eg, vH_eV=VH[m], vH_source=('prod7 argmax |eta(60 T)|' if os.path.exists(f'prod7_{m}_F.npz') else 'v6 hard-coded'), edge_eV=Eg+0.1, B=B, lpge_sha256=hashlib.sha256(open(L.__file__, 'rb').read()).hexdigest(),
                     script='decomp_v7.py', date=datetime.datetime.now().isoformat(), vH_grid='full BZ N=600 nu=0.01', edge_grid='R=0.9/a h=0.002/a nu=0.005'))
for where, om, kw in (('vH', VH[m], dict(N=600, nu=0.01)), ('edge', Eg+0.1, dict(N=None, nu=0.005, kpts=kl))):
    for v, fl in V.items():
        r = [L.eta(m, 'TNN', kw['N'], s*B, np.array([om]), kw['nu'], fl, kpts=kw.get('kpts')) for s in (1, -1)]
        chi = PHYS*r[0]['si_factor']*(r[0]['sym'][0, 0, 0, 0] - r[1]['sym'][0, 0, 0, 0])/(2*B)
        out[f"{m}|{where}|{v}"] = chi; print(m, where, v, "%+.4e" % chi, flush=True)
        json.dump(out, open(f"decomp7_{m}.json", "w"), indent=1)
