"""Data for the k.p figures (Figs. 6, 7, appendix k.p table) with TB-derived parameters (kp_params_TNN.json).
Sweeps: chi = [eta(+b) - eta(-b)]/(2B) at hbar*omega = Eg_active + DEL, one parameter varied about MoS2 (TNN).
Spectra: MoS2 eta^{x;xx}, sigma^{y;yy} vs omega for K+K' at B=0, K only at B=0, K+K' at B=30 T and at mu_B B = 80 meV.
Table: same components at omega = Eg_active + 0.1 eV, plus mirror-forbidden components."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import kp2
MUB = 5.7883818060e-5
P = json.load(open(os.path.join(HERE, "kp_params_TNN.json"))); base = P['MoS2']
R, NK, NU, BT = 0.6, 401, 0.005, 30.0
DEL = np.array([0.05, 0.10])

def chi(p):
    Eg = 2*p['mD']-p['lam']; om = Eg+DEL; b = MUB*BT
    ep = kp2.response('x', b, om, p=p, R=R, Nk=NK, nu=NU, want_shift=False)[0]
    em = kp2.response('x', -b, om, p=p, R=R, Nk=NK, nu=NU, want_shift=False)[0]
    return ((ep-em)/(2*BT)).tolist()

sw = {}
grids = {'kappa': np.linspace(-0.05, 0.05, 21), 'v': base['v']*np.linspace(0.5, 1.5, 21),
         'mD': np.linspace(0.5, 1.1, 21), 'lam': np.linspace(0.0, 0.25, 21)}
for name, g in grids.items():
    vals = []
    for x in g:
        p = dict(base); p[name] = float(x); vals.append(chi(p))
        print(name, f"{x:.4f}", vals[-1], flush=True)
    sw[name] = dict(x=g.tolist(), chi=vals)
sw['base'] = base; sw['DEL'] = DEL.tolist(); sw['B_T'] = BT; sw['nu'] = NU
json.dump(sw, open(os.path.join(HERE, "kp_sweeps.json"), "w"), indent=1)

# spectra
Eg = 2*base['mD']-base['lam']; om = np.linspace(Eg-0.1, Eg+1.0, 221); spec = dict(omega=om.tolist(), Eg=Eg)
for tag, b, pairs in [('KKp_B0', 0.0, ((1, 1), (-1, -1))), ('K_B0', 0.0, ((1, 1),)),
                      ('KKp_B30', MUB*BT, ((1, 1), (-1, -1))), ('KKp_b80meV', 0.08, ((1, 1), (-1, -1)))]:
    Lp, Sh = kp2.response('x', b, om, p=base, R=R, Nk=NK, nu=0.01, want_shift=True, pairs=pairs)
    spec[tag] = dict(lpge=Lp.tolist(), shift=Sh.tolist()); print('spectrum', tag, flush=True)
json.dump(spec, open(os.path.join(HERE, "kp_spectra.json"), "w"), indent=1)

# table: full tensors at Eg_active + 0.1 eV; nu = 0.01 eV
o = np.array([Eg+0.10]); tab = {}
for tag, b, pairs in [('KKp_B0', 0.0, ((1, 1), (-1, -1))), ('KKp_B30', MUB*BT, ((1, 1), (-1, -1))), ('K_B0', 0.0, ((1, 1),))]:
    Lt, St = kp2.response('x', b, o, p=base, R=R, Nk=NK, nu=0.01, want_shift=True, pairs=pairs, full=True)
    tab[tag] = dict(lpge_xxx=float(Lt[0, 0, 0, 0]), lpge_yyy=float(Lt[0, 1, 1, 1]), shift_yyy=float(St[0, 1, 1, 1]),
                    shift_xxx=float(St[0, 0, 0, 0]))
json.dump(tab, open(os.path.join(HERE, "kp_table.json"), "w"), indent=1); print(tab)
