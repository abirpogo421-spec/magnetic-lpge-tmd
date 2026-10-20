"""Production run: eta^{a;bc}(omega, B) for one material and one field-model variant.
usage: python3 run_prod.py MAT VARIANT [N] [nu_eV]
VARIANT  Z : on-site L_z + spin Zeeman (field model validated by the valley g-factor)
         F : Z + itinerant moment energy + phase-space weight (occupied band) + velocity shift
B >= 0 only: T-symmetry gives eta(-B) = -eta(B) exactly for the injection current (checked in A9/A18).
Writes prod_{MAT}_{VARIANT}.npz with metadata and the sha256 of lpge.py."""
import sys, os, json, time, hashlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lpge as L, tb_core as tb

mat, var = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 900
nu = float(sys.argv[4]) if len(sys.argv) > 4 else 0.010
FLAGS = {'Z': dict(spin=True, mom=False, dos='none', vel=False),
         'F': dict(spin=True, mom=True, dos='v', vel=True)}[var]
Bs = np.arange(0, 60.01, 3.0)
tb.set_config(mat, "GGA")
# gap at B=0 from a dense line through K
q = np.linspace(-0.05, 0.05, 201); up, dn = L._H6(4*np.pi/3+q, 0*q, "TNN", 0.0)
Eg = min((np.linalg.eigvalsh(up)[:, 1]-np.linalg.eigvalsh(up)[:, 0]).min(), (np.linalg.eigvalsh(dn)[:, 1]-np.linalg.eigvalsh(dn)[:, 0]).min())
om = np.linspace(Eg-0.15, Eg+2.6, 1101)
eta = np.zeros((len(Bs), len(om), 2, 2, 2)); raw = np.zeros_like(eta); viol = np.zeros((len(Bs), len(om)))
fn = f"prod_{mat}_{var}.npz"
sha = hashlib.sha256(open(L.__file__, 'rb').read()).hexdigest()
for i, B in enumerate(Bs):
    t = time.time()
    r = L.eta(mat, "TNN", N, B, om, nu, FLAGS)
    eta[i], raw[i], viol[i] = r['sym'], r['raw'], r['c3_violation']
    meta = dict(material=mat, variant=var, flags=FLAGS, N=N, nu_eV=nu, model="Liu TNN GGA", Eg_B0=float(Eg),
                si_factor=r['si_factor'], units="eta in code units; multiply by si_factor for A m V^-2 s^-1",
                done=i+1, lpge_sha256=sha, Bsym="eta(-B)=-eta(B)")
    np.savez(fn, B_tesla=Bs, omega=om, eta=eta, raw=raw, c3_violation=viol, meta=json.dumps(meta))
    print(f"{mat} {var} B={B:5.1f}T  done in {time.time()-t:.0f}s  maxviol(|om-Eg|<1)={viol[i][om<Eg+1].max():.1e}", flush=True)
