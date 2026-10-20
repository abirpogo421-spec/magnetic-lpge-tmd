"""
Publication pipeline: linear injection current (LPGE) of the Liu three-band TB under an out-of-plane field.

Independent of tmdsurvey2/response.py (only the verified Hamiltonian builder tb_core.make_H3 / assemble_SO is reused;
tmd_params checked against Liu Tables II-IV, TNN formulas checked by C3 symmetry, see checks/).

Physics (velocity form, interband, T=0, mu in gap):
  eta^{a;bc}(w) = pi * MEAS * sum_k sum_{v occ, c empty, same spin block}
                  W_v(k) * Delta^a_cv(k) * Re[v^b_vc v^c_cv]/e0_cv^2 * L_nu(w - e_cv(k))
  MEAS = dA/(2pi)^2 (k in 1/a, energies eV)  ->  SI by e^3 a / hbar^2  [A m V^-2 s^-1]
Field terms (switches):
  lz    : on-site orbital Zeeman mu_B B L_z in H (always on for 'external')
  spin  : spin Zeeman g_s mu_B B S_z in H
  mom   : itinerant orbital-moment energy shift e_n -> e_n - mu_B B M_n (resonance only; geometry uses bare e0)
  dos   : phase-space weight on the transition: 'none' | 'v' (occupied band) | 'avg' ((D_v+D_c)/2)
  vel   : velocity shift Delta^a -> Delta^a - mu_B B d_a(M_c - M_v)
Numerics:
  - k-derivatives by symmetric finite difference with step EPS in k (not the grid spacing), per k-point
  - per-spin-block diagonalization (S_z conserved exactly)
  - Lorentzian summed exactly over all transitions (no binning, no energy window; v7 engine 2026-10-14)
  - output: raw (unsymmetrized) tensor AND C3-symmetrized tensor; the raw C3 violation is reported as a convergence meter
"""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import tb_core as tb

E_SI = 1.602176634e-19; HBAR_SI = 1.054571817e-34
EPS = 1e-4          # k-step (1/a) for finite-difference velocities and dM/dk
CHUNK = 20000

def grid(N):
    return tb.build_grid(N)          # (alpha, beta, JinvT, dArea); k in 1/a, reciprocal of Liu lattice

def _H6(kx, ky, model, B, spin=True):
    al = 0.5*kx; be = (np.sqrt(3)/2)*ky
    H3 = tb.make_H3(model, al[:, None], be[:, None])[:, 0]           # (n,3,3)
    zorb = tb.MU_B*B*tb.LZ
    zsp = 0.5*tb.G_S*tb.MU_B*B*np.eye(3) if spin else 0*np.eye(3)
    up = H3 + (tb.LAM/2)*tb.LZ + zorb + zsp
    dn = H3 - (tb.LAM/2)*tb.LZ + zorb - zsp
    return up, dn

def _block_quantities(H, Hxp, Hxm, Hyp, Hym):
    """Eigen + velocity matrices for one 3x3 block (vectorized over k)."""
    E, V = np.linalg.eigh(H)
    dHx = (Hxp - Hxm)/(2*EPS); dHy = (Hyp - Hym)/(2*EPS)
    vx = np.einsum('kin,kij,kjm->knm', V.conj(), dHx, V)
    vy = np.einsum('kin,kij,kjm->knm', V.conj(), dHy, V)
    return E, V, vx, vy

def _moment_omega(E, vx, vy, ka):
    dE = E[:, :, None] - E[:, None, :]
    dE = np.where(np.abs(dE) < 1e-9, np.inf, dE)
    S = (vx*np.swapaxes(vy, -1, -2))                                  # v^x_nm v^y_mn
    Om = -2*(S/dE**2).sum(-1).imag
    M = -ka*2*(S/dE).sum(-1).imag                                     # Xiao-Chang-Niu, units mu_B
    return M, Om

def _M_only(H, ka):
    E, V = np.linalg.eigh(H)
    return E, V

def chunk_transitions(kx, ky, model, B, flags, ka, psm):
    """Return per-transition arrays for one chunk: e_res, e0, W, Delta(2), g(2,2)."""
    out = []
    e = EPS
    Hs = {}
    for tag, (dx, dy) in {'0': (0, 0), 'xp': (e, 0), 'xm': (-e, 0), 'yp': (0, e), 'ym': (0, -e)}.items():
        Hs[tag] = _H6(kx+dx, ky+dy, model, B, flags.get('spin', True))
    for blk in (0, 1):
        H = Hs['0'][blk]
        E, V, vx, vy = _block_quantities(H, Hs['xp'][blk], Hs['xm'][blk], Hs['yp'][blk], Hs['ym'][blk])
        M, Om = _moment_omega(E, vx, vy, ka)
        dM = None
        if flags.get('vel'):
            # dM/dk by finite difference of the moment itself (needs velocities at shifted k)
            Ms = {}
            for tag, (dx, dy) in {'xp': (e, 0), 'xm': (-e, 0), 'yp': (0, e), 'ym': (0, -e)}.items():
                Hc = Hs[tag][blk]
                # velocities at shifted point via its own +-EPS neighbours
                hp = _H6(kx+dx+e, ky+dy, model, B, flags.get('spin', True))[blk]
                hm = _H6(kx+dx-e, ky+dy, model, B, flags.get('spin', True))[blk]
                hq = _H6(kx+dx, ky+dy+e, model, B, flags.get('spin', True))[blk]
                hr = _H6(kx+dx, ky+dy-e, model, B, flags.get('spin', True))[blk]
                Es, Vs, vxs, vys = _block_quantities(Hc, hp, hm, hq, hr)
                Ms[tag], _ = _moment_omega(Es, vxs, vys, ka)
            dM = np.stack([(Ms['xp']-Ms['xm'])/(2*e), (Ms['yp']-Ms['ym'])/(2*e)], -1)   # (n,3,2)
        e0 = E.copy()
        er = E - tb.MU_B*B*M if flags.get('mom') else E
        D = 1.0 + psm*B*Om
        vd = np.stack([np.einsum('knn->kn', vx).real, np.einsum('knn->kn', vy).real], -1)  # (n,3,2)
        if dM is not None:
            vd = vd - tb.MU_B*B*dM
        v = 0                                                         # valence = lowest band of each block
        for c in (1, 2):
            ecv0 = e0[:, c] - e0[:, v]; ecv = er[:, c] - er[:, v]
            Da = vd[:, c, :] - vd[:, v, :]
            vb = np.stack([vx[:, v, c], vy[:, v, c]], -1); vc = np.stack([vx[:, c, v], vy[:, c, v]], -1)
            g = (vb[:, :, None]*vc[:, None, :]).real/ecv0[:, None, None]**2
            dos = flags.get('dos', 'none')
            W = np.ones_like(ecv) if dos == 'none' else (D[:, v] if dos == 'v' else 0.5*(D[:, v]+D[:, c]))
            out.append((ecv, W, Da, g))
    return out

def local_kpts(R, h):
    """Square grid of spacing h (1/a) on disks of radius R around K=(4pi/3,0) and K'=(-4pi/3,0).
    Built from integer indices so that the point set is exactly inversion-symmetric (k -> -k maps K disk onto K'
    disk). Without exact T-pairing the large per-valley B=0 terms (~10x the field-induced signal) do not cancel."""
    n = int(np.floor(R/h + 1e-9)); i = np.arange(-n, n+1); X, Y = np.meshgrid(i, i, indexing='ij')
    m = (X**2 + Y**2) <= (R/h)**2 + 1e-9
    x, y = X[m]*h, Y[m]*h; KX = 4*np.pi/3
    return np.concatenate([x+KX, x-KX]), np.concatenate([y, y]), h*h

def eta(material, model, N, B, omega, nu, flags, functional="GGA", bin_eV=2e-4, emax=None, kpts=None, lam=None):
    """Return dict with raw and C3-symmetrized eta^{abc}(omega) in code units and the SI factor.
    kpts=(kx, ky, dA) overrides the full-BZ grid (used for the local fine grid near K, K')."""
    tb.set_config(material, functional)
    if lam is not None:
        tb.LAM = lam                 # SOC override (appendix lambda sweep)
    if kpts is None:
        al, be, Ji, dA = grid(N)
        kx = (2*al).ravel(); ky = (2*be/np.sqrt(3)).ravel()
    else:
        kx, ky, dA = kpts
    ka, psm = tb.material_constants()
    MEAS = dA/(2*np.pi)**2
    om = np.asarray(omega, float)
    # Exact Lorentzian sum over every transition (v7 engine, 2026-10-14). The v6 engine binned transitions into
    # 0.2 meV bins and dropped transitions outside [min(om)-40 nu, max(om)+40 nu]; both are removed here
    # (verification report V-03, V-04). bin_eV and emax are accepted for backward compatibility and ignored.
    acc = np.zeros((len(om), 8))
    lchunk = max(1000, int(4e6//max(len(om), 1)))
    for a0 in range(0, len(kx), CHUNK):
        for ecv, W, Da, g in chunk_transitions(kx[a0:a0+CHUNK], ky[a0:a0+CHUNK], model, B, flags, ka, psm):
            T = (Da[:, :, None, None]*g[:, None, :, :]).reshape(len(ecv), 8)*W[:, None]
            for b0 in range(0, len(ecv), lchunk):
                L = (nu/np.pi)/((om[:, None]-ecv[None, b0:b0+lchunk])**2 + nu**2)
                acc += L @ T[b0:b0+lchunk]
    raw = acc.reshape(len(om), 2, 2, 2)*np.pi*MEAS
    raw = 0.5*(raw + np.swapaxes(raw, -1, -2))                       # symmetric in (b,c): linear polarization
    sym = tb.C3_sym3(raw)
    viol = np.abs(raw - sym).max(axis=(1, 2, 3))/np.maximum(np.abs(sym).max(axis=(1, 2, 3)), 1e-300)
    si = E_SI**3*tb.A_PHYS/HBAR_SI**2
    return dict(raw=raw, sym=sym, c3_violation=viol, si_factor=si, omega=om)
