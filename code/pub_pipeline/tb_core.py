#!/usr/bin/env python3
# ============================================================================
#  tb_core.py -- model + geometry for the SIX-MATERIAL x {GGA,LDA} x {NN,TNN}
#  field survey.  This is the tmdsurvey/tb_core.py engine (validated 2026-09-04),
#  extended ONLY to (a) load parameters for BOTH functionals from the verified
#  tmd_params.py table and (b) select material+functional via set_config().
#  The geometry, field terms and corrections are BYTE-FOR-BYTE identical to the
#  validated GGA build (cross-checked: engine_ref._H3 == this H_*_np, max|diff|=0).
#
#  Field terms (unchanged, added 2026-09-02, orbital-moment factor-2 fix 2026-09-04):
#     external (real B):  spin Zeeman + on-site orbital L_z
#                         + itinerant orbital moment  E_n -> E_n - mu_B B M_n(k)
#                         + phase-space measure       1 + (e/hbar) B Omega_n(k)
#     exchange (proximity): spin splitting ONLY (Delta in eV); no orbital terms.
#  e = hbar = a = 1 in the model; a_phys restores SI units for the orbital terms.
# ============================================================================
import numpy as np
from tmd_params import (get_params, MATERIALS, FUNCTIONALS, MODELS,
                        NN as _NN_TABLE, TNN as _TNN_TABLE, LAMBDA as _LAM_TABLE)

# ---- fixed physics constants (identical to the validated notebooks) ----------
LAM   = 0.073               # Ising SOC (eV) -- set per material/functional by set_config
G_S   = 2.0                 # electron spin g-factor
MU_B  = 5.7883818012e-5     # Bohr magneton (eV/T)
NU    = 0.03                # Lorentzian broadening (eV)
NFILL = 2                   # filled bands per spin sector (2-of-6)
VCELL = 0.886               # cell area for polarization cumulants
WIN   = 120.0              # omega window in units of nu (rel tail ~1e-4)
CD    = np.complex64

# ---- SI constants for the orbital-field unit conversion ---------------------
M_E_SI   = 9.1093837015e-31
HBAR_SI  = 1.054571817e-34
E_SI     = 1.602176634e-19
J_PER_EV = 1.602176634e-19

# ---- measured exciton valley-Zeeman g (|g|~4), VALIDATION reference only -----
G_VALLEY_MEASURED = {'MoS2': -1.8, 'WS2': -4.3, 'MoSe2': -4.0,
                     'WSe2': -4.4, 'MoTe2': -4.0, 'WTe2': -4.0}

# ---- reciprocal lattice + on-site orbital angular momentum ------------------
B1 = 2*np.pi*np.array([1.0, -1/np.sqrt(3)])
B2 = 2*np.pi*np.array([0.0,  2/np.sqrt(3)])
LZ = np.array([[0, 0, 0], [0, 0, 2j], [0, -2j, 0]], dtype=np.complex128)

_NN_KEYS = ['e1', 'e2', 't0', 't1', 't2', 't11', 't12', 't22']
_TNN_KEYS = _NN_KEYS + ['r0', 'r1', 'r2', 'r11', 'r12', 'u0', 'u1', 'u2', 'u11', 'u12', 'u22']

# --- CURRENT config globals (used by make_H3/assemble_SO/material_constants) ---
#  Default = MoS2/GGA so behaviour matches the validated build until set_config().
def _load_cfg(material, functional):
    """Build P_NN, P_TNN dicts (tb_core key names) + lambda + lattice(m) from tmd_params."""
    pnn = get_params(material, functional, "NN")
    ptnn = get_params(material, functional, "TNN")
    # tmd_params uses eps1/eps2; tb_core formulas use e1/e2. Remap.
    P_NN = {'e1': pnn['eps1'], 'e2': pnn['eps2']}
    for k in ('t0','t1','t2','t11','t12','t22'):
        P_NN[k] = pnn[k]
    P_TNN = {'e1': ptnn['eps1'], 'e2': ptnn['eps2']}
    for k in _TNN_KEYS[2:]:
        P_TNN[k] = ptnn[k]
    lam = pnn['lambda']
    a_m = pnn['a'] * 1e-10          # Angstrom -> metre
    return P_NN, P_TNN, lam, a_m

P_NN, P_TNN, LAM, A_PHYS = _load_cfg('MoS2', 'GGA')
CURRENT_MATERIAL = 'MoS2'
CURRENT_FUNCTIONAL = 'GGA'


def set_config(material, functional):
    """Point module globals (hoppings, SOC lambda, lattice) at (material, functional).
    make_H3/assemble_SO/material_constants read these globals, so all downstream follows."""
    global P_NN, P_TNN, LAM, A_PHYS, CURRENT_MATERIAL, CURRENT_FUNCTIONAL
    if material not in MATERIALS or functional not in FUNCTIONALS:
        raise ValueError(f"bad config {material!r}/{functional!r}; "
                         f"materials={MATERIALS} functionals={FUNCTIONALS}")
    P_NN, P_TNN, LAM, A_PHYS = _load_cfg(material, functional)
    CURRENT_MATERIAL = material; CURRENT_FUNCTIONAL = functional


def set_material(material):
    """Back-compat: set material, keep current functional."""
    set_config(material, CURRENT_FUNCTIONAL)


def material_constants(material=None, functional=None):
    """Return (kappa_orb [1/eV], psm_coeff [1/T]) for the orbital-field terms,
    using the CURRENT config's lattice constant (functional-dependent)."""
    if material is None:
        a = A_PHYS
    else:
        _, _, _, a = _load_cfg(material, functional or CURRENT_FUNCTIONAL)
    kappa_orb = M_E_SI * a**2 / HBAR_SI**2 * J_PER_EV
    psm_coeff = E_SI * a**2 / HBAR_SI
    return kappa_orb, psm_coeff


# ============================================================================
#  Backend shim: numpy default (float64 reference); torch/CUDA (complex64) when
#  requested.  geometry() is written once against these; einsum strings identical.
# ============================================================================
class NumpyBackend:
    name = "numpy"
    def asarray(self, a): return np.ascontiguousarray(a)
    def tonp(self, a): return np.asarray(a)
    def roll(self, a, sh, ax): return np.roll(a, sh, axis=ax)
    def eigh(self, a): return np.linalg.eigh(a)
    def block_eigh(self, a):
        Eu, Vu = np.linalg.eigh(a[..., :3, :3]); Ed, Vd = np.linalg.eigh(a[..., 3:, 3:])
        E = np.concatenate([Eu, Ed], axis=-1); V = np.zeros_like(a)
        V[..., :3, :3] = Vu; V[..., 3:, 3:] = Vd
        return E, V
    def einsum(self, expr, *ops): return np.einsum(expr, *ops, optimize='optimal')
    def stack(self, xs, ax): return np.stack(xs, axis=ax)

class TorchBackend:
    name = "torch"
    def __init__(self, dev):
        import torch; self.t = torch; self.dev = dev
    def asarray(self, a):
        a = np.ascontiguousarray(a)
        if a.dtype == np.complex128: a = a.astype(np.complex64)   # complex64 on GPU
        elif a.dtype == np.float64:  a = a.astype(np.float32)
        return self.t.as_tensor(a, device=self.dev)
    def tonp(self, a): return a.detach().cpu().numpy()
    def roll(self, a, sh, ax): return self.t.roll(a, shifts=sh, dims=ax)
    def eigh(self, a): return self.t.linalg.eigh(a)
    def block_eigh(self, a):
        t = self.t
        Eu, Vu = t.linalg.eigh(a[..., :3, :3]); Ed, Vd = t.linalg.eigh(a[..., 3:, 3:])
        E = t.cat([Eu, Ed], dim=-1); V = t.zeros_like(a)
        V[..., :3, :3] = Vu; V[..., 3:, 3:] = Vd
        return E, V
    def einsum(self, expr, *ops): return self.t.einsum(expr, *ops)
    def stack(self, xs, ax): return self.t.stack(list(xs), dim=ax)

def make_backend(use_gpu):
    if use_gpu:
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            if dev == "cpu":
                print("  [backend] torch present but CUDA unavailable -> torch/CPU")
            return TorchBackend(dev)
        except Exception as e:
            print("  [backend] torch unavailable (%s) -> numpy" % e)
    return NumpyBackend()


# ============================================================================
#  Grid + Hamiltonians (identical formulas to the validated build)
# ============================================================================
def build_grid(N):
    f = (np.arange(N) + 0.5)/N
    F1, F2 = np.meshgrid(f, f, indexing='ij')
    kx = F1*B1[0] + F2*B2[0]; ky = F1*B1[1] + F2*B2[1]
    alpha = 0.5*kx; beta = (np.sqrt(3)/2)*ky
    J = np.array([[B1[0], B2[0]], [B1[1], B2[1]]])
    JinvT = np.linalg.inv(J).T
    dArea = abs(np.linalg.det(J))/N**2
    return alpha, beta, JinvT, dArea

def H_NN_np(al, be, p):
    h0 = 2*p['t0']*(np.cos(2*al)+2*np.cos(al)*np.cos(be)) + p['e1']
    h1 = -2*np.sqrt(3)*p['t2']*np.sin(al)*np.sin(be) + 2j*p['t1']*(np.sin(2*al)+np.sin(al)*np.cos(be))
    h2 = 2*p['t2']*(np.cos(2*al)-np.cos(al)*np.cos(be)) + 2*np.sqrt(3)*1j*p['t1']*np.cos(al)*np.sin(be)
    h11 = 2*p['t11']*np.cos(2*al) + (p['t11']+3*p['t22'])*np.cos(al)*np.cos(be) + p['e2']
    h22 = 2*p['t22']*np.cos(2*al) + (3*p['t11']+p['t22'])*np.cos(al)*np.cos(be) + p['e2']
    h12 = np.sqrt(3)*(p['t22']-p['t11'])*np.sin(al)*np.sin(be) + 4j*p['t12']*np.sin(al)*(np.cos(al)-np.cos(be))
    H = np.array([[h0, h1, h2], [np.conj(h1), h11, h12], [np.conj(h2), np.conj(h12), h22]])
    return np.moveaxis(H, (0, 1), (-2, -1))

def H_TNN_np(al, be, p):
    V0 = p['e1'] + 2*p['t0']*(2*np.cos(al)*np.cos(be)+np.cos(2*al)) + 2*p['r0']*(2*np.cos(3*al)*np.cos(be)+np.cos(2*be)) + 2*p['u0']*(2*np.cos(2*al)*np.cos(2*be)+np.cos(4*al))
    ReV1 = -2*np.sqrt(3)*p['t2']*np.sin(al)*np.sin(be) + 2*(p['r1']+p['r2'])*np.sin(3*al)*np.sin(be) - 2*np.sqrt(3)*p['u2']*np.sin(2*al)*np.sin(2*be)
    ImV1 = 2*p['t1']*np.sin(al)*(2*np.cos(al)+np.cos(be)) + 2*(p['r1']-p['r2'])*np.sin(3*al)*np.cos(be) + 2*p['u1']*np.sin(2*al)*(2*np.cos(2*al)+np.cos(2*be))
    V1 = ReV1 + 1j*ImV1
    ReV2 = 2*p['t2']*(np.cos(2*al)-np.cos(al)*np.cos(be)) - (2/np.sqrt(3))*(p['r1']+p['r2'])*(np.cos(3*al)*np.cos(be)-np.cos(2*be)) + 2*p['u2']*(np.cos(4*al)-np.cos(2*al)*np.cos(2*be))
    ImV2 = 2*np.sqrt(3)*p['t1']*np.cos(al)*np.sin(be) + (2/np.sqrt(3))*np.sin(be)*(p['r1']-p['r2'])*(np.cos(3*al)+2*np.cos(be)) + 2*np.sqrt(3)*p['u1']*np.cos(2*al)*np.sin(2*be)
    V2 = ReV2 + 1j*ImV2
    V11 = p['e2'] + (p['t11']+3*p['t22'])*np.cos(al)*np.cos(be) + 2*p['t11']*np.cos(2*al) + 4*p['r11']*np.cos(3*al)*np.cos(be) + 2*(p['r11']+np.sqrt(3)*p['r12'])*np.cos(2*be) + (p['u11']+3*p['u22'])*np.cos(2*al)*np.cos(2*be) + 2*p['u11']*np.cos(4*al)
    ReV12 = np.sqrt(3)*(p['t22']-p['t11'])*np.sin(al)*np.sin(be) + 4*p['r12']*np.sin(3*al)*np.sin(be) + np.sqrt(3)*(p['u22']-p['u11'])*np.sin(2*al)*np.sin(2*be)
    ImV12 = 4*p['t12']*np.sin(al)*(np.cos(al)-np.cos(be)) + 4*p['u12']*np.sin(2*al)*(np.cos(2*al)-np.cos(2*be))
    V12 = ReV12 + 1j*ImV12
    V22 = p['e2'] + (3*p['t11']+p['t22'])*np.cos(al)*np.cos(be) + 2*p['t22']*np.cos(2*al) + 2*p['r11']*(2*np.cos(3*al)*np.cos(be)+np.cos(2*be)) + (2/np.sqrt(3))*p['r12']*(4*np.cos(3*al)*np.cos(be)-np.cos(2*be)) + (3*p['u11']+p['u22'])*np.cos(2*al)*np.cos(2*be) + 2*p['u22']*np.cos(4*al)
    H = np.array([[V0, V1, V2], [np.conj(V1), V11, V12], [np.conj(V2), np.conj(V12), V22]])
    return np.moveaxis(H, (0, 1), (-2, -1))

def make_H3(model, al, be):
    return H_NN_np(al, be, P_NN) if model == "NN" else H_TNN_np(al, be, P_TNN)

def assemble_SO(H3, up_add, dn_add):
    N0, N1 = H3.shape[0], H3.shape[1]
    z = np.zeros((N0, N1, 3, 3), dtype=np.complex128)
    up = H3 + (LAM/2)*LZ + up_add
    dn = H3 - (LAM/2)*LZ + dn_add
    top = np.concatenate([up, z], axis=-1); bot = np.concatenate([z, dn], axis=-1)
    return np.concatenate([top, bot], axis=-2)

def H_external(H3, B):
    """Real magnetic field: on-site orbital L_z + spin Zeeman (both blocks)."""
    zorb = MU_B*B*LZ; zsp = 0.5*G_S*MU_B*B*np.eye(3)
    return assemble_SO(H3, zorb + zsp, zorb - zsp)

def H_exchange(H3, dz):
    """Proximity exchange: spin splitting only, no orbital coupling."""
    s = (dz/2.0)*np.eye(3)
    return assemble_SO(H3, s, -s)


# ============================================================================
#  Geometry at one field (backend-agnostic; heavy einsums chunked over k).
#  study: "external" or "exchange"; x: field value (Tesla or eV).
#  Applies the orbital-moment energy shift + phase-space weight for "external".
#
#  PRECISION NOTE (complex64 GPU path): the shift current uses SECOND k-derivatives
#  of the projector (ddP), which are f32-sensitive. To protect them, the finite-
#  difference geometry (H, dH, P, dP, ddP, velocities, Berry, moment) is computed
#  on the NUMPY (float64) backend regardless of `be`; only the heavy transition
#  CONTRACTIONS (Q, Csym, Casym) run on `be` (complex64 on GPU). This is the
#  "f64 geometry -> f32 contraction" split. Validated f32-vs-f64 in gauntlet.py.
# ============================================================================
KCHUNK = 16384
# Production precision mode for the finite-difference geometry:
#   False -> full complex64 on the compute backend (FAST, pure-GPU; validated
#            relerr ~1e-6 on SHIFT/LPGE up to N=160, not growing -- gauntlet.py block B).
#   True  -> finite-diff geometry in float64 (numpy), cast to c64 only for the heavy
#            contractions (marginally more accurate; moves geometry to CPU -> slower).
# The run driver may flip this and runs a one-time f32-vs-f64 self-check at the real N.
F64_GEOM = False

# Spin is a good quantum number in this model (Ising SOC, Zeeman, exchange are all
# spin-diagonal; hoppings spin-independent), so H is EXACTLY block-diagonal in the two
# 3x3 spin sectors. Diagonalizing each block separately (BLOCK_SAFE=True) is exact and
# avoids the energy-sorting artifact that corrupts the projector finite differences at
# inter-spin-block band crossings (which otherwise fakes a nonzero spin/exchange LPGE).
# See gauntlet.py invariants A8-A11. Set False only to reproduce the old (buggy) path.
BLOCK_SAFE = True

def _diag(geo_be, H):
    """Eigendecomposition that respects the conserved spin blocks when BLOCK_SAFE."""
    if BLOCK_SAFE:
        return geo_be.block_eigh(H)
    return geo_be.eigh(H)

def geometry(be, H6_np, N, JinvT, study, x, kappa_orb, psm_coeff, kchunk=None,
             orbital_corr=True, f64_geom=None):
    kchunk = kchunk or KCHUNK
    if f64_geom is None:
        f64_geom = F64_GEOM
    # --- finite-difference geometry in float64 (numpy) when f64_geom, else on `be`
    geo_be = NumpyBackend() if f64_geom else be
    H = geo_be.asarray(H6_np)
    Ji = JinvT; hf = 1.0/N
    def df1(T): return (geo_be.roll(T, -1, 0) - geo_be.roll(T, 1, 0))/(2*hf)
    def df2(T): return (geo_be.roll(T, -1, 1) - geo_be.roll(T, 1, 1))/(2*hf)
    def gkx(T): return Ji[0, 0]*df1(T) + Ji[0, 1]*df2(T)
    def gky(T): return Ji[1, 0]*df1(T) + Ji[1, 1]*df2(T)
    dHx = gkx(H); dHy = gky(H)
    E, V = _diag(geo_be, H)
    vkx = geo_be.einsum('ijkn,ijkl,ijlm->ijnm', V.conj(), dHx, V)
    vky = geo_be.einsum('ijkn,ijkl,ijlm->ijnm', V.conj(), dHy, V)
    P = geo_be.einsum('ijkn,ijln->ijnkl', V, V.conj())
    dP = geo_be.stack([gkx(P), gky(P)], 0)
    ddP = geo_be.stack([geo_be.stack([gkx(dP[0]), gky(dP[0])], 0),
                        geo_be.stack([gkx(dP[1]), gky(dP[1])], 0)], 0)
    Nk = N*N
    # numpy host copies of the projector tensors (used by cumulants + return).
    # Kept on HOST; only per-chunk slices are moved to the compute backend below,
    # so peak VRAM ~ chunk size (not the full N*N projector tensors) -- lets N=600
    # fit on an 8 GB card. (Validated bit-identical to full-materialize.)
    P_np = geo_be.tonp(P).reshape(Nk, 6, 6, 6)
    dP_np = geo_be.tonp(dP).reshape(2, Nk, 6, 6, 6)
    ddP_np = geo_be.tonp(ddP).reshape(2, 2, Nk, 6, 6, 6)
    del H, dHx, dHy, P, dP, ddP
    Q = np.empty((Nk, 6, 6, 2, 2), np.complex128)
    Csym = np.empty((Nk, 6, 6, 2, 2, 2), np.complex128)
    Casym = np.empty((Nk, 6, 6, 2, 2, 2), np.complex128)
    ii = np.arange(6)
    for a0 in range(0, Nk, kchunk):
        b0 = min(a0 + kchunk, Nk)
        Pb = be.asarray(P_np[a0:b0]); dPb = be.asarray(dP_np[:, a0:b0]); ddPb = be.asarray(ddP_np[:, :, a0:b0])
        Qb = be.einsum('knpq,akmqr,bknrp->kmnab', Pb, dPb, dPb)
        Qb = be.tonp(Qb); Qb[:, ii, ii, :, :] = 0.0
        hess = be.einsum('knpq,bkmqr,aynrp->kmnaby'.replace('yn', 'ykn'), Pb, dPb, ddPb)
        cube = be.einsum('knpq,bkmqr,akmrs,yknsp->kmnaby', Pb, dPb, dPb, dPb)
        Cmn = be.tonp(hess) + be.tonp(cube)
        Ca = 0.5*(Cmn - np.einsum('kmnaby->knmaby', Cmn))
        Cs_by = 0.5*(Ca + np.einsum('kmnaby->kmnayb', Ca))
        Cas_by = 0.5*(Ca - np.einsum('kmnaby->kmnayb', Ca))
        Q[a0:b0] = Qb; Csym[a0:b0] = Cs_by; Casym[a0:b0] = Cas_by

    E = geo_be.tonp(E).reshape(Nk, 6)
    vkxn = geo_be.tonp(vkx).reshape(Nk, 6, 6)
    vkyn = geo_be.tonp(vky).reshape(Nk, 6, 6)

    # ---- Berry curvature, orbital moment, phase-space weight (numpy f64, cheap) ----
    dEnm = E[:, :, None] - E[:, None, :]                        # E_n - E_m
    dEnm = np.where(np.abs(dEnm) < 1e-8, np.inf, dEnm)
    vy_T = np.swapaxes(vkyn, -1, -2); vx_T = np.swapaxes(vkxn, -1, -2)
    Omega = -2*(vkxn*vy_T/dEnm**2).sum(axis=-1).imag           # (Nk,6)  Kubo Berry
    Mraw  = ((vkxn*vy_T - vkyn*vx_T)/dEnm).sum(axis=-1).imag    # = 2 Im sum v^x_nm v^y_mn/(E_n-E_m)
    # Xiao-Chang-Niu orbital moment (Bohr magnetons): M_n = -kappa_orb*Mraw.
    M     = kappa_orb*(-1.0)*Mraw                               # orbital moment (mu_B)
    if study == "external" and orbital_corr:
        E = E - MU_B*x*M                                        # E_n -> E_n - mu_B B M_n
        dos_weight = 1.0 + psm_coeff*x*Omega                    # 1 + (e/hbar) B Omega_n
    else:                                                       # exchange, or exaggerated static
        dos_weight = np.ones_like(Omega)

    dkx = np.einsum('knn->kn', vkxn).real
    dky = np.einsum('knn->kn', vkyn).real
    Delta = np.stack([dkx[:, None, :] - dkx[:, :, None],
                      dky[:, None, :] - dky[:, :, None]], -1)
    return dict(E=E, vkx=vkxn, vky=vkyn, V=geo_be.tonp(V).reshape(Nk, 6, 6),
                P=P_np, dP=dP_np, ddP=ddP_np,
                Q=Q, Delta=Delta, Csym_by=Csym, Casym_by=Casym,
                Omega=Omega, Mmom=M, dos_weight=dos_weight)


def light_geometry(be, H6_np, N, JinvT, f64_geom=None):
    """Cheap geometry: E and velocity matrix elements only (no projector tensors).
    For optical conductivity / metric on a FINE grid where projectors aren't needed."""
    if f64_geom is None:
        f64_geom = F64_GEOM
    geo_be = NumpyBackend() if f64_geom else be
    H = geo_be.asarray(H6_np); Ji = JinvT; hf = 1.0/N
    def df1(T): return (geo_be.roll(T, -1, 0) - geo_be.roll(T, 1, 0))/(2*hf)
    def df2(T): return (geo_be.roll(T, -1, 1) - geo_be.roll(T, 1, 1))/(2*hf)
    def gkx(T): return Ji[0, 0]*df1(T) + Ji[0, 1]*df2(T)
    def gky(T): return Ji[1, 0]*df1(T) + Ji[1, 1]*df2(T)
    dHx = gkx(H); dHy = gky(H)
    E, V = _diag(geo_be, H)
    vkx = geo_be.einsum('ijkn,ijkl,ijlm->ijnm', V.conj(), dHx, V)
    vky = geo_be.einsum('ijkn,ijkl,ijlm->ijnm', V.conj(), dHy, V)
    Nk = N*N
    return dict(E=geo_be.tonp(E).reshape(Nk, 6), vkx=geo_be.tonp(vkx).reshape(Nk, 6, 6),
                vky=geo_be.tonp(vky).reshape(Nk, 6, 6), dos_weight=np.ones((Nk, 6)))


# ============================================================================
#  C3 index symmetrization of a rank-3 in-plane tensor (identical to notebook)
# ============================================================================
def C3_sym3(T):
    th = 2*np.pi/3
    Rs = [np.array([[np.cos(k*th), -np.sin(k*th)], [np.sin(k*th), np.cos(k*th)]]) for k in (0, 1, 2)]
    out = np.zeros_like(T)
    for R in Rs:
        Tr = np.zeros_like(T)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    acc = 0.0
                    for a in range(2):
                        for b in range(2):
                            for c in range(2):
                                acc = acc + R[i, a]*R[j, b]*R[k, c]*T[..., a, b, c]
                    Tr[..., i, j, k] = acc
        out = out + Tr
    return out/3.0


# ---- back-compat aliases (old code referenced these names) ------------------
LATTICE_A = {m: get_params(m, 'GGA', 'NN')['a']*1e-10 for m in MATERIALS}
LAM_ALL = dict(_LAM_TABLE)
