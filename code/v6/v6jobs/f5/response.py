#!/usr/bin/env python3
# ============================================================================
#  response.py -- linear + nonlinear response at one field, from a geometry dict.
#
#  VECTORIZED / GPU build (2026-09-04): the per-omega Python loops are replaced
#  by chunked matmuls on the backend (torch/CUDA when available, numpy otherwise).
#  The math is UNCHANGED and bit-faithful to the previous version:
#    * nonlinear keeps the SAME Lorentzian window (|omega-dE| < WIN*nu) via a mask,
#      so the summed terms are identical to the old windowed loop.
#    * sigma_linear is EXACTLY the notebook all-band-pair Kubo form.
#  Only HOW the omega dependence is evaluated changed (loop -> matmul), which is
#  what turns the ~150-220 s/field nonlinear pass into seconds on the GPU.
#
#  Nonlinear (2nd order, DC photocurrent), C3-symmetrized:
#     LPGE  = symmetric-in-(b,c) injection   (T-odd; nonzero once field breaks T)
#     SHIFT = symmetric-in-(b,c) shift        (T-even; allowed at B=0)
#     CPGE / gyration are C3-forbidden -> machine zero, omitted.
# ============================================================================
import numpy as np
import tb_core as tb

NU = tb.NU; NFILL = tb.NFILL; VCELL = tb.VCELL; WIN = tb.WIN; CD = tb.CD

# transition-chunk size for the matmuls (bounds peak memory of the L/den blocks)
TCHUNK = 400000


def _occ(E):
    order = np.argsort(E, axis=-1)
    occ = np.zeros_like(E)
    np.put_along_axis(occ, order[:, :NFILL], 1.0, axis=-1)
    return occ


# ---------------------------------------------------------------------------
#  small backend helpers so the same code runs on numpy or torch tensors
# ---------------------------------------------------------------------------
def _is_torch(be):
    return getattr(be, "name", "numpy") == "torch"


def nonlinear(G, omega, dArea, be=None):
    """LPGE + SHIFT vs omega (C3-symmetrized). Vectorized over omega as chunked
    matmuls; identical Lorentzian window (WIN*nu) as the reference loop."""
    E = G['E']; occ = _occ(E)
    dE = E[:, :, None] - E[:, None, :]                    # E_m - E_n
    fdiff = occ[:, None, :] - occ[:, :, None]             # occ_n - occ_m
    mask = (np.arange(6)[:, None] > np.arange(6)[None, :]).astype(float)
    w0 = mask[None]*fdiff
    # Phase-space (DOS) measure of the OCCUPIED (valence) band: d2k -> d2k * D_v(k),
    #   D_v = 1 + (e/hbar) B Omega_v(k)  (tb_core.geometry -> G['dos_weight']).
    # Applied on the transition's valence index (last axis). Linear-in-B correction the paper
    # quotes (+-45% at 30 T); D_v>0 so it never flips signs. See PHYSICS_VERIFICATION.md.
    dosw = G.get('dos_weight')
    if dosw is not None:
        w0 = w0 * dosw[:, None, :]
    nz = np.abs(w0) > 1e-12
    dEf = dE[nz].astype(np.float64); amp = w0[nz].astype(np.float64)
    Delta = G['Delta'][nz]; Q = G['Q'][nz]; Csym = G['Csym_by'][nz]
    Tn = dEf.shape[0]
    nom = len(omega)
    if Tn == 0:
        inj = np.zeros((nom, 8), np.complex128); lin = np.zeros((nom, 8))
    else:
        injpl = (Delta[:, :, None, None]*Q[:, None, :, :]).reshape(Tn, 8).astype(np.complex128)
        linpl = Csym.imag.reshape(Tn, 8).astype(np.float64)
        half = WIN*NU
        om = np.asarray(omega, np.float64)
        inj, lin = _accum_lorentz(be, om, dEf, amp, injpl, linpl, half)
    MEAS = dArea/(2*np.pi)**2
    inj = tb.C3_sym3((inj.reshape(nom, 2, 2, 2))*np.pi*MEAS)
    lin = tb.C3_sym3((lin.reshape(nom, 2, 2, 2))*(-2*np.pi)*MEAS)
    LPGE = (0.5*(inj + np.swapaxes(inj, -1, -2))).astype(CD)   # symmetric-bc injection
    SHIFT = lin.astype(np.float64)
    return LPGE, SHIFT


def _accum_lorentz(be, om, dEf, amp, injpl, linpl, half):
    """sum_t Lorentzian(om,dEf_t)*amp_t * {injpl_t, linpl_t}, windowed, chunked.
    Returns (inj[nom,8] complex, lin[nom,8] real)."""
    nom = om.shape[0]; Tn = dEf.shape[0]
    use_torch = _is_torch(be)
    if use_torch:
        t = be.t; dev = be.dev
        omt = t.as_tensor(om, device=dev, dtype=t.float32)
        inj = t.zeros((nom, 8), device=dev, dtype=t.complex64)
        lin = t.zeros((nom, 8), device=dev, dtype=t.float32)
        for a0 in range(0, Tn, TCHUNK):
            b0 = min(a0+TCHUNK, Tn)
            dEc = t.as_tensor(dEf[a0:b0], device=dev, dtype=t.float32)
            ampc = t.as_tensor(amp[a0:b0], device=dev, dtype=t.float32)
            d = omt[:, None] - dEc[None, :]                       # (nom, chunk)
            L = (NU/np.pi)*ampc[None, :]/(d*d + NU*NU)
            L = t.where(d.abs() < half, L, t.zeros_like(L))       # SAME window
            inj = inj + (L.to(t.complex64) @ t.as_tensor(injpl[a0:b0], device=dev, dtype=t.complex64))
            lin = lin + (L @ t.as_tensor(linpl[a0:b0], device=dev, dtype=t.float32))
        return be.tonp(inj).astype(np.complex128), be.tonp(lin).astype(np.float64)
    # numpy path (chunked; float64)
    inj = np.zeros((nom, 8), np.complex128); lin = np.zeros((nom, 8), np.float64)
    for a0 in range(0, Tn, TCHUNK):
        b0 = min(a0+TCHUNK, Tn)
        d = om[:, None] - dEf[a0:b0][None, :]
        L = (NU/np.pi)*amp[a0:b0][None, :]/(d*d + NU*NU)
        L = np.where(np.abs(d) < half, L, 0.0)
        inj += L.astype(np.complex128) @ injpl[a0:b0]
        lin += L @ linpl[a0:b0]
    return inj, lin


def nonlinear_channels(G, omega, dArea, be=None):
    """ALL FOUR second-order channels vs omega, C3-symmetrized (for the static
    B0/fullZ/spinZ comparison figure). Returns real (nom,2,2,2) tensors:
      LPGE  = symmetric-(bc) injection   (T-odd, field-activated)
      CPGE  = antisymmetric-(bc) injection (C3-forbidden in-plane -> ~0)
      SHIFT = symmetric-(bc) shift        (T-even, allowed at B=0)
      GYR   = antisymmetric-(bc) shift    (C3-forbidden in-plane -> ~0)
    Same windowed transition sum / prefactors as nonlinear(); only here we also
    keep the antisymmetric parts so the forbidden channels can be shown = 0."""
    E = G['E']; occ = _occ(E)
    dE = E[:, :, None] - E[:, None, :]
    fdiff = occ[:, None, :] - occ[:, :, None]
    mask = (np.arange(6)[:, None] > np.arange(6)[None, :]).astype(float)
    w0 = mask[None]*fdiff
    dosw = G.get('dos_weight')                 # phase-space measure D_v of the occupied band
    if dosw is not None:
        w0 = w0 * dosw[:, None, :]
    nz = np.abs(w0) > 1e-12
    dEf = dE[nz].astype(np.float64); amp = w0[nz].astype(np.float64)
    Delta = G['Delta'][nz]; Q = G['Q'][nz]
    Csym = G['Csym_by'][nz]; Casym = G['Casym_by'][nz]
    Tn = dEf.shape[0]; nom = len(omega); half = WIN*NU; om = np.asarray(omega, np.float64)
    MEAS = dArea/(2*np.pi)**2
    injpl = (Delta[:, :, None, None]*Q[:, None, :, :]).reshape(Tn, 8).astype(np.complex128)
    shiftpl = Csym.imag.reshape(Tn, 8).astype(np.float64)
    gyrpl = Casym.imag.reshape(Tn, 8).astype(np.float64)
    inj, shift = _accum_lorentz(be, om, dEf, amp, injpl, shiftpl, half)
    _, gyr = _accum_lorentz(be, om, dEf, amp, injpl, gyrpl, half)
    inj = tb.C3_sym3(inj.reshape(nom, 2, 2, 2)*np.pi*MEAS)
    shift = tb.C3_sym3(shift.reshape(nom, 2, 2, 2)*(-2*np.pi)*MEAS)
    gyr = tb.C3_sym3(gyr.reshape(nom, 2, 2, 2)*(-2*np.pi)*MEAS)
    swb = lambda T: np.swapaxes(T, -1, -2)
    LPGE = 0.5*(inj + swb(inj)).real
    CPGE = 0.5*(inj - swb(inj)).real
    SHIFT = 0.5*(shift + swb(shift))
    GYR = 0.5*(gyr - swb(gyr))
    return LPGE, CPGE, SHIFT, GYR


def sigma_linear(G, omega_sig, dArea, be=None):
    """Linear optical conductivity sigma^{ab}(omega) -- PHYSICAL interband Kubo:
    only occupied->unoccupied pairs (Pauli factor f_v - f_c = 1) with the resonant
    two-pole /epsilon structure, so it is ZERO below the gap and turns on at the gap
    (verified run-spec 2026-09-02; the old all-pairs form gave a spurious ~1/omega
    sub-gap tail). This ALSO restricts the sum to NFILL*(6-NFILL)=8 pairs per k
    instead of 36, so it is ~4-5x cheaper -- the main linear-response speedup.
        sigma^{ab}(w) = i*MEAS * sum_{v in occ, c in unocc} (v^a_cv v^b_vc / e_cv)
                        [ 1/(e_cv - w - i nu) - 1/(e_cv + w + i nu) ]
    Full complex 2x2 tensor; chunked over transitions on the backend."""
    E = G['E']; Nk = E.shape[0]
    order = np.argsort(E, axis=-1)
    occ_idx = order[:, :NFILL]          # (Nk, NFILL) occupied band indices per k
    unocc_idx = order[:, NFILL:]        # (Nk, 6-NFILL) unoccupied
    vkx = G['vkx']; vky = G['vky']
    kk = np.arange(Nk)[:, None, None]
    v = occ_idx[:, :, None]             # (Nk, occ, 1)
    c = unocc_idx[:, None, :]           # (Nk, 1, unocc)
    ecv = (E[kk, c] - E[kk, v]).reshape(-1).astype(np.float64)          # (ntr,)  > 0
    vx_cv = vkx[kk, c, v]; vy_cv = vky[kk, c, v]                        # v^a_{cv}
    vx_vc = vkx[kk, v, c]; vy_vc = vky[kk, v, c]                        # v^a_{vc}
    Mxx = (vx_cv*vx_vc).reshape(-1); Mxy = (vx_cv*vy_vc).reshape(-1)
    Myx = (vy_cv*vx_vc).reshape(-1); Myy = (vy_cv*vy_vc).reshape(-1)
    Mab = np.stack([Mxx, Mxy, Myx, Myy], axis=-1) / ecv[:, None]        # (ntr,4)  /e_cv
    Mab = Mab.astype(np.complex128)
    dosw = G.get('dos_weight')                 # phase-space measure D_v of the occupied band
    if dosw is not None:
        Dv = np.broadcast_to(dosw[kk, v], (Nk, v.shape[1], c.shape[2])).reshape(-1)
        Mab = Mab * Dv[:, None]
    om = np.asarray(omega_sig, np.float64); nom = om.shape[0]; ntr = ecv.shape[0]
    MEAS = dArea/(2*np.pi)**2
    use_torch = _is_torch(be)
    if use_torch:
        t = be.t; dev = be.dev
        omt = t.as_tensor(om, device=dev, dtype=t.complex64)
        acc = t.zeros((nom, 4), device=dev, dtype=t.complex64)
        for a0 in range(0, ntr, TCHUNK):
            b0 = min(a0+TCHUNK, ntr)
            ec = t.as_tensor(ecv[a0:b0], device=dev, dtype=t.complex64)
            den = 1.0/(ec[None, :] - omt[:, None] - 1j*NU) - 1.0/(ec[None, :] + omt[:, None] + 1j*NU)
            acc = acc + den @ t.as_tensor(Mab[a0:b0], device=dev, dtype=t.complex64)
        out = be.tonp(acc).astype(np.complex128)
    else:
        out = np.zeros((nom, 4), np.complex128)
        for a0 in range(0, ntr, TCHUNK):
            b0 = min(a0+TCHUNK, ntr)
            ec = ecv[a0:b0]
            den = 1.0/(ec[None, :] - om[:, None] - 1j*NU) - 1.0/(ec[None, :] + om[:, None] + 1j*NU)
            out += den @ Mab[a0:b0]
    out = -1j*out          # sign: Re sigma_xx > 0 (absorption) above the gap
    return (out.reshape(nom, 2, 2)*MEAS).astype(np.complex128)


def cumulants(G, dArea):
    """Polarization variance & skewness (occupied), phase-space-weighted."""
    f = np.zeros(6); f[0] = 1; f[1] = 1
    fw = f[None, :]*G['dos_weight']                       # phase-space measure per band
    Pocc = np.einsum('kn,knpq->kpq', fw, G['P'])
    dPo = np.einsum('kn,aknpq->akpq', fw, G['dP'])
    ddPo = np.einsum('kn,abknpq->abkpq', fw, G['ddP'])
    MEAS = dArea/(2*np.pi)**2
    var = np.einsum('kpq,akqr,bkrp->ab', Pocc, dPo, dPo).real*VCELL*MEAS
    skew = np.einsum('kpq,akqr,bykrp->aby', Pocc, dPo, ddPo).imag*VCELL*MEAS
    return var, skew


def all_responses(G, omega_resp, omega_sig, dArea, be=None):
    LPGE, SHIFT = nonlinear(G, omega_resp, dArea, be=be)
    sigma = sigma_linear(G, omega_sig, dArea, be=be)
    var, skew = cumulants(G, dArea)
    Es = np.sort(G['E'], axis=-1)
    gap = float((Es[:, NFILL] - Es[:, NFILL-1]).min())
    return dict(LPGE=LPGE, SHIFT=SHIFT, sigma=sigma, var=var, skew=skew, gap=gap)


def omega_grids(G0, nom, nom_sig=None):
    """Response grid: 0.9*min-gap -> 1.05*max-interband-transition (nom points).
    Sigma grid: ~gap -> same top (nom_sig points; the linear conductivity is smooth
    and secondary here, so it uses far fewer points than the nonlinear grid)."""
    if nom_sig is None:
        nom_sig = nom
    Es = np.sort(G0['E'], axis=-1)
    gmin = float((Es[:, NFILL] - Es[:, NFILL-1]).min())
    wtop = 1.05*float((Es[:, -1] - Es[:, 0]).max())
    omega_resp = np.linspace(0.9*gmin, wtop, nom)
    omega_sig = np.linspace(max(1e-2, 0.5*gmin), wtop, nom_sig)   # sigma: start below gap, coarse
    return omega_resp, omega_sig, gmin, wtop
