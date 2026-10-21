#!/usr/bin/env python3
# ============================================================================
#  field_scans.py -- extra .npz products focused on the LINEAR COEFFICIENT of the
#  full-Zeeman LPGE (the paper's central quantity) + Jyesta's requested views.
#  FULL external Zeeman (orbital L_z + spin Zeeman + moment + DOS) -- the spin-only/exchange
#  variant was dropped from the paper.
#
#  Per (material, functional) it writes, for NN and TNN:
#    fixed_field/nonlinear_fixed.npz   LPGE eta^{x;xx}(omega) & shift sigma^{y;yy}(omega)
#                                      at a few fixed fields, realistic -> exaggerated.
#    fixed_field/sigma_fixed.npz       sigma_xx(omega), sigma_xy(omega) at the same fields
#                                      (frozen linear conductivity vs field).
#    linear_coefficient/lpge_vs_B.npz  eta^{x;xx} vs field on a fine grid, the fitted linear
#                                      coefficient chi = d eta/d(muB B), the linear-regime edge
#                                      (where it stops being ~linear), R^2, and band-edge
#                                      geometry (lambda, gap, m_D, Dirac v, Omega(K), tr g(K))
#                                      -- everything needed to assess the lambda/TMD dependence
#                                      and to compare against the k.p closed form kappa v^3/m_tau^3.
#
#  Fields are the FULL external Zeeman (H_external: orbital L_z + spin Zeeman) WITH the
#  itinerant orbital moment + phase-space DOS weight (orbital_corr=True), parameterised by
#  b = muB*B (eV). Spin-Zeeman alone gives LPGE=0, so the LPGE is orbital-activated -- but the
#  model here is the full Zeeman of the paper, not an orbital-only reduction. The field axis is
#  still directly comparable to the k.p mass shift m_tau = m_D - tau b.
#
#  Engine is BLOCK-SAFE (tb_core.BLOCK_SAFE=True): spin/exchange give exactly zero LPGE,
#  so the LPGE below is the genuine orbital-activated effect within the full Zeeman field.
# ============================================================================
import os, numpy as np
import tb_core as tb
import response as resp

XI = (0, 0, 0)   # eta^{x;xx}
YI = (1, 1, 1)   # eta^{y;yy} (mirror-forbidden -- must be ~0)
SY = (1, 1, 1)   # sigma^{y;yy} shift independent component

# fixed fields for the omega-scans: 0 (reference) + realistic -> exaggerated (muB B, eV)
FIXED_B = np.array([0.0, 0.0035, 0.010, 0.020, 0.040, 0.080])     # 0.0035 eV ~ 60 T
# fine field grid for the linear-coefficient fit / linear-regime scoping (muB B, eV)
LIN_B = np.linspace(0.0, 0.10, 21)


def _orbital_H(H3, b_eV):
    """On-site orbital L_z Zeeman (external/orbital case), field b = muB*B in eV."""
    orb = b_eV*tb.LZ                         # muB*B*L_z ; b_eV = muB*B
    return tb.assemble_SO(H3, orb, orb)


def dirac_params(material, functional, model='TNN'):
    """Band-edge quantities at K: gap Delta, m_D=Delta/2, Dirac velocity v (eV*a),
    Berry Omega(K), metric trace tr g(K), SOC lambda. v from gap(k)^2 = 4 v^2 k^2 + Delta^2."""
    tb.set_config(material, functional)
    kxK = 4*np.pi/3.0
    def H6(kx, ky):
        al = 0.5*kx; be = (np.sqrt(3)/2)*ky
        return tb.H_external(tb.make_H3(model, np.array([[al]]), np.array([[be]])), 0.0)[0, 0]
    def gap(kx, ky):
        w = np.linalg.eigvalsh(H6(kx, ky)); return float(w[tb.NFILL]-w[tb.NFILL-1])
    Delta = gap(kxK, 0.0); mD = 0.5*Delta
    ks = np.array([0.0, 0.01, 0.02, 0.03])                 # small displacements from K
    g2 = np.array([gap(kxK+kk, 0.0)**2 for kk in ks])
    slope = np.polyfit(ks**2, g2, 1)[0]                    # = 4 v^2
    v = float(np.sqrt(max(slope, 0.0))/2.0)
    # Omega(K), tr g(K) via Kubo at K (6-band)
    e = 1e-4; Hc = H6(kxK, 0.0); w, V = np.linalg.eigh(Hc)
    dHx = (H6(kxK+e, 0.0)-H6(kxK-e, 0.0))/(2*e); dHy = (H6(kxK, e)-H6(kxK, -e))/(2*e)
    vx = V.conj().T@dHx@V; vy = V.conj().T@dHy@V
    dE = w[:, None]-w[None, :]; dE = np.where(np.abs(dE) < 1e-8, np.inf, dE)
    Om = -2*(vx*vy.T/dE**2).sum(1).imag                    # Omega_n = -2 Im sum v^x_nm v^y_mn/dE^2
    gxx = (vx*vx.T/dE**2).sum(1).real; gyy = (vy*vy.T/dE**2).sum(1).real
    vb = int(np.argsort(w)[tb.NFILL-1])
    return dict(gap=Delta, mD=mD, v=v, OmegaK=float(Om[vb]), trgK=float((gxx+gyy)[vb]), lam=float(tb.LAM))


def _lpge_shift(H3, b_eV, N, Ji, dA, om, be, ka=None, psm=None):
    """Full-Zeeman LPGE/shift at field b_eV = muB*B (eV): orbital L_z + spin Zeeman (H_external),
    itinerant orbital moment + phase-space DOS weight (orbital_corr=True, geometry field in Tesla).
    Spin-Zeeman alone gives LPGE=0, so the LPGE is orbital-activated; the FULL Zeeman is the paper's
    model. (Accepts H3 + field, not a pre-built H6, so it can build the full-field geometry.)"""
    if ka is None:
        ka, psm = tb.material_constants()
    B = b_eV/tb.MU_B                                       # geometry moment/DOS field in Tesla
    G = tb.geometry(be, tb.H_external(H3, B), N, Ji, "external", B, ka, psm, orbital_corr=True)
    L, S = resp.nonlinear(G, om, dA, be=be)   # LPGE (complex, .real physical), SHIFT (real)
    return L.real, S


def run_field_scans(mout, material, functional, be, N=480, Nsig=800, nom=800, nom_sig=200,
                    models=('NN', 'TNN'), verbose=True):
    """Compute the fixed-field omega-scans, the frozen sigma-vs-field, and the LPGE-vs-field
    linear-coefficient extraction for one (material, functional). FULL external Zeeman."""
    tb.set_config(material, functional)
    al, beta, JinvT, dArea = tb.build_grid(N)
    ka, psm = tb.material_constants()
    # omega grids from the zero-field TNN geometry
    G0 = tb.geometry(be, _orbital_H(tb.make_H3('TNN', al, beta), 0.0), N, JinvT,
                     "external", 0.0, ka, psm, orbital_corr=False)
    omega, omega_sig, gmin, wtop = resp.omega_grids(G0, nom, nom_sig)

    fdir = os.path.join(mout, 'fixed_field'); os.makedirs(fdir, exist_ok=True)
    ldir = os.path.join(mout, 'linear_coefficient'); os.makedirs(ldir, exist_ok=True)

    nf = len(FIXED_B); nb = len(LIN_B)
    alS, beS, JiS, dAS = tb.build_grid(Nsig)          # fine grid for frozen sigma (built once)
    nl = {}; sg = {}; lin = {}
    for model in models:
        H3 = tb.make_H3(model, al, beta); H3s = tb.make_H3(model, alS, beS)
        # ---- fixed-field omega scans (LPGE eta^{x;xx}, shift sigma^{y;yy}) ----
        LP = np.empty((nf, nom)); SH = np.empty((nf, nom)); LPn = np.empty((nf, nom)); SHn = np.empty((nf, nom))
        SXX = np.empty((nf, nom_sig), complex); SXY = np.empty((nf, nom_sig), complex)
        for i, b in enumerate(FIXED_B):
            L, S = _lpge_shift(H3, float(b), N, JinvT, dArea, omega, be, ka, psm)
            LP[i] = L[:, 0, 0, 0]; SH[i] = S[:, 1, 1, 1]
            LPn[i] = np.sqrt((L**2).sum((-1, -2, -3))); SHn[i] = np.sqrt((S**2).sum((-1, -2, -3)))
            # frozen sigma at this field (light geometry, fine grid)
            Gs = tb.light_geometry(be, _orbital_H(H3s, float(b)), Nsig, JiS)
            sig = resp.sigma_linear(Gs, omega_sig, dAS, be=be)
            SXX[i] = sig[:, 0, 0]; SXY[i] = sig[:, 0, 1]
            if verbose:
                print(f"    [{material}/{functional} {model}] fixed b={b:.4f} eV done", flush=True)
        nl[f'{model}_LPGE_xxx'] = LP; nl[f'{model}_shift_yyy'] = SH
        nl[f'{model}_LPGE_norm'] = LPn; nl[f'{model}_shift_norm'] = SHn
        sg[f'{model}_sxx'] = SXX; sg[f'{model}_sxy'] = SXY

        # ---- LPGE vs field (fine grid) + linear-coefficient fit ----
        peakB = np.empty(nb); ipk = None
        etaW = np.empty((nb, nom))                        # eta^{x;xx}(b, omega)
        forb = np.empty(nb)                               # forbidden y;yy peak (should stay ~0)
        for j, b in enumerate(LIN_B):
            L, S = _lpge_shift(H3, float(b), N, JinvT, dArea, omega, be, ka, psm)
            e = L[:, 0, 0, 0]; etaW[j] = e
            if ipk is None and j == len(LIN_B)-1:
                ipk = int(np.argmax(np.abs(e)))            # peak-omega index at the largest field
            peakB[j] = e[np.argmax(np.abs(e))] if b > 0 else 0.0
            forb[j] = np.max(np.abs(L[:, 1, 1, 1]))
        # value at the fixed peak-omega vs field (cleaner for a slope)
        etaPk = etaW[:, ipk]
        # SEED-ANCHORED through-origin linear fit: chi is the slope over the DEEP-linear seed
        # (b <= b_seed), which is inside the linear regime for every material (so a short-linear-
        # regime high-SOC material cannot bias chi). The linear-regime EDGE is then measured
        # outward as the largest b within 10% of that true slope, and R^2 reports how linear the
        # response is across [0, edge].
        b_seed = 0.012
        mseed = (LIN_B > 0) & (LIN_B <= b_seed)
        if mseed.sum() < 1:                                   # coarse grid: take smallest nonzero field
            mseed = LIN_B == np.sort(LIN_B[LIN_B > 0])[0]
        chi = float(np.sum(LIN_B[mseed]*etaPk[mseed])/np.sum(LIN_B[mseed]**2))
        chi_om = (np.sum(LIN_B[mseed, None]*etaW[mseed], 0)/np.sum(LIN_B[mseed]**2))
        with np.errstate(divide='ignore', invalid='ignore'):
            dev = np.abs(etaPk - chi*LIN_B)/np.abs(chi*LIN_B)
        good = (LIN_B > 0) & (dev < 0.10)
        edge = float(LIN_B[good].max()) if good.any() else float(LIN_B[mseed].max())
        mlin = (LIN_B > 0) & (LIN_B <= edge)                  # R^2 across the linear regime
        pred = chi*LIN_B[mlin]; ss_res = np.sum((etaPk[mlin]-pred)**2)
        ss_tot = np.sum((etaPk[mlin]-etaPk[mlin].mean())**2)
        R2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else 1.0
        geo = dirac_params(material, functional, model)
        lin[f'{model}_eta_peak_vs_b'] = np.abs(etaPk)      # |eta^{x;xx}| at peak-omega vs b
        lin[f'{model}_eta_peak_signed'] = etaPk
        lin[f'{model}_eta_bw'] = etaW                       # full eta^{x;xx}(b, omega)
        lin[f'{model}_chi'] = chi                           # linear coeff /eV, at peak-omega
        lin[f'{model}_chi_omega'] = chi_om                 # /eV, per omega
        lin[f'{model}_R2'] = R2; lin[f'{model}_linear_edge_eV'] = edge
        lin[f'{model}_omega_peak'] = float(omega[ipk]); lin[f'{model}_forbidden_peak'] = float(forb.max())
        for kk, vv in geo.items():
            lin[f'{model}_{kk}'] = vv
        if verbose:
            print(f"    [{material}/{functional} {model}] chi={chi:+.4e}/eV  R2={R2:.4f}  "
                  f"linear<= {edge:.3f} eV  (gap={geo['gap']:.3f}, lam={geo['lam']:.3f})", flush=True)

    _META_NL = ("Fixed-field omega-scans, ORBITAL Zeeman (Hamiltonian L_z, no perturbative corr). "
                "b_eV = muB*B (eV): 0 then realistic(~0.0035=60T) -> exaggerated(0.08). "
                "{NN,TNN}_LPGE_xxx[nf,nom] = eta^{x;xx}(omega) (signed, the C3v-independent LPGE); "
                "{NN,TNN}_shift_yyy[nf,nom] = sigma^{y;yy}(omega); *_norm = tensor norms. index 0=x,1=y.")
    np.savez_compressed(os.path.join(fdir, 'nonlinear_fixed.npz'),
                        b_eV=FIXED_B, omega=omega, README=_META_NL, **nl)
    _META_SG = ("Frozen linear optical conductivity vs field (ORBITAL Zeeman). b_eV=muB*B (eV) as above. "
                "{NN,TNN}_sxx[nf,nom_sig], {NN,TNN}_sxy[nf,nom_sig] complex; Re sxx=absorption, sxy=Hall.")
    np.savez_compressed(os.path.join(fdir, 'sigma_fixed.npz'),
                        b_eV=FIXED_B, omega_sig=omega_sig, README=_META_SG, **sg)
    _META_LIN = ("LPGE eta^{x;xx} vs ORBITAL field for the LINEAR-COEFFICIENT extraction. "
                 "b_eV=muB*B (eV) on a fine grid [0,0.10]. Per model: eta_peak_vs_b[nb]=|eta| at peak-omega; "
                 "eta_peak_signed[nb]; eta_bw[nb,nom]=eta^{x;xx}(b,omega); chi=d eta/d(muB B) /eV (linear coeff, "
                 "peak-omega, origin fit b<=0.02); chi_omega[nom]=per-omega slope; R2; linear_edge_eV=largest b "
                 "with <10% deviation from linear; omega_peak; forbidden_peak (y;yy, ~0 check). "
                 "Band-edge geometry per model: gap, mD=gap/2, v (Dirac velocity, eV*a), OmegaK, trgK, lam. "
                 "chi is evaluated at the (van-Hove) peak omega, so its ABSOLUTE value is NOT directly the "
                 "near-K k.p coefficient; the valid k.p comparisons are (i) SIGN (set by the warping sign) and "
                 "(ii) the cross-material scaling of |chi| with v^3/m_D^4 (see summarize_chi.py / LAMBDA_DEPENDENCE).")
    np.savez_compressed(os.path.join(ldir, 'lpge_vs_B.npz'),
                        b_eV=LIN_B, b_fit=0.02, README=_META_LIN, **lin)
    return lin
