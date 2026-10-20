INDEPENDENT REFERENCE IMPLEMENTATION -- magnetic LPGE in monolayer MX2
=======================================================================

Files
  refimpl.py           the implementation (numpy only, ~800 lines, heavily commented)
  selftest.py          internal checks; run `python3 selftest.py` (about 2.5 min, one core)
  selftest_output.txt  output of the last run (96/96 code checks pass; 16/17 paper
                       comparisons agree, 1 borderline discrepancy, see below)

Provenance
  Written 2026-10-14 from two sources only:
    (1) the paper: /home/claude/v/paper/main.tex and sections/{theory,results,appendix}.tex,
        with the audit corrections listed under "Conventions" below;
    (2) Liu, Shan, Yao, Yao, Xiao, PRB 88, 085433 (2013), arXiv:1305.6089. The parameter
        tables were transcribed from the arXiv LaTeX source (e-print), Tables II, III, IV.
  No engine code (lpge.py, tb_core.py, tmd_params.py, kp2.py, ...), team or report
  files, or project documents were opened.

API (all k in units of 1/a; energies in eV)
  params(material, model='TNN', functional='GGA') -> dict
      a (Angstrom), eps1, eps2, t0..t22, r0..r12, u0..u22 (eV; zero for NN), lam (eV).
      material in MoS2, WS2, MoSe2, WSe2, MoTe2, WTe2; model 'NN'|'TNN'; functional 'GGA'|'LDA'.
  H3(kx, ky, material, model, functional, p=None, derivs=('H',))
      spinless 3x3 Liu Hamiltonian, basis {dz2, dxy, dx2-y2}. With derivs including
      'dx','dy','dxx','dxy','dyy' it returns exact analytic derivatives (dict).
  H_block(kx, ky, s, B, ...)   3x3 spin block: H3 + (lam/2) s Lz + mu_B B (Lz + (g_s/2) s)
  H6(kx, ky, B, ...)           full 6x6 I2(x)H3 + (lam/2) sz(x)Lz + mu_B B (I2(x)Lz + (g_s/2) sz(x)I3)
  band_quantities(kx, ky, s, B, ..., dM_step=1e-3) -> dict per k and band:
      E, U, Vx, Vy (velocity matrices <n|dH/dk_a|m>, analytic dH), v (band velocities),
      Omega (a^2), M (mu_B), D = 1 + (e/hbar) B Omega a^2, E_shift = E - mu_B B M,
      dM (dM/dk, 4th-order central differences, step 1e-3/a), v_shift = -mu_B B dM/dk.
  eta_tensor(material, model, functional, B, omega, nu, kpoints, dA, flags, electric=False)
      -> (Nw,2,2,2) physical eta^{a;bc} in A m V^-2 s^-1 [w,a,b,c]; electric=True also returns
      the electric-injection tensor (interband Berry curvature, prefactor -pi e^3 a/hbar^2).
  eta_xxx(...)  -> eta^{x;xx} only.
      flags (defaults): spin_zeeman=True, orbital_zeeman=True, moment_energy=True,
      phase_space=True, velocity_shift=True, c3=False, pairs='resonant' ('all' = literal
      sum over m != n including anti-resonant Lorentzian tails).
  sigma_shift_tensor(...)  shift current sigma^{a;(bc)} in A m V^-2 (x1e15 -> nm muA V^-2),
      projector formalism, Im Cbar^{mn}; d2P by central differences of analytic dP (step 1e-4).
      Extra (not in the requested list); symmetry-checked only, not compared with paper numbers.
  bz_grid(N) -> (k, dA)            Gamma-centred N x N grid (i b1 + j b2)/N; inversion, C3 and
                                   mirror invariant mod G.
  disk_grid(R, h) -> (k, dA)       square lattice of spacing h inside |q|<=R around K, and the
                                   K' disk built as exactly -(K disk).
  c3_expand(k, dA)                 {k, C3k, C3^2 k} with weight dA/3 (used by flags c3=True).
  direct_gap_K(...)                E_g = e_c(K) - e_v(K), active spin, B=0.
  kappa_lowdin(material, model, functional, spin_active=True, with_soc=True, tau=+1, lam=None)
      second-order Lowdin reduction at tau K onto {dz2, (dx2-y2 + i tau dxy)/sqrt2} using exact
      analytic first and second derivatives; returns v (eV A), kappa (eV A^2), w (eV A^2),
      gamma1..3, E_c, E_v, E_u, fit residuals, and for NN the paper's closed forms.
  kp_dvec, kp_kernels, kp_ring(k, v, kappa, m, tau)   two-band warped Dirac kernels:
      ring averages of Delta^a_cv g_bc (all 8), Delta^a Omega^v, Im C^{cv}_{a;(bc)}, Im Cbar.
  kp_inj_closed_lead / kp_inj_closed_next / kp_shift_closed   the paper's closed forms.
  kp_H(...)   2x2 d.sigma and analytic derivatives, to run the TB machinery on a pure 2-band H.

Conventions used (paper + audit corrections)
  * x along a metal-metal bond (NN vectors (+-1,0)a), b1=(2pi,2pi/sqrt3)/a, b2=(2pi,-2pi/sqrt3)/a,
    K=(4pi/3a,0), K'=-K. Checked: in Liu's H as transcribed the valence state at K has L_z=+2
    (and -2 at K'), so no relabelling of axes or valleys is needed.
  * b = mu_B B with mu_B = 5.7883818060e-5 eV/T (no hbar; audit correction). g_s = 2.
  * Omega_n = -2 Im sum_m v^x_nm v^y_mn/(e_n-e_m)^2 (= curl of A = i<u|grad u>); for the two-band
    lower band this is +1/2 n.(d_x n x d_y n) (audit correction; verified numerically, and the sum
    rule is independently checked against a plaquette Berry phase).
  * Interband geometry: g^{bc}_mn = Re[v^b_nm v^c_mn]/(e_n-e_m)^2 >= 0 and
    Omega^{bc}_mn = 2 Im[...]/(e_n-e_m)^2, i.e. Q^{mn}_{bc} = tr[P_n dP_m dP_n] = -g - (i/2) Omega.
  * M_n = -(2 m_e a^2/hbar^2) sum_m Im[v^x_nm v^y_mn]/(e_n - e_m) in mu_B.
  * eta^{a;(bc)} = -(2 pi e^3 a/hbar^2) int d^2k/(2pi)^2 sum f~_nm Delta^a_mn g^{bc}_mn L_nu(e_mn - hbar w),
    f~_nm = f_n D_n - f_m D_m, L_nu(x) = (nu/pi)/(x^2+nu^2), every k and every transition summed
    exactly (no binning, no window). The single-valley kernel carries the overall factor tau
    (verified: the ring average is odd in tau).
  * Physical sign: I re-derived it from the golden rule. For carriers of charge q,
    deta/dt gives eta = +(2 pi q^3/hbar^2) int Delta_cv g delta, so electrons (q=-e) give exactly
    the paper's negative prefactor with e = +1.602e-19 C. The code returns that number directly.

Assumptions and resolved ambiguities
  A1  LDA lambda: Liu gives lambda (Table IV) only for GGA; the GGA value is used for LDA too.
  A2  TNN lattice constant: Liu's TNN table has no a; the Table II value of the same functional is used.
  A3  H^(2) prefactor: Liu writes a^2 [gamma1 k^2, gamma3(...)^2; ...], the paper writes a^2/2. kappa is
      defined without ambiguity as the coefficient of (tau qx + i qy)^2 in H_eff,cv (paper d-vector,
      Eq. app_dvec), so kappa = a^2 gamma3/2 in the paper's convention (= a^2 gamma3_Liu). With this
      reading the paper's NN closed form Eq. (kappa_nn) and t = 3sqrt2/4 (t1 + sqrt3 t2) match the
      numerical reduction to 1e-12.
  A4  Lowdin gauge: the v-state phase is fixed so that v is real and positive (kappa*v and kappa/v are
      gauge invariant). In Liu's basis the phase is already 1 and kappa comes out real (Im = 0).
      E_c, E_v, E_u include the SOC of the chosen spin (active: v +lambda, u -lambda at K), B = 0.
  A5  Occupation: the lowest band of each spin block is filled and the two upper bands are empty
      (T=0 insulator); no Fermi level is used. Transitions v->c and v->remote are both included.
  A6  Transitions: default pairs='resonant' (n occupied -> m empty with L(e_mn - hbar w)), as in the
      appendix master formula. The literal "sum over m != n" also contains anti-resonant tails
      L(-e_cv - hbar w); pairs='all' includes them. Their size is 1e-5 to 6e-5 of eta^xxx (MoS2, N=90,
      nu=10 meV), so the engine comparison needs to use the same choice.
  A7  Geometry (g, Omega, M, v) comes from the exact eigenstates of H_s at field B (H_Z diagonalised);
      the moment enters only through e - mu_B B M, v - mu_B B dM/dk and D. M is evaluated at field B,
      not at B=0. The difference is O(B^2) in eta.
  A8  D_n multiplies only the occupied band (f_c = 0); g in Delta*g uses the bare eigenvalues.
  A9  E_g = e_c(K) - e_v(K) of the active spin block at B = 0 (the direct gap at K).
  A10 Spin Zeeman is a constant shift of each spin block, so it cannot change any spin-conserving
      response. Verified: eta is unchanged when it is switched off.
  A11 k.p kernels use only the warped massive-Dirac d-vector of Eq. (app_dvec): no eps0, no w k^2,
      no cubic terms, which is what the closed forms assume.

Self-test summary (selftest_output.txt)
  Code checks, 96/96 pass. They cover: Hermiticity; the term-list H3 against a literal transcription
  of the factored formulas (24 parameter sets, 1e-13); periodicity; time reversal; the vertical mirror
  (dxy odd); C3 invariance of the spectra with SOC and B; analytic dH and d2H against finite differences;
  the Liu Table I NN energies at Gamma/K/M (12 sets, 1e-12); spin splitting = 2 lambda; L_z = +-2 at K/K';
  sum_n Omega = 0; Omega against the plaquette Berry phase (1e-5); Hellmann-Feynman velocities; dM step
  convergence (1e-8); valley g from H_Z = -4; two-band identities run through the generic machinery
  (Delta, g, Omega = +1/2 n.(dn x dn), M_c = M_v, M(0) = -(m_e a^2/hbar^2) tau v^2/m); exact ring
  averages; C3 relations; the mirror-forbidden and electric (gyration-type) kernels vanishing; valley
  parity; the K+K' linear-in-b law; the next-order closed form Eq. (app_injfull) exact through O(k^4)
  (the residual scales as k^6 with coefficient +4.0); Lowdin residuals; kappa and v at K' equal to K;
  w checked independently against the TB band curvatures at K, (hbar^2/4m_e)(1/m_c+1/|m_v|) = w + v^2/2m
  (1e-5); grid inversion pairing and areas; eta(-B) = -eta(B) to 1e-9 and eta(0) = 0 on both disk and
  full grids; linearity at small B; C3 alpha-family relations; beta family of eta = 0; in-plane electric
  injection = 0; (bc) symmetry; shift current beta relations, mirror-forbidden part = 0, even in B, FD step.
  Paper comparisons, 16/17 agree. These include: the Table app_numcheck entries (3.922e-4, 1.083e-3,
  2.733e-3 exact; 3.935e-4, 1.093e-3, 2.798e-3 closed; shift 1.8145e-1); Table II (v, kappa TNN,
  kappa NN) for MoS2, MoSe2 and WS2; WS2 kappa(lambda) = +0.127/-0.022/-0.110 (paper +0.13/-0.02/-0.11);
  MoS2 LDA kappa = +0.008 (+0.01); Liu Delta = 1.663 eV; m_e a^2/hbar^2 = 1.34-1.45 1/eV; |D-1| < 3% at
  60 T (max 0.029); the E_g+0.01 deviation from linear at 60 T (0.3% < 1%).
  DISCREPANCY (borderline): WSe2 TNN kappa = +0.0849 eV A^2, which rounds to +0.08; the paper prints +0.09.
  INFO (not pass/fail):
    threshold chi(E_g+0.01), TNN GGA, coarse disk R=0.3/a h=0.003/a nu=5 meV:
      MoS2 +1.89e-4, MoSe2 +2.15e-5, WS2 +7.4e-6, WSe2 -4.97e-5 A m V^-2 s^-1 T^-1
      (paper: +1.9, +0.2, +0.07, -0.5 x 1e-4); the signs and magnitudes agree.
    Fig. kp_spectra's "kappa=-0.18, w=-6.5" equals the TNN reduction WITHOUT SOC (kappa = -0.179,
      w = -6.52), whereas Table II uses the active spin with SOC (kappa = -0.24, w = -6.46).
    |M| at K: 2.6-5.9 mu_B (paper "3-6"; MoSe2 M_v = 2.58). M < 0 at K for both bands.

Runtime guide: a 3x3 eigensolve plus moment derivative costs about 9 eigh per k per spin. On one
core, 31k k-points take about 5 s per field; a full N=90 grid takes about 2 s per field.
