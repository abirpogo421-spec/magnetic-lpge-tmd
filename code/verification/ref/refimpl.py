"""
refimpl.py -- independent reference implementation of the magnetic linear
photogalvanic effect (magnetic injection current) in monolayer MX2 within the
three-band tight-binding model of Liu, Shan, Yao, Yao, Xiao, PRB 88, 085433
(2013) [arXiv:1305.6089], following the paper
"Magnetic activation of the injection current in the LPGE of monolayer TMDs"
(Mehta & Adhidewata).

Written from the paper (main.tex + sections/{theory,results,appendix}.tex,
with the audit corrections listed in README) and from Liu et al. ONLY.
No other code was consulted.  numpy only.

Conventions (see README for the full list of assumptions)
---------------------------------------------------------
* k is measured in units of 1/a (a = lattice constant from Liu Table II).
  alpha = kx/2, beta = sqrt(3) ky/2.  x is along a metal-metal bond.
  Reciprocal vectors b1 = (2pi, 2pi/sqrt3), b2 = (2pi, -2pi/sqrt3);
  K = (4pi/3, 0), K' = -K, M = (pi, pi/sqrt3).
* Orbital basis {d_z2, d_xy, d_x2-y2}; L_z = [[0,0,0],[0,0,2i],[0,-2i,0]].
* Spin block s = +1 (up) / -1 (down):
      H_s(k) = H3(k) + (lam/2) s L_z + mu_B B (L_z + (g_s/2) s 1),   g_s = 2
  (b = mu_B B, no hbar; mu_B = 5.7883818060e-5 eV/T).
* Energies in eV; velocities v^a = dH/dk_a in eV (i.e. eV*a);
  Berry curvature Omega in a^2; quantum metric g in a^2; orbital moment M in mu_B.
* Berry curvature Omega_n = -2 Im sum_{m!=n} v^x_nm v^y_mn/(e_n-e_m)^2
  (= d_x A_y - d_y A_x with A = i<u|grad u>; two-band lower band:
   Omega^v = +1/2 n.(d_x n x d_y n)).
* Interband metric g^{bc}_{mn} = Re[v^b_nm v^c_mn]/(e_n-e_m)^2 (>= 0 for b=c),
  interband curvature Omega^{bc}_{mn} = 2 Im[v^b_nm v^c_mn]/(e_n-e_m)^2,
  so that Q^{mn}_{bc} = tr[P_n dP_m dP_n] = -g - (i/2) Omega.
* Itinerant orbital moment (Bohr magnetons):
      M_n = -(2 m_e a^2/hbar^2) sum_{m!=n} Im[v^x_nm v^y_mn]/(e_n - e_m)
* Field corrections (each switchable): e_n -> e_n - mu_B B M_n ;
  v_n -> v_n - mu_B B dM_n/dk ; phase-space D_n = 1 + (e/hbar) B Omega_n a^2.
* Physical magnetic injection coefficient (current of electrons, charge -e):
      eta^{a;(bc)} = -(2 pi e^3 a/hbar^2) int d^2k/(2pi)^2
                     sum_{n!=m} f~_nm Delta^a_mn g^{bc}_mn L_nu(e_mn - hbar w)
  with f~_nm = f_n D_n - f_m D_m, Delta^a_mn = v^a_m - v^a_n,
  L_nu(x) = (nu/pi)/(x^2+nu^2), summed exactly over every k-point and every
  transition (no binning, no window cut).  Result in A m V^-2 s^-1.
"""

import numpy as np

# --------------------------------------------------------------------------
# Physical constants (CODATA 2018, exact where defined)
# --------------------------------------------------------------------------
E_CHARGE = 1.602176634e-19      # C   (also J per eV)
HBAR = 1.054571817e-34          # J s
M_E = 9.1093837015e-31          # kg
MU_B = 5.7883818060e-5          # eV / T
G_S = 2.0                       # free-electron spin g-factor (paper: g_s = 2)
ANG = 1e-10                     # m

SQ3 = np.sqrt(3.0)
LZ = np.array([[0, 0, 0], [0, 0, 2j], [0, -2j, 0]], dtype=complex)
K_POINT = np.array([4 * np.pi / 3, 0.0])          # units 1/a
M_POINT = np.array([np.pi, np.pi / SQ3])
B1 = np.array([2 * np.pi, 2 * np.pi / SQ3])
B2 = np.array([2 * np.pi, -2 * np.pi / SQ3])
BZ_AREA = abs(B1[0] * B2[1] - B1[1] * B2[0])       # = 8 pi^2/sqrt3  (1/a^2)

# --------------------------------------------------------------------------
# Liu et al. parameter tables (transcribed from the arXiv source, Tables II,
# III (TNN) and IV (lambda)).  Energies in eV, a in Angstrom.
# --------------------------------------------------------------------------
MATERIALS = ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]
_NN_KEYS = ["a", "zXX", "eps1", "eps2", "t0", "t1", "t2", "t11", "t12", "t22"]
_NN = {
    "GGA": {
        "MoS2":  [3.190, 3.130, 1.046, 2.104, -0.184, 0.401, 0.507, 0.218, 0.338, 0.057],
        "WS2":   [3.191, 3.144, 1.130, 2.275, -0.206, 0.567, 0.536, 0.286, 0.384, -0.061],
        "MoSe2": [3.326, 3.345, 0.919, 2.065, -0.188, 0.317, 0.456, 0.211, 0.290, 0.130],
        "WSe2":  [3.325, 3.363, 0.943, 2.179, -0.207, 0.457, 0.486, 0.263, 0.329, 0.034],
        "MoTe2": [3.557, 3.620, 0.605, 1.972, -0.169, 0.228, 0.390, 0.207, 0.239, 0.252],
        "WTe2":  [3.560, 3.632, 0.606, 2.102, -0.175, 0.342, 0.410, 0.233, 0.270, 0.190],
    },
    "LDA": {
        "MoS2":  [3.129, 3.115, 1.238, 2.366, -0.218, 0.444, 0.533, 0.250, 0.360, 0.047],
        "WS2":   [3.132, 3.126, 1.355, 2.569, -0.238, 0.626, 0.557, 0.324, 0.405, -0.076],
        "MoSe2": [3.254, 3.322, 1.001, 2.239, -0.222, 0.350, 0.488, 0.244, 0.314, 0.129],
        "WSe2":  [3.253, 3.338, 1.124, 2.447, -0.242, 0.506, 0.514, 0.305, 0.353, 0.025],
        "MoTe2": [3.472, 3.598, 0.618, 2.126, -0.202, 0.254, 0.423, 0.241, 0.263, 0.269],
        "WTe2":  [3.476, 3.611, 0.623, 2.251, -0.209, 0.388, 0.442, 0.272, 0.295, 0.200],
    },
}
_TNN_KEYS = ["eps1", "eps2", "t0", "t1", "t2", "t11", "t12", "t22", "r0", "r1",
             "r2", "r11", "r12", "u0", "u1", "u2", "u11", "u12", "u22"]
_TNN = {
    "GGA": {
        "MoS2":  [0.683, 1.707, -0.146, -0.114, 0.506, 0.085, 0.162, 0.073, 0.060, -0.236,
                  0.067, 0.016, 0.087, -0.038, 0.046, 0.001, 0.266, -0.176, -0.150],
        "WS2":   [0.717, 1.916, -0.152, -0.097, 0.590, 0.047, 0.178, 0.016, 0.069, -0.261,
                  0.107, -0.003, 0.109, -0.054, 0.045, 0.002, 0.325, -0.206, -0.163],
        "MoSe2": [0.684, 1.546, -0.146, -0.130, 0.432, 0.144, 0.117, 0.075, 0.039, -0.209,
                  0.069, 0.052, 0.060, -0.042, 0.036, 0.008, 0.272, -0.172, -0.150],
        "WSe2":  [0.728, 1.655, -0.146, -0.124, 0.507, 0.117, 0.127, 0.015, 0.036, -0.234,
                  0.107, 0.044, 0.075, -0.061, 0.032, 0.007, 0.329, -0.202, -0.164],
        "MoTe2": [0.588, 1.303, -0.226, -0.234, 0.036, 0.400, 0.098, 0.017, 0.003, -0.025,
                  -0.169, 0.082, 0.051, 0.057, 0.103, 0.187, -0.045, -0.141, 0.087],
        "WTe2":  [0.697, 1.380, -0.109, -0.164, 0.368, 0.204, 0.093, 0.038, -0.015, -0.209,
                  0.107, 0.115, 0.009, -0.066, 0.011, -0.013, 0.312, -0.177, -0.132],
    },
    "LDA": {
        "MoS2":  [0.820, 1.931, -0.176, -0.101, 0.531, 0.084, 0.169, 0.070, 0.070, -0.252,
                  0.084, 0.019, 0.093, -0.043, 0.047, 0.005, 0.304, -0.192, -0.162],
        "WS2":   [0.905, 2.167, -0.175, -0.090, 0.611, 0.043, 0.181, 0.008, 0.075, -0.282,
                  0.127, 0.001, 0.114, -0.063, 0.047, 0.004, 0.374, -0.224, -0.177],
        "MoSe2": [0.715, 1.687, -0.154, -0.134, 0.437, 0.124, 0.119, 0.072, 0.048, -0.248,
                  0.090, 0.066, 0.045, -0.067, 0.041, 0.005, 0.327, -0.194, -0.151],
        "WSe2":  [0.860, 1.892, -0.152, -0.125, 0.508, 0.094, 0.129, 0.009, 0.044, -0.278,
                  0.129, 0.059, 0.058, -0.090, 0.039, 0.001, 0.392, -0.224, -0.165],
        "MoTe2": [0.574, 1.410, -0.148, -0.173, 0.333, 0.203, 0.186, 0.127, 0.007, -0.280,
                  0.067, 0.073, 0.081, -0.054, 0.008, 0.037, 0.145, -0.078, 0.035],
        "WTe2":  [0.675, 1.489, -0.124, -0.159, 0.362, 0.196, 0.101, 0.044, -0.009, -0.250,
                  0.129, 0.131, -0.007, -0.086, 0.012, -0.020, 0.361, -0.193, -0.129],
    },
}
# Liu Table IV: second-order corrected SOC parameter lambda (GGA).  Liu gives no
# LDA lambda; the same value is used for LDA (assumption A3 in README).
_LAMBDA = {"MoS2": 0.073, "WS2": 0.211, "MoSe2": 0.091, "WSe2": 0.228,
           "MoTe2": 0.107, "WTe2": 0.237}
# Liu Table IV: valence SOC splitting at K (GGA), for reference.
DELTA_SOC_V = {"MoS2": 0.148, "WS2": 0.430, "MoSe2": 0.184, "WSe2": 0.466,
               "MoTe2": 0.215, "WTe2": 0.486}


def params(material, model="TNN", functional="GGA"):
    """Liu parameters as a dict: a (Angstrom), on-site/hopping energies (eV),
    lam (eV).  NN models have all r_i, u_i = 0.  The TNN model uses the lattice
    constant of Liu Table II for the same functional (Table III lists none)."""
    model = model.upper()
    functional = functional.upper()
    nn = dict(zip(_NN_KEYS, _NN[functional][material]))
    p = {k: 0.0 for k in _TNN_KEYS}
    if model == "NN":
        for k in _NN_KEYS[2:]:
            p[k] = nn[k]
    elif model == "TNN":
        p.update(dict(zip(_TNN_KEYS, _TNN[functional][material])))
    else:
        raise ValueError("model must be 'NN' or 'TNN'")
    p["a"] = nn["a"]
    p["lam"] = _LAMBDA[material]
    p["material"], p["model"], p["functional"] = material, model, functional
    return p


def _resolve(material, model, functional, p=None):
    if p is not None:
        return p
    if isinstance(material, dict):
        return material
    return params(material, model, functional)


# --------------------------------------------------------------------------
# Spinless three-band Hamiltonian, written as an explicit list of trig terms
#     coef * F(na * alpha) * G(nb * beta),  F, G in {cos, sin}
# so that first and second k-derivatives are exact (analytic).
# Products such as sin(a)cos(a) have been expanded (sin2a/2); the selftest
# checks this list against a literal transcription of Liu's factored formulas.
# --------------------------------------------------------------------------
def _terms(p):
    s3 = SQ3
    e1, e2 = p["eps1"], p["eps2"]
    t0, t1, t2, t11, t12, t22 = (p[k] for k in ("t0", "t1", "t2", "t11", "t12", "t22"))
    r0, r1, r2, r11, r12 = (p[k] for k in ("r0", "r1", "r2", "r11", "r12"))
    u0, u1, u2, u11, u12, u22 = (p[k] for k in ("u0", "u1", "u2", "u11", "u12", "u22"))
    I = 1j
    T = {}
    T[(0, 0)] = [(e1, "c", 0, "c", 0), (4 * t0, "c", 1, "c", 1), (2 * t0, "c", 2, "c", 0),
                 (4 * r0, "c", 3, "c", 1), (2 * r0, "c", 0, "c", 2),
                 (4 * u0, "c", 2, "c", 2), (2 * u0, "c", 4, "c", 0)]
    T[(0, 1)] = [  # V1
        (-2 * s3 * t2, "s", 1, "s", 1), (2 * (r1 + r2), "s", 3, "s", 1),
        (-2 * s3 * u2, "s", 2, "s", 2),
        (I * 2 * t1, "s", 2, "c", 0), (I * 2 * t1, "s", 1, "c", 1),
        (I * 2 * (r1 - r2), "s", 3, "c", 1),
        (I * 2 * u1, "s", 4, "c", 0), (I * 2 * u1, "s", 2, "c", 2)]
    T[(0, 2)] = [  # V2
        (2 * t2, "c", 2, "c", 0), (-2 * t2, "c", 1, "c", 1),
        (-2 / s3 * (r1 + r2), "c", 3, "c", 1), (2 / s3 * (r1 + r2), "c", 0, "c", 2),
        (2 * u2, "c", 4, "c", 0), (-2 * u2, "c", 2, "c", 2),
        (I * 2 * s3 * t1, "c", 1, "s", 1),
        (I * 2 / s3 * (r1 - r2), "c", 3, "s", 1), (I * 2 / s3 * (r1 - r2), "c", 0, "s", 2),
        (I * 2 * s3 * u1, "c", 2, "s", 2)]
    T[(1, 1)] = [(e2, "c", 0, "c", 0), (t11 + 3 * t22, "c", 1, "c", 1), (2 * t11, "c", 2, "c", 0),
                 (4 * r11, "c", 3, "c", 1), (2 * (r11 + s3 * r12), "c", 0, "c", 2),
                 (u11 + 3 * u22, "c", 2, "c", 2), (2 * u11, "c", 4, "c", 0)]
    T[(1, 2)] = [  # V12
        (s3 * (t22 - t11), "s", 1, "s", 1), (4 * r12, "s", 3, "s", 1),
        (s3 * (u22 - u11), "s", 2, "s", 2),
        (I * 2 * t12, "s", 2, "c", 0), (I * -4 * t12, "s", 1, "c", 1),
        (I * 2 * u12, "s", 4, "c", 0), (I * -4 * u12, "s", 2, "c", 2)]
    T[(2, 2)] = [(e2, "c", 0, "c", 0), (3 * t11 + t22, "c", 1, "c", 1), (2 * t22, "c", 2, "c", 0),
                 (4 * r11, "c", 3, "c", 1), (2 * r11, "c", 0, "c", 2),
                 (8 / s3 * r12, "c", 3, "c", 1), (-2 / s3 * r12, "c", 0, "c", 2),
                 (3 * u11 + u22, "c", 2, "c", 2), (2 * u22, "c", 4, "c", 0)]
    return T


def _trig(f, n, x, d):
    """d-th derivative (d=0,1,2) of f(n x) with respect to x."""
    if f == "c":
        return [np.cos(n * x), -n * np.sin(n * x), -n * n * np.cos(n * x)][d]
    return [np.sin(n * x), n * np.cos(n * x), -n * n * np.sin(n * x)][d]


_DERIVS = {"H": (0, 0), "dx": (1, 0), "dy": (0, 1), "dxx": (2, 0), "dxy": (1, 1), "dyy": (0, 2)}


def H3(kx, ky, material="MoS2", model="TNN", functional="GGA", p=None, derivs=("H",)):
    """Spinless 3x3 Liu Hamiltonian at k = (kx, ky) in units of 1/a.

    Returns an array (..., 3, 3) if derivs == ("H",), otherwise a dict mapping
    each requested key in {"H","dx","dy","dxx","dxy","dyy"} to (..., 3, 3)
    arrays (exact analytic derivatives with respect to k in units of 1/a)."""
    p = _resolve(material, model, functional, p)
    kx = np.asarray(kx, float)
    ky = np.asarray(ky, float)
    shape = np.broadcast(kx, ky).shape
    al = np.broadcast_to(kx / 2, shape)
    be = np.broadcast_to(SQ3 * ky / 2, shape)
    T = _terms(p)
    out = {}
    for key in derivs:
        i, j = _DERIVS[key]
        fac = 0.5 ** i * (SQ3 / 2) ** j
        Hm = np.zeros(shape + (3, 3), complex)
        for (r, c), lst in T.items():
            acc = np.zeros(shape, complex)
            for coef, fa, na, fb, nb in lst:
                if coef == 0:
                    continue
                acc = acc + coef * _trig(fa, na, al, i) * _trig(fb, nb, be, j)
            acc = acc * fac
            Hm[..., r, c] = acc
            if r != c:
                Hm[..., c, r] = np.conj(acc)
        out[key] = Hm
    if tuple(derivs) == ("H",):
        return out["H"]
    return out


def zeeman_soc_block(s, B, lam, spin_zeeman=True, orbital_zeeman=True):
    """k-independent 3x3 term of spin block s (=+1 up, -1 down):
    (lam/2) s Lz + mu_B B (Lz + (g_s/2) s)."""
    b = MU_B * B
    M = 0.5 * lam * s * LZ
    if orbital_zeeman:
        M = M + b * LZ
    if spin_zeeman:
        M = M + b * 0.5 * G_S * s * np.eye(3)
    return M


def H_block(kx, ky, s, B=0.0, material="MoS2", model="TNN", functional="GGA", p=None,
            lam=None, spin_zeeman=True, orbital_zeeman=True):
    """3x3 spin block of the total Hamiltonian (paper Eq. app_htot)."""
    p = _resolve(material, model, functional, p)
    lam = p["lam"] if lam is None else lam
    return H3(kx, ky, p=p) + zeeman_soc_block(s, B, lam, spin_zeeman, orbital_zeeman)


def H6(kx, ky, B=0.0, material="MoS2", model="TNN", functional="GGA", p=None, lam=None,
       spin_zeeman=True, orbital_zeeman=True):
    """Full 6x6 H_tot in basis {up x (dz2,dxy,dx2-y2), down x (...)}, built as
    I2 (x) H3 + (lam/2) sigma_z (x) Lz + mu_B B (I2 (x) Lz + (g_s/2) sigma_z (x) I3)."""
    p = _resolve(material, model, functional, p)
    lam = p["lam"] if lam is None else lam
    h = H3(kx, ky, p=p)
    sz = np.diag([1.0, -1.0])
    b = MU_B * B
    K6 = 0.5 * lam * np.kron(sz, LZ)
    if orbital_zeeman:
        K6 = K6 + b * np.kron(np.eye(2), LZ)
    if spin_zeeman:
        K6 = K6 + b * 0.5 * G_S * np.kron(sz, np.eye(3))
    out = np.zeros(h.shape[:-2] + (6, 6), complex)
    out[..., :3, :3] = h
    out[..., 3:, 3:] = h
    return out + K6


# --------------------------------------------------------------------------
# Generic band geometry from H and its analytic first derivatives
# --------------------------------------------------------------------------
def eig_velocity(H, dHx, dHy):
    """Diagonalize a batch of Hermitian matrices; return energies (..., n),
    eigenvectors U (..., n, n) (columns) and velocity matrices
    V^a_{nm} = <u_n|dH/dk_a|u_m> (..., n, n)."""
    E, U = np.linalg.eigh(H)
    Ud = np.conj(np.swapaxes(U, -1, -2))
    Vx = Ud @ dHx @ U
    Vy = Ud @ dHy @ U
    return E, U, Vx, Vy


def _offdiag_denoms(E, power):
    dE = E[..., :, None] - E[..., None, :]          # e_n - e_m
    n = E.shape[-1]
    eye = np.eye(n, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(eye, 0.0, 1.0 / np.where(eye, 1.0, dE) ** power)
    return inv


def berry_and_moment(E, Vx, Vy, a_ang):
    """Berry curvature Omega_n (units a^2) and itinerant orbital moment M_n
    (units mu_B) from the velocity sum rules."""
    X = np.imag(Vx * np.swapaxes(Vy, -1, -2))       # Im[v^x_nm v^y_mn]
    Om = -2.0 * np.sum(X * _offdiag_denoms(E, 2), axis=-1)
    cM = 2 * M_E * (a_ang * ANG) ** 2 / HBAR ** 2 * E_CHARGE   # 2 m_e a^2/hbar^2 in 1/eV
    Mn = -cM * np.sum(X * _offdiag_denoms(E, 1), axis=-1)
    return Om, Mn


def interband_geometry(E, Vb, Vc):
    """Interband metric g^{bc}_{mn} and curvature Omega^{bc}_{mn} as (..., n, m)
    arrays indexed [n, m] (units a^2):
        g = Re[v^b_nm v^c_mn]/(e_n-e_m)^2,  Omega = 2 Im[v^b_nm v^c_mn]/(e_n-e_m)^2."""
    P = Vb * np.swapaxes(Vc, -1, -2)                 # v^b_nm v^c_mn
    inv2 = _offdiag_denoms(E, 2)
    return np.real(P) * inv2, 2 * np.imag(P) * inv2


# --------------------------------------------------------------------------
# Band quantities of the TB model per k and spin block
# --------------------------------------------------------------------------
def band_quantities(kx, ky, s, B=0.0, material="MoS2", model="TNN", functional="GGA",
                    p=None, lam=None, spin_zeeman=True, orbital_zeeman=True,
                    dM_step=1e-3, need_dM=True):
    """All per-band quantities for spin block s at field B (Tesla).

    Returns dict with
      E      (Nk,3)   bare eigenvalues of H_s (eV)
      U      (Nk,3,3) eigenvectors
      Vx, Vy (Nk,3,3) velocity matrices in the eigenbasis (eV, i.e. eV*a)
      v      (2,Nk,3) band velocities v^a_n = V^a_nn (Hellmann-Feynman)
      Omega  (Nk,3)   Berry curvature (a^2)
      M      (Nk,3)   orbital moment (mu_B)
      D      (Nk,3)   phase-space factor 1 + (e/hbar) B Omega a^2
      E_shift(Nk,3)   e_n - mu_B B M_n
      dM     (2,Nk,3) dM_n/dk_a  (mu_B * a), 4th-order central differences with
                      step dM_step (1/a); error O(h^4) ~ 1e-12 relative at h=1e-3
      v_shift(2,Nk,3) -mu_B B dM_n/dk_a (eV*a)
    """
    p = _resolve(material, model, functional, p)
    lam = p["lam"] if lam is None else lam
    kx = np.atleast_1d(np.asarray(kx, float))
    ky = np.atleast_1d(np.asarray(ky, float))
    Kz = zeeman_soc_block(s, B, lam, spin_zeeman, orbital_zeeman)

    def core(qx, qy):
        d = H3(qx, qy, p=p, derivs=("H", "dx", "dy"))
        E, U, Vx, Vy = eig_velocity(d["H"] + Kz, d["dx"], d["dy"])
        Om, Mn = berry_and_moment(E, Vx, Vy, p["a"])
        return E, U, Vx, Vy, Om, Mn

    E, U, Vx, Vy, Om, Mn = core(kx, ky)
    v = np.stack([np.real(np.diagonal(Vx, axis1=-2, axis2=-1)),
                  np.real(np.diagonal(Vy, axis1=-2, axis2=-1))])
    b = MU_B * B
    out = dict(E=E, U=U, Vx=Vx, Vy=Vy, v=v, Omega=Om, M=Mn,
               D=1.0 + E_CHARGE / HBAR * B * Om * (p["a"] * ANG) ** 2,
               E_shift=E - b * Mn)
    if need_dM:
        h = dM_step
        dM = []
        for ex, ey in ((1, 0), (0, 1)):
            f = {}
            for st in (-2, -1, 1, 2):
                f[st] = core(kx + st * h * ex, ky + st * h * ey)[5]
            dM.append((-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * h))
        out["dM"] = np.stack(dM)
        out["v_shift"] = -b * out["dM"]
    return out


# --------------------------------------------------------------------------
# k-grids
# --------------------------------------------------------------------------
def bz_grid(N):
    """Gamma-centred N x N Monkhorst-Pack grid k = (i b1 + j b2)/N, i,j=0..N-1.
    Exactly inversion paired (k -> -k maps the grid onto itself mod G) and
    C3- and mirror(x->-x)-invariant mod G.  Returns k (N^2,2) in 1/a and the
    area element dA (1/a^2) so that int d^2k f = sum dA f."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    k = (i.reshape(-1, 1) * B1 + j.reshape(-1, 1) * B2) / N
    return k, BZ_AREA / N ** 2


def disk_grid(R, h, valleys=("K", "Kp")):
    """Square-lattice points q = (i h, j h) with |q| <= R around K and around
    K' = -K.  The K' disk is constructed as exactly -(K disk), so the two disks
    are mapped onto each other by k -> -k.  Returns k (Nk,2) and dA = h^2."""
    n = int(np.floor(R / h)) + 1
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    q = np.stack([i.ravel() * h, j.ravel() * h], axis=1)
    q = q[np.hypot(q[:, 0], q[:, 1]) <= R * (1 + 1e-12)]
    pts = []
    for vname in valleys:
        if vname == "K":
            pts.append(K_POINT + q)
        elif vname in ("Kp", "K'"):
            pts.append(-(K_POINT + q))
        else:
            raise ValueError(vname)
    return np.concatenate(pts, axis=0), h * h


def c3_expand(k, dA):
    """Replace each k by {k, C3 k, C3^2 k} with weight dA/3 (C3 about Gamma).
    Symmetrizes any k-sum under C3 while keeping inversion pairing."""
    c, s_ = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    R = np.array([[c, -s_], [s_, c]])
    ks = [k, k @ R.T, k @ (R @ R).T]
    dA = np.broadcast_to(np.asarray(dA, float), (k.shape[0],))
    return np.concatenate(ks), np.concatenate([dA / 3] * 3)


# --------------------------------------------------------------------------
# Injection current (magnetic, linear polarisation) and electric injection
# --------------------------------------------------------------------------
DEFAULT_FLAGS = dict(spin_zeeman=True, orbital_zeeman=True, moment_energy=True,
                     phase_space=True, velocity_shift=True, c3=False, pairs="resonant")


def _lorentz(x, nu):
    return (nu / np.pi) / (x * x + nu * nu)


def prefactor_injection(a_ang):
    """-2 pi e^3 a / hbar^2 (SI).  Multiplies sum_k dA/(2pi)^2 f Delta[eV] g L[1/eV]
    (dimensionless) to give A m V^-2 s^-1."""
    return -2 * np.pi * E_CHARGE ** 3 * (a_ang * ANG) / HBAR ** 2


def _transitions(bq, flags):
    """Yield (n, m, ftilde, e_mn, Delta^a_mn (2,Nk)) for every ordered pair
    contributing.  Occupation: lowest band of each spin block filled (T=0
    insulator), the two upper bands empty."""
    f = np.array([1.0, 0.0, 0.0])
    E_res = bq["E_shift"] if flags["moment_energy"] else bq["E"]
    vel = bq["v"] + (bq["v_shift"] if flags["velocity_shift"] else 0.0)
    D = bq["D"] if flags["phase_space"] else np.ones_like(bq["E"])
    for n in range(3):
        for m in range(3):
            if n == m:
                continue
            if f[n] == f[m]:
                continue
            if flags["pairs"] == "resonant" and not (f[n] == 1 and f[m] == 0):
                continue
            ft = f[n] * D[:, n] - f[m] * D[:, m]
            yield n, m, ft, E_res[:, m] - E_res[:, n], vel[:, :, m] - vel[:, :, n]


def eta_tensor(material="MoS2", model="TNN", functional="GGA", B=0.0, omega=2.0, nu=0.01,
               kpoints=None, dA=None, flags=None, p=None, lam=None, spins=(1, -1),
               electric=False, chunk=20000, dM_step=1e-3):
    """Physical injection tensor eta^{a;bc}(omega) in A m V^-2 s^-1.

    omega : photon energy hbar*omega in eV (scalar or 1-D array)
    nu    : Lorentzian half width (eV)
    kpoints, dA : k-points (Nk,2) in 1/a and area weights (scalar or (Nk,)) in 1/a^2
    flags : dict overriding DEFAULT_FLAGS:
        spin_zeeman, orbital_zeeman (in H_Z), moment_energy (e - mu_B B M),
        phase_space (D_v), velocity_shift (-mu_B B dM/dk), c3 (C3-expand grid),
        pairs = 'resonant' (n occupied -> m empty, L(e_mn - w); default, as in
                the paper's master formula) or 'all' (literal sum over m != n,
                adds the anti-resonant Lorentzian tails).
    Returns array (Nw, 2, 2, 2) indexed [w, a, b, c] (magnetic: symmetric metric);
    if electric=True also returns the electric-injection tensor built from the
    interband Berry curvature with prefactor -pi e^3 a/hbar^2.
    """
    p = _resolve(material, model, functional, p)
    fl = dict(DEFAULT_FLAGS)
    if flags:
        fl.update(flags)
    if kpoints is None:
        kpoints, dA = bz_grid(60)
    kpoints = np.asarray(kpoints, float)
    dA = np.broadcast_to(np.asarray(dA, float), (kpoints.shape[0],))
    if fl["c3"]:
        kpoints, dA = c3_expand(kpoints, dA)
    w = np.atleast_1d(np.asarray(omega, float))
    acc = np.zeros((w.size, 2, 2, 2))
    acc_el = np.zeros((w.size, 2, 2, 2))
    need_dM = fl["velocity_shift"] and B != 0.0
    for s in spins:
        for i0 in range(0, kpoints.shape[0], chunk):
            ks = kpoints[i0:i0 + chunk]
            wk = dA[i0:i0 + chunk] / (2 * np.pi) ** 2
            bq = band_quantities(ks[:, 0], ks[:, 1], s, B, p=p, lam=lam,
                                 spin_zeeman=fl["spin_zeeman"],
                                 orbital_zeeman=fl["orbital_zeeman"],
                                 need_dM=need_dM, dM_step=dM_step)
            if not need_dM:
                bq["v_shift"] = np.zeros_like(bq["v"])
            V = (bq["Vx"], bq["Vy"])
            G = {}
            for b_ in range(2):
                for c_ in range(2):
                    G[b_, c_] = interband_geometry(bq["E"], V[b_], V[c_])
            for n, m, ft, emn, Dl in _transitions(bq, fl):
                L = _lorentz(emn[None, :] - w[:, None], nu)          # (Nw, Nk)
                for a_ in range(2):
                    base = wk * ft * Dl[a_]
                    for b_ in range(2):
                        for c_ in range(2):
                            g, Om = G[b_, c_]
                            acc[:, a_, b_, c_] += L @ (base * g[:, n, m])
                            if electric:
                                acc_el[:, a_, b_, c_] += L @ (base * Om[:, n, m])
    pref = prefactor_injection(p["a"])
    if electric:
        return pref * acc, 0.5 * pref * acc_el
    return pref * acc


def eta_xxx(material="MoS2", model="TNN", functional="GGA", B=0.0, omega=2.0, nu=0.01,
            kpoints=None, dA=None, flags=None, **kw):
    """Physical magnetic injection coefficient eta^{x;xx} (A m V^-2 s^-1)."""
    return eta_tensor(material, model, functional, B, omega, nu, kpoints, dA, flags, **kw)[:, 0, 0, 0]


# --------------------------------------------------------------------------
# Shift current (tight binding), projector formalism, for completeness
# --------------------------------------------------------------------------
def _dP_analytic(E, U, Vx, Vy):
    """Analytic first derivatives of the band projectors in the orbital basis:
    dP_n = sum_{l!=n} (|l><l|dH|n><n| + h.c.)/(e_n - e_l).  Returns (2, Nk, n, d, d)."""
    nb = E.shape[-1]
    inv = _offdiag_denoms(E, 1)          # [n, l] -> 1/(e_n - e_l)
    out = []
    Ud = np.conj(np.swapaxes(U, -1, -2))
    for V in (Vx, Vy):
        per = []
        for n in range(nb):
            Xn = np.zeros_like(V)
            Xn[..., :, n] = V[..., :, n] * inv[..., n, :]          # |l><l|dH|n>/(e_n-e_l)
            Xn = Xn + np.conj(np.swapaxes(Xn, -1, -2))
            per.append(U @ Xn @ Ud)
        out.append(np.stack(per, axis=1))
    return np.stack(out)


def sigma_shift_tensor(material="MoS2", model="TNN", functional="GGA", B=0.0, omega=2.0,
                       nu=0.01, kpoints=None, dA=None, flags=None, p=None, lam=None,
                       spins=(1, -1), fd_step=1e-4, chunk=20000):
    """Shift-current tensor sigma^{a;(bc)} in A m V^-2 (multiply by 1e15 for
    nm muA V^-2), from
        sigma = -(2 pi e^3 a/hbar) int d^2k/(2pi)^2 sum_{n!=m} f~_nm Im Cbar^{mn}_{a;(bc)} L,
    C^{mn}_{a;bc} = tr[P_n dP_m^b (d_a d_c P_n + dP_m^a dP_n^c)], Cbar = (C^{mn}-C^{nm})/2.
    First projector derivatives analytic; d_a d_c P_n by central differences
    of the analytic dP_n with step fd_step (1/a).  Same flags as eta_tensor
    (velocity_shift is irrelevant here)."""
    p = _resolve(material, model, functional, p)
    fl = dict(DEFAULT_FLAGS)
    if flags:
        fl.update(flags)
    kpoints = np.asarray(kpoints, float)
    dA = np.broadcast_to(np.asarray(dA, float), (kpoints.shape[0],))
    if fl["c3"]:
        kpoints, dA = c3_expand(kpoints, dA)
    w = np.atleast_1d(np.asarray(omega, float))
    acc = np.zeros((w.size, 2, 2, 2))
    lam_ = p["lam"] if lam is None else lam
    Kz_ = {s: zeeman_soc_block(s, B, lam_, fl["spin_zeeman"], fl["orbital_zeeman"]) for s in spins}

    def projdata(qx, qy, s):
        d = H3(qx, qy, p=p, derivs=("H", "dx", "dy"))
        E, U, Vx, Vy = eig_velocity(d["H"] + Kz_[s], d["dx"], d["dy"])
        P = np.einsum("kin,kjn->knij", U, np.conj(U))
        return E, P, _dP_analytic(E, U, Vx, Vy)

    for s in spins:
        for i0 in range(0, kpoints.shape[0], chunk):
            ks = kpoints[i0:i0 + chunk]
            wk = dA[i0:i0 + chunk] / (2 * np.pi) ** 2
            bq = band_quantities(ks[:, 0], ks[:, 1], s, B, p=p, lam=lam,
                                 spin_zeeman=fl["spin_zeeman"], orbital_zeeman=fl["orbital_zeeman"],
                                 need_dM=False)
            E, P, dP = projdata(ks[:, 0], ks[:, 1], s)
            ddP = np.zeros((2, 2) + dP.shape[1:], complex)        # [a, c] d_a d_c P
            for c_, (ex, ey) in enumerate(((1, 0), (0, 1))):
                dPp = projdata(ks[:, 0] + fd_step * ex, ks[:, 1] + fd_step * ey, s)[2]
                dPm = projdata(ks[:, 0] - fd_step * ex, ks[:, 1] - fd_step * ey, s)[2]
                ddP[:, c_] = (dPp - dPm) / (2 * fd_step)
            ddP = 0.5 * (ddP + np.swapaxes(ddP, 0, 1))             # symmetrize mixed derivative
            bq["v_shift"] = np.zeros_like(bq["v"])
            fl2 = dict(fl, velocity_shift=False)
            for n, m, ft, emn, _ in _transitions(bq, fl2):
                L = _lorentz(emn[None, :] - w[:, None], nu)
                for a_ in range(2):
                    for b_ in range(2):
                        for c_ in range(2):
                            def C(mm, nn, bb, cc):
                                X = ddP[a_, cc][:, nn] + dP[a_][:, mm] @ dP[cc][:, nn]
                                return np.trace(P[:, nn] @ dP[bb][:, mm] @ X, axis1=-2, axis2=-1)
                            Cs = 0.5 * (C(m, n, b_, c_) + C(m, n, c_, b_))
                            Cr = 0.5 * (C(n, m, b_, c_) + C(n, m, c_, b_))
                            ImCbar = np.imag(0.5 * (Cs - Cr))
                            acc[:, a_, b_, c_] += L @ (wk * ft * ImCbar)
    pref = -2 * np.pi * E_CHARGE ** 2 * (p["a"] * ANG) / HBAR    # L in 1/eV -> 1/J absorbs one e
    return pref * acc


# --------------------------------------------------------------------------
# Band-gap helper
# --------------------------------------------------------------------------
def direct_gap_K(material="MoS2", model="TNN", functional="GGA", p=None, lam=None):
    """E_g = e_c(K) - e_v(K) for the active (spin-up at K) block at B=0 (eV)."""
    p = _resolve(material, model, functional, p)
    E = np.linalg.eigvalsh(H_block(K_POINT[0], K_POINT[1], +1, 0.0, p=p, lam=lam))
    return E[1] - E[0]


# --------------------------------------------------------------------------
# Lowdin reduction at K onto {d_z2, (d_x2-y2 + i tau d_xy)/sqrt2}
# --------------------------------------------------------------------------
def lowdin_basis(tau):
    """Columns: c = d_z2, v = (d_x2-y2 + i tau d_xy)/sqrt2, u = (d_x2-y2 - i tau d_xy)/sqrt2
    in the orbital basis (d_z2, d_xy, d_x2-y2)."""
    r = 1 / np.sqrt(2)
    return np.array([[1, 0, 0],
                     [0, 1j * tau * r, -1j * tau * r],
                     [0, r, r]], dtype=complex)


def kappa_lowdin(material="MoS2", model="TNN", functional="GGA", spin_active=True,
                 with_soc=True, tau=+1, p=None, lam=None, B=0.0):
    """Second-order Lowdin reduction of the TB model about tau*K onto {c, v}.

    H_eff,ij(q) = Ht_ij(q) + 1/2 Ht_iu Ht_uj [1/(E_i-E_u) + 1/(E_j-E_u)],  i,j in {c,v}
    with Ht = W^dag H_s(tau K + q) W, expanded to O(q^2) using exact analytic
    first and second derivatives at tau K.  Writes
        H_eff,cv = v (tau qx - i qy) + kappa (tau qx + i qy)^2 + O(q^3)
    and gauges the v state so that v is real positive.  Returns dict with
    v (eV A), kappa (eV A^2), their values in a-units, gamma3 = 2 kappa/a^2 (eV),
    w = (gamma1-gamma2) a^2/4 (eV A^2), E_c, E_v, E_u (eV), residuals of the
    fit (should vanish by symmetry), and the paper's NN closed forms."""
    p = _resolve(material, model, functional, p)
    lam_ = (p["lam"] if lam is None else lam) if with_soc else 0.0
    s = tau if spin_active else -tau
    Kt = tau * K_POINT
    d = H3(Kt[0], Kt[1], p=p, derivs=tuple(_DERIVS))
    W = lowdin_basis(tau)
    Wd = W.conj().T
    Kz = zeeman_soc_block(s, B, lam_)
    H0 = Wd @ (d["H"] + Kz) @ W
    Hx, Hy = Wd @ d["dx"] @ W, Wd @ d["dy"] @ W
    Hxx, Hxy, Hyy = Wd @ d["dxx"] @ W, Wd @ d["dxy"] @ W, Wd @ d["dyy"] @ W
    offdiag_K = np.max(np.abs(H0 - np.diag(np.diag(H0))))
    Ec, Ev, Eu = np.real(np.diag(H0))
    E = {0: Ec, 1: Ev, 2: Eu}

    def quad(i, j):
        """Coefficients (A_xx, A_xy+A_yx, A_yy) of the O(q^2) part of H_eff,ij."""
        fac = 0.5 * (1 / (E[i] - Eu) + 1 / (E[j] - Eu))
        Axx = 0.5 * Hxx[i, j] + fac * Hx[i, 2] * Hx[2, j]
        Ayy = 0.5 * Hyy[i, j] + fac * Hy[i, 2] * Hy[2, j]
        Axy = Hxy[i, j] + fac * (Hx[i, 2] * Hy[2, j] + Hy[i, 2] * Hx[2, j])
        return Axx, Axy, Ayy

    # linear part: Hx_cv qx + Hy_cv qy = v (tau qx - i qy)
    v_raw = tau * Hx[0, 1]
    lin_resid = abs(Hy[0, 1] - (-1j) * v_raw) / abs(v_raw)
    phase = v_raw / abs(v_raw)
    Axx, Axy, Ayy = quad(0, 1)
    # kappa (tau qx + i qy)^2 = kappa (qx^2 - qy^2 + 2 i tau qx qy)
    kap_raw = Axx
    quad_resid = max(abs(Ayy + kap_raw), abs(Axy - 2j * tau * kap_raw)) / max(abs(kap_raw), 1e-300)
    kap = kap_raw / phase
    v = abs(v_raw)
    cxx, cxy_c, cyy = quad(0, 0)
    vxx, vxy_v, vyy = quad(1, 1)
    gam1_half = np.real(cxx)        # (a^2/2) gamma1 in a-units  (H_cc = gamma1/2 k^2)
    gam2_half = np.real(vxx)
    a = p["a"]
    out = dict(v_a=v, kappa_a2=np.real(kap), kappa_imag_a2=np.imag(kap),
               v=v * a, kappa=np.real(kap) * a ** 2, gamma3=2 * np.real(kap),
               w=(gam1_half - gam2_half) / 2 * a ** 2,
               gamma1=2 * gam1_half, gamma2=2 * gam2_half,
               E_c=Ec, E_v=Ev, E_u=Eu, offdiag_at_K=offdiag_K,
               lin_resid=lin_resid, quad_resid=quad_resid,
               diag_isotropy_resid=max(abs(cxx - cyy), abs(cxy_c), abs(vxx - vyy), abs(vxy_v)),
               phase=phase)
    if p["model"] == "NN" or all(p[k] == 0 for k in ("r0", "r1", "u0", "u1")):
        t1, t2, t11, t12, t22 = (p[k] for k in ("t1", "t2", "t11", "t12", "t22"))
        g3 = 3 * np.sqrt(2) / 8 * (t2 - SQ3 * t1) \
            + 9 * np.sqrt(6) / 16 * (t1 - SQ3 * t2) * (t11 - t22) * (1 / (Ec - Eu) + 1 / (Ev - Eu))
        out["gamma3_closed_paper"] = g3
        out["kappa_closed_paper"] = g3 / 2 * a ** 2
        out["t_closed_paper"] = 3 * np.sqrt(2) / 4 * (t1 + SQ3 * t2)
    return out


# --------------------------------------------------------------------------
# Two-band k.p (warped massive Dirac) kernels
# --------------------------------------------------------------------------
def kp_dvec(kx, ky, v, kappa, m, tau=+1, deriv=False):
    """d = (v tau kx + kappa (kx^2-ky^2), v ky - 2 kappa tau kx ky, m) and,
    if deriv, its first and second derivatives (exact)."""
    kx = np.asarray(kx, float)
    ky = np.asarray(ky, float)
    d = np.stack([v * tau * kx + kappa * (kx ** 2 - ky ** 2),
                  v * ky - 2 * kappa * tau * kx * ky,
                  np.full(np.broadcast(kx, ky).shape, float(m))], axis=-1)
    if not deriv:
        return d
    z = np.zeros_like(kx * ky)
    one = np.ones_like(z)
    dx = np.stack([v * tau * one + 2 * kappa * kx, -2 * kappa * tau * ky, z], axis=-1)
    dy = np.stack([-2 * kappa * ky, v * one - 2 * kappa * tau * kx, z], axis=-1)
    dxx = np.stack([2 * kappa * one, z, z], axis=-1)
    dyy = np.stack([-2 * kappa * one, z, z], axis=-1)
    dxy = np.stack([z, -2 * kappa * tau * one, z], axis=-1)
    return d, {"x": dx, "y": dy, "xx": dxx, "yy": dyy, "xy": dxy, "yx": dxy}


def _n_derivs(d, D):
    """Unit vector n = d/|d| with exact first and second derivatives."""
    dn = np.linalg.norm(d, axis=-1)
    n = d / dn[..., None]
    out1 = {}
    for a in "xy":
        da = D[a]
        out1[a] = da / dn[..., None] - d * (np.sum(d * da, -1) / dn ** 3)[..., None]
    out2 = {}
    for a in "xy":
        for c in "xy":
            da, dc, dac = D[a], D[c], D[a + c]
            sa, sc = np.sum(d * da, -1), np.sum(d * dc, -1)
            sac = np.sum(dc * da, -1) + np.sum(d * dac, -1)
            out2[a + c] = (dac / dn[..., None] - da * (sc / dn ** 3)[..., None]
                           - dc * (sa / dn ** 3)[..., None] - d * (sac / dn ** 3)[..., None]
                           + 3 * d * (sa * sc / dn ** 5)[..., None])
    return dn, n, out1, out2


def kp_kernels(kx, ky, v, kappa, m, tau=+1):
    """Two-band geometric kernels at k: Delta^a_cv = 2 d_a|d|,
    g_ab = 1/4 d_a n . d_b n, Omega^v = +1/2 n.(d_x n x d_y n), and
    the shift kernel C^{cv}_{a;bc} = tr[P_v dP_c^b (d_a d_c P_v + dP_c^a dP_v^c)]
    (all 8 components) and C^{vc} likewise; P_+- = (1 +- n.sigma)/2."""
    d, D = kp_dvec(kx, ky, v, kappa, m, tau, deriv=True)
    dn, n, n1, n2 = _n_derivs(d, D)
    Delta = {a: 2 * np.sum(n * D[a], -1) for a in "xy"}
    g = {a + b: 0.25 * np.sum(n1[a] * n1[b], -1) for a in "xy" for b in "xy"}
    Om_v = 0.5 * np.sum(n * np.cross(n1["x"], n1["y"]), -1)
    sig = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)

    def S(vec):
        return np.einsum("...i,ijk->...jk", vec, sig)
    I2 = np.eye(2)
    Pc, Pv = 0.5 * (I2 + S(n)), 0.5 * (I2 - S(n))
    dPc = {a: 0.5 * S(n1[a]) for a in "xy"}
    dPv = {a: -0.5 * S(n1[a]) for a in "xy"}
    ddPv = {ac: -0.5 * S(n2[ac]) for ac in n2}
    ddPc = {ac: 0.5 * S(n2[ac]) for ac in n2}
    Ccv, Cvc = {}, {}
    for a in "xy":
        for b in "xy":
            for c in "xy":
                X = ddPv[a + c] + dPc[a] @ dPv[c]
                Ccv[a + b + c] = np.trace(Pv @ dPc[b] @ X, axis1=-2, axis2=-1)
                Y = ddPc[a + c] + dPv[a] @ dPc[c]
                Cvc[a + b + c] = np.trace(Pc @ dPv[b] @ Y, axis1=-2, axis2=-1)
    return dict(d=d, dabs=dn, n=n, Delta=Delta, g=g, Omega_v=Om_v, Ccv=Ccv, Cvc=Cvc)


def kp_ring(k, v, kappa, m, tau=+1, nphi=2048):
    """Exact ring averages <.>_phi at |k| = k (trapezoid rule on a periodic
    integrand, spectrally accurate) of
       inj[abc] = Delta^a_cv g_bc,  el[a] = Delta^a_cv Omega^v,
       shift[abc] = Im C^{cv}_{a;(bc)},  shiftbar[abc] = Im Cbar^{cv}_{a;(bc)}."""
    phi = 2 * np.pi * np.arange(nphi) / nphi
    kk = kp_kernels(k * np.cos(phi), k * np.sin(phi), v, kappa, m, tau)
    out = {"inj": {}, "el": {}, "shift": {}, "shiftbar": {}}
    for a in "xy":
        out["el"][a] = np.mean(kk["Delta"][a] * kk["Omega_v"])
        for b in "xy":
            for c in "xy":
                out["inj"][a + b + c] = np.mean(kk["Delta"][a] * kk["g"][b + c])
                Cs = 0.5 * (kk["Ccv"][a + b + c] + kk["Ccv"][a + c + b])
                Cr = 0.5 * (kk["Cvc"][a + b + c] + kk["Cvc"][a + c + b])
                out["shift"][a + b + c] = np.mean(np.imag(Cs))
                out["shiftbar"][a + b + c] = np.mean(np.imag(0.5 * (Cs - Cr)))
    return out


def kp_inj_closed_lead(k, v, kappa, m):
    """Paper Eq. (app_injlead): kappa v^3 k^2 / m^3."""
    return kappa * v ** 3 * k ** 2 / m ** 3


def kp_inj_closed_next(k, v, kappa, m):
    """Paper Eq. (app_injfull): kappa v k^2 (8 m^2 v^2 - 21 k^2 v^4 + 16 k^2 m^2 kappa^2)/(8 m^5)."""
    return kappa * v * k ** 2 * (8 * m ** 2 * v ** 2 - 21 * k ** 2 * v ** 4
                                 + 16 * k ** 2 * m ** 2 * kappa ** 2) / (8 * m ** 5)


def kp_shift_closed(k, v, kappa, m):
    """Paper Eq. (app_shift): kappa v (2 m^2 - 3 k^2 v^2)/(4 m^4)."""
    return kappa * v * (2 * m ** 2 - 3 * k ** 2 * v ** 2) / (4 * m ** 4)


def kp_H(kx, ky, v, kappa, m, tau=+1, eps0=None):
    """2x2 H = d.sigma (+ eps0) and its analytic first derivatives, for feeding
    the generic TB machinery (eig_velocity etc.)."""
    d, D = kp_dvec(kx, ky, v, kappa, m, tau, deriv=True)
    sig = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
    S = lambda vec: np.einsum("...i,ijk->...jk", vec, sig)
    return S(d), S(D["x"]), S(D["y"])
