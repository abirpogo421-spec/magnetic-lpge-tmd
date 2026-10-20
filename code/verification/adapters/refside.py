"""Thin helpers around the frozen reference implementation (ref/refimpl.py, not modified).

Only argument shaping lives here: engine k-arrays -> (Nk,2), variant names -> reference flags.
The two-band extension (two_band_eta) adds the mass curvature w k^2 that refimpl's kp_dvec omits
(README A11); it reuses refimpl's generic eig/velocity/geometry routines with analytic derivatives.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ref"))
import refimpl as R
from engine import REF_FLAGS


def K(kpts):
    kx, ky, dA = kpts
    return np.stack([np.asarray(kx, float), np.asarray(ky, float)], 1), dA


def ref_eta(material, model, functional, B, omega, nu, variant, kpts, c3=False, lam=None, p=None, dA=None):
    """dA: the reference's OWN area element (h^2 for disks, BZ_AREA/N^2 for an N x N grid). Callers pass it so that a
    wrong area element in the engine (local_kpts / build_grid) cannot cancel out of the comparison (red team RT18)."""
    k, dA_eng = K(kpts)
    if dA is None:
        raise ValueError("ref_eta needs the reference's own dA (disk_dA(h) or grid_dA(N))")
    fl = dict(REF_FLAGS[variant]); fl["c3"] = c3
    return R.eta_tensor(material, model, functional, float(B), np.atleast_1d(np.asarray(omega, float)), nu,
                        k, dA, flags=fl, lam=lam, p=p)


def disk_dA(h):
    return h * h


def grid_dA(N):
    return R.BZ_AREA / N ** 2


def eg(material, model="TNN", functional="GGA", lam=None):
    return float(R.direct_gap_K(material, model, functional, lam=lam))


# ---------------------------------------------------------------- C3 tensor average (independent of tb_core)
def c3_tensor(T):
    out = np.zeros_like(T)
    for n in range(3):
        th = 2 * np.pi * n / 3
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        out += np.einsum("ia,jb,kc,...abc->...ijk", Rm, Rm, Rm, T)
    return out / 3


# ---------------------------------------------------------------- two-band warped Dirac with w k^2
SIG = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)


def _S(vec):
    return np.einsum("...i,ijk->...jk", vec, SIG)


def two_band_H(kx, ky, v, kappa, mD, w, tau, s, b, lam):
    """d = (v tau kx + kappa(kx^2-ky^2), v ky - 2 kappa tau kx ky, mD + w k^2 - tau(b + s lam/2)),
    the d-vector convention of kp2 / paper Eq. (app_dvec) plus the w k^2 mass curvature."""
    d, D = R.kp_dvec(kx, ky, v, kappa, 0.0, tau, deriv=True)
    d = d.copy(); dx = D["x"].copy(); dy = D["y"].copy()
    d[..., 2] = mD + w * (kx ** 2 + ky ** 2) - tau * (b + s * lam / 2)
    dx[..., 2] = 2 * w * kx; dy[..., 2] = 2 * w * ky
    return _S(d), _S(dx), _S(dy)


def two_band_eta(kx, ky, dA, omega, nu, p, b, pairs=((1, 1), (-1, -1))):
    """Code-unit two-band injection tensor pi*MEAS*sum Delta g L (same normalization as kp2 before its
    C3 average), using refimpl's generic eigen/velocity/geometry machinery. k in 1/Angstrom."""
    w_ = np.atleast_1d(np.asarray(omega, float))
    acc = np.zeros((w_.size, 2, 2, 2))
    for tau, s in pairs:
        H, Hx, Hy = two_band_H(kx, ky, p["v"], p["kappa"], p["mD"], p["beta"], tau, s, b, p["lam"])
        E, U, Vx, Vy = R.eig_velocity(H, Hx, Hy)
        V = (Vx, Vy)
        Dl = np.stack([np.real(Vx[:, 1, 1] - Vx[:, 0, 0]), np.real(Vy[:, 1, 1] - Vy[:, 0, 0])])
        L = R._lorentz((E[:, 1] - E[:, 0])[None, :] - w_[:, None], nu)
        for a in range(2):
            for bb in range(2):
                for cc in range(2):
                    g, _ = R.interband_geometry(E, V[bb], V[cc])
                    acc[:, a, bb, cc] += L @ (Dl[a] * g[:, 0, 1])
    MEAS = dA / (2 * np.pi) ** 2
    T = c3_tensor(acc * np.pi * MEAS)
    return 0.5 * (T + np.swapaxes(T, -1, -2))
