"""Adapter: production engine -> SI physical units, with the same call shapes as the reference.

The adapter never changes engine numerics. It only
  * imports lpge / tb_core / tmd_params / kp2 from the code root under test,
  * converts the engine's code-unit tensor to physical SI units: eta_phys = -2 * si_factor * eta_code,
  * exposes the engine's own k-arrays (full-BZ grid, local disks) in Cartesian 1/a,
  * maps the variant names Z, D, E, V, ED, EV, F onto engine flags and reference flags (protocol sec. 3).
"""
import os, sys, importlib
import numpy as np

L = tb = tp = kp2 = None
CODE_ROOT = None

# engine flags (protocol sec. 3 flag map)
FLAGS = {
    "Z": dict(spin=True),
    "D": dict(spin=True, dos="v"),
    "E": dict(spin=True, mom=True),
    "V": dict(spin=True, vel=True),
    "ED": dict(spin=True, mom=True, dos="v"),
    "EV": dict(spin=True, mom=True, vel=True),
    "F": dict(spin=True, mom=True, dos="v", vel=True),
}
# reference flags; the orbital Zeeman term is always on in both codes
REF_FLAGS = {
    "Z": dict(moment_energy=False, phase_space=False, velocity_shift=False),
    "D": dict(moment_energy=False, phase_space=True, velocity_shift=False),
    "E": dict(moment_energy=True, phase_space=False, velocity_shift=False),
    "V": dict(moment_energy=False, phase_space=False, velocity_shift=True),
    "ED": dict(moment_energy=True, phase_space=True, velocity_shift=False),
    "EV": dict(moment_energy=True, phase_space=False, velocity_shift=True),
    "F": dict(moment_energy=True, phase_space=True, velocity_shift=True),
}
for _v in REF_FLAGS.values():
    _v.update(spin_zeeman=True, orbital_zeeman=True, pairs="resonant", c3=False)


def load(code_root):
    """Import the engine modules from code_root (a folder holding pub_pipeline/ and kp/)."""
    global L, tb, tp, kp2, CODE_ROOT
    CODE_ROOT = os.path.abspath(code_root)
    for sub in ("kp", "pub_pipeline"):
        p = os.path.join(CODE_ROOT, sub)
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    for name in ("lpge", "tb_core", "tmd_params", "kp2"):
        sys.modules.pop(name, None)
    import tmd_params as _tp, tb_core as _tb, lpge as _L, kp2 as _kp2
    for m in (_tp, _tb, _L, _kp2):
        assert os.path.abspath(m.__file__).startswith(CODE_ROOT), (m.__file__, CODE_ROOT)
    L, tb, tp, kp2 = _L, _tb, _tp, _kp2
    return L, tb, tp, kp2


# ---------------------------------------------------------------- k-sets (engine's own arrays)
def grid_cart(N):
    """Engine full-BZ grid (lpge.grid) in Cartesian 1/a: (kx, ky, dA)."""
    al, be, Ji, dA = L.grid(N)
    return (2 * al).ravel(), (2 * be / np.sqrt(3)).ravel(), float(dA)


def disks(R, h):
    """Engine local disks around K and K' (lpge.local_kpts)."""
    kx, ky, dA = L.local_kpts(R, h)
    return np.asarray(kx, float), np.asarray(ky, float), float(dA)


def split_valleys(kpts):
    """Split a disk k-set into its K (kx>0) and K' (kx<0) halves without assuming any ordering."""
    kx, ky, dA = kpts
    m = kx > 0
    return (kx[m], ky[m], dA), (kx[~m], ky[~m], dA)


# ---------------------------------------------------------------- response
def eta_phys(material, model, functional, B, omega, nu, variant, kpts, lam=None):
    """Engine eta in physical SI units (A m V^-2 s^-1): raw (b,c-symmetrized) and C3-symmetrized."""
    fl = dict(FLAGS[variant]) if isinstance(variant, str) else dict(variant)
    r = L.eta(material, model, None, float(B), np.atleast_1d(np.asarray(omega, float)), nu, fl,
              functional=functional, kpts=kpts, lam=lam)
    s = -2.0 * r["si_factor"]
    return dict(raw=s * r["raw"], sym=s * r["sym"], si=r["si_factor"], viol=r["c3_violation"], code=r)


def eta_phys_grid(material, model, functional, N, B, omega, nu, variant):
    fl = dict(FLAGS[variant])
    r = L.eta(material, model, N, float(B), np.atleast_1d(np.asarray(omega, float)), nu, fl, functional=functional)
    s = -2.0 * r["si_factor"]
    return dict(raw=s * r["raw"], sym=s * r["sym"], si=r["si_factor"])


# ---------------------------------------------------------------- Hamiltonian-level access
def H_blocks(material, model, functional, kx, ky, B, spin=True):
    tb.set_config(material, functional)
    return L._H6(np.atleast_1d(np.asarray(kx, float)), np.atleast_1d(np.asarray(ky, float)), model, float(B), spin)


def band_quantities(material, model, functional, kx, ky, B, spin=True):
    """Per block: E, vx, vy, M, Omega, D from the engine's own internal functions."""
    tb.set_config(material, functional)
    ka, psm = tb.material_constants()
    kx = np.atleast_1d(np.asarray(kx, float)); ky = np.atleast_1d(np.asarray(ky, float))
    e = L.EPS
    Hs = {t: L._H6(kx + dx, ky + dy, model, float(B), spin)
          for t, (dx, dy) in {"0": (0, 0), "xp": (e, 0), "xm": (-e, 0), "yp": (0, e), "ym": (0, -e)}.items()}
    out = []
    for blk in (0, 1):
        E, V, vx, vy = L._block_quantities(Hs["0"][blk], Hs["xp"][blk], Hs["xm"][blk], Hs["yp"][blk], Hs["ym"][blk])
        M, Om = L._moment_omega(E, vx, vy, ka)
        out.append(dict(E=E, V=V, vx=vx, vy=vy, M=M, Omega=Om, D=1.0 + psm * B * Om))
    return out


def transitions(material, model, functional, kx, ky, B, variant):
    """lpge.chunk_transitions output: list over (block 0 c=1, block 0 c=2, block 1 c=1, block 1 c=2)
    of (e_res, W, Delta(2), g(2,2))."""
    tb.set_config(material, functional)
    ka, psm = tb.material_constants()
    fl = dict(FLAGS[variant]) if isinstance(variant, str) else dict(variant)
    return L.chunk_transitions(np.atleast_1d(np.asarray(kx, float)), np.atleast_1d(np.asarray(ky, float)),
                               model, float(B), fl, ka, psm)


def gap_K_engine(material, model, functional, lam=None):
    """Engine-style E_g (min over a line through K of the lowest interband gap of either block, B=0)."""
    tb.set_config(material, functional)
    if lam is not None:
        tb.LAM = lam
    q = np.linspace(-0.05, 0.05, 401); g = []
    for blk in L._H6(4 * np.pi / 3 + q, 0 * q, model, 0.0):
        ev = np.linalg.eigvalsh(blk); g.append((ev[:, 1] - ev[:, 0]).min())
    tb.set_config(material, functional)
    return float(min(g))
