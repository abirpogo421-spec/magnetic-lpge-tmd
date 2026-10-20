"""Core of the verification suite: result records, test registry, context, tolerance helpers.

Nothing in here knows about physics. Every test module registers its checks with @check(...).
A check is a function f(ctx) that returns one Result or a list of Results.
"""
import os, sys, json, time, hashlib, traceback, importlib, platform
import numpy as np

SUITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUSES = ("PASS", "FAIL", "KNOWN", "WARN", "STALE", "SKIP", "INFO")


class Result:
    def __init__(self, id, name, status, tol="", measured=None, detail="", claims=(), seconds=0.0):
        assert status in STATUSES, status
        self.id, self.name, self.status = id, name, status
        self.tol, self.measured, self.detail = str(tol), measured, detail
        self.claims, self.seconds = list(claims), seconds

    def as_dict(self):
        m = self.measured
        if isinstance(m, (np.floating, np.integer)):
            m = m.item()
        return dict(id=self.id, name=self.name, status=self.status, tol=self.tol, measured=m,
                    detail=self.detail, claims=self.claims, seconds=round(self.seconds, 2))


def fmt_meas(m):
    if m is None:
        return ""
    if isinstance(m, (float, np.floating)):
        return f"{float(m):.3e}"
    return str(m)


def judge(id, name, measured, tol, detail="", claims=(), warn_only=False):
    """PASS if measured <= tol (both finite numbers)."""
    ok = measured is not None and np.isfinite(measured) and measured <= tol
    st = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    return Result(id, name, st, tol=f"<= {tol:.1e}", measured=float(measured) if measured is not None else None,
                  detail=detail, claims=claims)


def exact(id, name, ok, detail="", measured=None, claims=()):
    return Result(id, name, "PASS" if ok else "FAIL", tol="exact", measured=measured, detail=detail, claims=claims)


def rel(x, y):
    """rel(X,Y) = max|X-Y| / max|Y| (protocol sec. 3)."""
    x, y = np.asarray(x), np.asarray(y)
    return float(np.max(np.abs(x - y)) / max(float(np.max(np.abs(y))), 1e-300))


def rel_scaled(x, y, scale):
    """|X-Y| relative to an external scale (single-valley max for K+K' sums)."""
    x, y = np.asarray(x), np.asarray(y)
    return float(np.max(np.abs(x - y)) / max(float(np.max(np.abs(scale))), 1e-300))


# ------------------------------------------------------------------------------------------------
REGISTRY = []   # list of dicts(id, family, func, modes, title, cost)


def check(id, family, title, modes=("fast", "full"), cost=1.0):
    def deco(f):
        REGISTRY.append(dict(id=id, family=family, func=f, modes=tuple(modes), title=title, cost=cost))
        return f
    return deco


class Ctx:
    """Shared state for one run: config, engine adapter, reference module, memo cache."""

    def __init__(self, cfg, mode, code_root, scratch):
        self.cfg, self.mode, self.code_root, self.scratch = cfg, mode, code_root, scratch
        self.memo = {}
        self.fixtures = {}            # name -> array, saved with --fixtures-out (mutation equivalence)
        self._eng = None
        self._ref = None

    def path(self, key):
        """Resolve a path from the config (relative to the suite folder)."""
        p = self.cfg["paths"][key]
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(SUITE_DIR, p))

    @property
    def eng(self):
        if self._eng is None:
            sys.path.insert(0, os.path.join(SUITE_DIR, "adapters"))
            import engine as E
            E.load(self.code_root)
            self._eng = E
        return self._eng

    @property
    def ref(self):
        if self._ref is None:
            sys.path.insert(0, os.path.join(SUITE_DIR, "ref"))
            import refimpl
            self._ref = refimpl
        return self._ref

    def cached(self, key, fn):
        if key not in self.memo:
            self.memo[key] = fn()
        return self.memo[key]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def env_info():
    try:
        import numpy as _np
        cfgs = _np.show_config(mode="dicts") if hasattr(_np, "show_config") else {}
        blas = cfgs.get("Build Dependencies", {}).get("blas", {}).get("name", "?") if isinstance(cfgs, dict) else "?"
    except Exception:
        blas = "?"
    return dict(python=platform.python_version(), numpy=np.__version__, blas=blas, host=platform.node(),
                machine=platform.machine(), system=platform.system())


def load_config(path=None):
    path = path or os.environ.get("VERIF_CONFIG") or os.path.join(SUITE_DIR, "config.json")
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(SUITE_DIR, path)
    with open(path) as f:
        cfg = json.load(f)
    if "extends" in cfg:                       # simple override mechanism
        base = load_config(os.path.join(os.path.dirname(path), cfg["extends"]))
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v
        cfg = base
    cfg["_config_file"] = os.path.abspath(path)
    return cfg
