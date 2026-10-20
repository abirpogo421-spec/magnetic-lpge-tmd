"""X: the reference implementation's own self-test.

fast: frozen-copy integrity (refimpl.py sha256 == the sha recorded with selftest_output.txt) and the recorded
      result line (96/96 code checks); plus three quick live checks (mu_B from CODATA, Hermiticity, NN closed forms
      run by L2). The 2.5-min selftest itself runs in full mode.
full: python3 ref/selftest.py, all code checks must pass.
"""
import os, sys, re, subprocess
from vcore import check, Result, exact, SUITE_DIR

FROZEN_REF_SHA = "856c59719e75848289c7fdb6ae6fbdf0b8ef3d2279c4752e10776972fba77679"


@check("X1", "X", "reference frozen copy unchanged; recorded selftest result", modes=("fast", "full"))
def x1(ctx):
    import vcore
    sha = vcore.sha256_file(os.path.join(SUITE_DIR, "ref", "refimpl.py"))
    txt = open(os.path.join(SUITE_DIR, "ref", "selftest_output.txt")).read()
    fails = re.findall(r"^\[FAIL\]", txt, re.M)
    npass = len(re.findall(r"^\[PASS\]", txt, re.M))
    return [exact("X1", "ref/refimpl.py sha256 == frozen 2026-10-14 copy", sha == FROZEN_REF_SHA, measured=sha[:16]),
            exact("X1", "recorded selftest: no FAIL lines", not fails, measured=f"{npass} PASS, {len(fails)} FAIL")]


@check("X2", "X", "reference selftest.py (all code checks)", modes=("full",))
def x2(ctx):
    p = subprocess.run([sys.executable, os.path.join(SUITE_DIR, "ref", "selftest.py")], capture_output=True, text=True,
                       cwd=os.path.join(SUITE_DIR, "ref"))
    fails = re.findall(r"^\[FAIL\].*$", p.stdout, re.M)
    return exact("X2", "selftest.py code checks", p.returncode == 0 and not fails,
                 measured=f"{len(re.findall(r'^.PASS.', p.stdout, re.M))} pass", detail="; ".join(fails[:5]))
