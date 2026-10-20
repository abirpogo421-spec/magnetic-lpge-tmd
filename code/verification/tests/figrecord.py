#!/usr/bin/env python3
"""Run a figure script and record what it actually plots.

    python3 figrecord.py OUT.json SCRIPT [script args...]

Every Figure.savefig call is intercepted (under Agg) and the plotted arrays are written to OUT.json:
  {output basename: [ {"lines": [[x...], [y...]], ...], "meshes": [{"array": [[...]], "shape": [...]}]} per axes ] }
Lines with fewer than 5 points (zero lines, markers of a single point) are kept too, flagged by length.
The script runs in its own folder semantics (cwd unchanged), with sys.argv = [SCRIPT, args...].
"""
import sys, os, json, runpy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as MF
from matplotlib.collections import QuadMesh

OUT, SCRIPT = sys.argv[1], sys.argv[2]
REC = {}
_orig = MF.Figure.savefig


def _arr(a):
    a = np.ma.filled(np.ma.asarray(a, float), np.nan)
    return a.tolist()


def _savefig(self, fname, *a, **k):
    name = os.path.splitext(os.path.basename(str(fname)))[0]
    if name not in REC:
        axes = []
        for ax in self.axes:
            if ax.get_label() == "<colorbar>":
                continue
            lines = [dict(x=_arr(l.get_xdata()), y=_arr(l.get_ydata()), marker=str(l.get_marker()), ls=str(l.get_linestyle()))
                     for l in ax.get_lines()]
            meshes = []
            for c in ax.collections:
                if isinstance(c, QuadMesh):
                    A = c.get_array()
                    meshes.append(dict(array=_arr(A), shape=list(np.shape(A)), coords_shape=list(np.shape(c.get_coordinates()))))
            axes.append(dict(lines=lines, meshes=meshes))
        REC[name] = axes
    return _orig(self, fname, *a, **k)


MF.Figure.savefig = _savefig
sys.argv = [SCRIPT] + sys.argv[3:]
try:
    runpy.run_path(SCRIPT, run_name="__main__")
finally:
    json.dump(REC, open(OUT, "w"))
