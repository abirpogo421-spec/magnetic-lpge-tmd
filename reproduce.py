"""Rerun the notebooks top to bottom and rebuild every figure of the paper.

The notebooks in notebooks/ are the code. This script only executes them, several at a time, and saves each executed
copy (with its outputs and figures) to notebooks/executed/. The notebooks themselves are not changed.

    python3 reproduce.py                  # the notebooks behind the paper, at the paper settings (about 1 h)
    python3 reproduce.py --quick          # the quick settings from the comments in each Settings cell (about 2 h)
    python3 reproduce.py field edge       # selected notebooks

--quick reads the "(X for a quick look)" note next to each line of a notebook's Settings cell and runs with X instead.
Results are cached in notebooks/data/ under file names that include the settings. The long paper-setting runs are stored
there, so a plain run takes about an hour and only recomputes the short ones. To redo a run from scratch, delete its file.
"""
import argparse, os, re, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor

import nbformat
from nbclient import NotebookClient

warnings.filterwarnings("ignore", category=nbformat.warnings.MissingIDFieldWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(HERE, "notebooks")
OUT_DIR = os.path.join(NB_DIR, "executed")
NOTEBOOKS = ["nn_tnn", "response", "kp", "field", "edge"]     # the paper; soc.ipynb runs only when named (about 10 h)
QUICK = re.compile(r"^(\s*\w+\s*=\s*)([^#\n]+?)(\s*#.*\(([^\s;)]+) for a quick look)")


def quick_settings(nb):
    """Swap each Settings value for the quick value given in its comment."""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and cell.source.strip().startswith("## Settings"):
            code = nb.cells[i + 1]
            code.source = "\n".join(QUICK.sub(lambda m: m.group(1) + m.group(4) + m.group(3), line)
                                    for line in code.source.splitlines())
            return


def run(name, quick):
    nb = nbformat.read(os.path.join(NB_DIR, name + ".ipynb"), as_version=4)
    if quick:
        quick_settings(nb)
    t = time.time()
    NotebookClient(nb, timeout=None, startup_timeout=600, kernel_name="python3", resources={"metadata": {"path": NB_DIR}}).execute()
    nbformat.write(nb, os.path.join(OUT_DIR, name + ".ipynb"))
    return name, time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebooks", nargs="*", default=NOTEBOOKS, help="notebooks to run (default: all)")
    ap.add_argument("--quick", action="store_true", help="use the quick settings")
    ap.add_argument("--jobs", type=int, default=len(NOTEBOOKS), help="notebooks run at the same time")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(os.path.join(NB_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(NB_DIR, "figures"), exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")                         # one core per notebook
    with ProcessPoolExecutor(a.jobs) as pool:
        for name, sec in pool.map(run, a.notebooks, [a.quick]*len(a.notebooks)):
            print(f"{name:10s} done in {sec/60:6.1f} min -> notebooks/executed/{name}.ipynb", flush=True)
    print("figures in notebooks/figures/")


if __name__ == "__main__":
    sys.exit(main())
