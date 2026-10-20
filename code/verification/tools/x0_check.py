#!/usr/bin/env python3
"""X0: tagging must not change the PDF. Copies PAPER_DIR twice, tags one copy with tests/tag_tex.py, compiles both
with latexmk and compares pdftotext output and page counts.

    python3 tools/x0_check.py PAPER_DIR [--figures FIGDIR] [--out WORKDIR]
(After tagging the real Overleaf source, run it with an untagged copy of the previous version vs the tagged one.)
"""
import os, sys, shutil, subprocess, argparse, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(d):
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=d,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    txt = subprocess.run(["pdftotext", "-layout", os.path.join(d, "main.pdf"), "-"], capture_output=True, text=True).stdout
    pages = [l for l in subprocess.run(["pdfinfo", os.path.join(d, "main.pdf")], capture_output=True, text=True).stdout.splitlines() if l.startswith("Pages")]
    return txt, pages


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("paper"); ap.add_argument("--figures"); ap.add_argument("--out")
    a = ap.parse_args()
    w = a.out or tempfile.mkdtemp(prefix="x0_")
    for name in ("orig", "tagged"):
        d = os.path.join(w, name)
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(a.paper, d)
        if a.figures and not os.path.exists(os.path.join(d, "figures")):
            shutil.copytree(a.figures, os.path.join(d, "figures"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "tests", "tag_tex.py"), os.path.join(w, "tagged")], capture_output=True, text=True)
    print(r.stdout.strip())
    (t0, p0), (t1, p1) = build(os.path.join(w, "orig")), build(os.path.join(w, "tagged"))
    same = t0 == t1 and p0 == p1
    print(f"X0 {'PASS' if same else 'FAIL'}: pdftotext identical={t0 == t1}, pages {p0} vs {p1}  (work dir {w})")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
