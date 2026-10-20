"""Provenance helpers (protocol sec. 2, 6, 8).

tree_sha(code_root)      sha256 of "{relpath}\t{sha256}\n" lines (sorted) over the fixed engine file list
script_sha(path)         sha256 of one producing script
stamp(out_file, ...)     writes <out_file>.prov.json (hashes, argv, params, units/sign conventions, env, timing)
savefig_prov(fig, path)  saves a matplotlib figure with the sidecar JSON in /Subject, tree_sha in /Keywords,
                         CreationDate=None and SOURCE_DATE_EPOCH honoured, so rebuilds are byte-reproducible
require_verified(...)    refuses to run (SystemExit) unless verified.lock holds a passing fast run for the
                         current tree_sha and script sha; '--unverified' redirects output to scratch/

The run scripts do not call these yet (decision 8: no production edits until the v7 queue is done).
"""
import os, sys, json, time, hashlib, datetime, platform

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_FILES = ["kp/kp2.py", "pub_pipeline/lpge.py", "pub_pipeline/tb_core.py", "pub_pipeline/tmd_params.py"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_lines(code_root, files=ENGINE_FILES):
    return "".join(f"{rp}\t{sha256_file(os.path.join(code_root, rp))}\n" for rp in sorted(files))


def tree_sha(code_root, files=ENGINE_FILES):
    return hashlib.sha256(tree_lines(code_root, files).encode()).hexdigest()


def file_shas(code_root, files=ENGINE_FILES):
    return {rp: sha256_file(os.path.join(code_root, rp)) for rp in sorted(files)}


def resolve_script(cfg, code_root, rel):
    """Path of a producing script: code_root/rel, unless cfg['script_paths'] maps rel elsewhere."""
    o = (cfg or {}).get("script_paths", {}).get(rel)
    if o:
        return o if os.path.isabs(o) else os.path.normpath(os.path.join(HERE, o))
    return os.path.join(code_root, rel)


def script_shas(cfg, code_root):
    return {s: script_sha(resolve_script(cfg, code_root, s)) for s in cfg["scripts"] if os.path.exists(resolve_script(cfg, code_root, s))}


def script_sha(path):
    return sha256_file(path)


def _env():
    import numpy as np
    return dict(python=platform.python_version(), numpy=np.__version__, host=platform.node(), machine=platform.machine())


def stamp(out_file, code_root, script, argv, params, arrays, inputs=(), started=None, verified_lock=None):
    """Write <out_file>.prov.json. arrays = {name: {"units": ..., "sign_convention": "physical"|"code"}}."""
    side = dict(file=os.path.basename(out_file), sha256=sha256_file(out_file),
                tree_sha=tree_sha(code_root), engine_files=file_shas(code_root),
                script=os.path.relpath(script, code_root), script_sha=script_sha(script),
                argv=list(argv), params=params, arrays=arrays,
                inputs=[dict(path=p, sha256=sha256_file(p)) for p in inputs],
                env=_env(), created=datetime.datetime.now().isoformat(timespec="seconds"),
                duration_s=None if started is None else round(time.time() - started, 1),
                verified_lock=verified_lock)
    with open(out_file + ".prov.json", "w") as f:
        json.dump(side, f, indent=1)
    return side


def savefig_prov(fig, path, sidecar):
    """Save a figure reproducibly with provenance metadata (PDF: /Subject, /Keywords)."""
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    meta = {}
    if path.endswith(".pdf"):
        meta = {"Subject": json.dumps(sidecar, sort_keys=True), "Keywords": sidecar.get("tree_sha", ""),
                "CreationDate": None, "Producer": None}
    fig.savefig(path, metadata=meta, bbox_inches="tight")
    with open(path + ".prov.json", "w") as f:
        json.dump(dict(sidecar, file=os.path.basename(path), sha256=sha256_file(path)), f, indent=1)


def require_verified(code_root, script, lock_path=None, argv=None):
    """Call at the top of a run script. Returns the output folder to use ('.' or 'scratch/')."""
    argv = sys.argv if argv is None else argv
    lock_path = lock_path or os.path.join(code_root, "verification", "verified.lock")
    if "--unverified" in argv:
        argv.remove("--unverified")
        os.makedirs("scratch", exist_ok=True)
        print("UNVERIFIED run: output goes to scratch/ and is rejected by P2", flush=True)
        return "scratch"
    try:
        lock = json.load(open(lock_path))
    except Exception:
        raise SystemExit(f"no verified.lock at {lock_path}: run verify_all.py --fast first (or pass --unverified)")
    ts = tree_sha(code_root)
    rel = os.path.relpath(os.path.abspath(script), code_root)
    if lock.get("tree_sha") != ts:
        raise SystemExit(f"verified.lock is for tree {lock.get('tree_sha', '')[:12]}, engine is {ts[:12]}: rerun verify_all.py --fast")
    if lock.get("script_shas", {}).get(rel) != script_sha(script):
        raise SystemExit(f"{rel} changed since the last passing verification run")
    return "."


def write_lock(code_root, scripts, mode, report, lock_path, cfg=None):
    lock = dict(tree_sha=tree_sha(code_root),
                script_shas={s: script_sha(resolve_script(cfg, code_root, s)) for s in scripts if os.path.exists(resolve_script(cfg, code_root, s))},
                mode=mode, report=report, written=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=1)
    return lock
