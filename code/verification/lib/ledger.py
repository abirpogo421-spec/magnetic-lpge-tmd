"""--ledger-legacy: hash the engine files and scripts on this machine and record the legacy mapping
(protocol sec. 2). The v7 queue stores only lpge_sha256 in its outputs; this ledger maps that hash to the full
tree_sha once the other engine files have been checked to be identical."""
import os, sys, json, datetime
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import prov

PROTOCOL_FILES = {"pub_pipeline/lpge.py": "ff4a692dbbd465b045bd82519ed7602d918793ddbfe9484d1912c59411a7b2f8",
                  "pub_pipeline/tb_core.py": "54d6a0b8951eb8ee075b5fc2ed14d3fcd89407f6555ae41346dedded42454e31",
                  "pub_pipeline/tmd_params.py": "120e92e68e2a4a399adfa878f12e97c1f026fab0297c83ce10d5d701f4c0e4b7",
                  "kp/kp2.py": "5e7b0aad0e5df18ea50f0e541c5e01471a6d69fc84527006c9081440076ec1ac"}
PROTOCOL_TREE = "0c8e29953579c54f6659361a06f11af75146c1fe1fa86f0c24585d409aae97da"
PROTOCOL_SCRIPTS = {"v7jobs/edge_v7.py": "d17ce39359ea1257", "v7jobs/maps_v7.py": "8b5deb5bd808506b",
                    "v7jobs/chiw9_v7.py": "35f5f7656b7837f1", "v7jobs/lam_v7.py": "287e5094e129bb8b",
                    "v7jobs/decomp_v7.py": "dab427a706fb01cd", "v7jobs/fig_v7.py": "5f052a46c0f76868",
                    "kp/kp_fp_all_v7.py": "10dad1ad83f40b21"}


def main(cfg, code_root=None):
    code_root = os.path.abspath(code_root or (cfg["paths"]["code_root"] if os.path.isabs(cfg["paths"]["code_root"])
                                              else os.path.join(HERE, cfg["paths"]["code_root"])))
    p = os.path.join(HERE, "engine_ledger.json")
    led = json.load(open(p))
    files = prov.file_shas(code_root, cfg["engine_files"])
    ts = prov.tree_sha(code_root, cfg["engine_files"])
    scripts = prov.script_shas(cfg, code_root)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    diff = [f for f, h in PROTOCOL_FILES.items() if files.get(f) != h]
    print("tree_sha", ts, "(protocol", PROTOCOL_TREE[:16] + ")")
    for f, h in files.items():
        print(f"  {f:28s} {h[:16]} {'OK' if f not in diff else 'DIFFERS from protocol table'}")
    ent = next((e for e in led["legacy"] if e["legacy_lpge_sha256"] == files.get("pub_pipeline/lpge.py")), None)
    prev = (ent or {}).get("scripts", {})
    changed = []
    for s_, h in scripts.items():
        same = s_ in prev and (h.startswith(prev[s_]) or prev[s_].startswith(h))
        tag = "new" if s_ not in prev else ("OK" if same else "CHANGED since the last ledger check")
        if s_ in prev and not same:
            changed.append(s_)
        print(f"  {s_:28s} {h[:16]} {tag}" + (f" (protocol {PROTOCOL_SCRIPTS[s_]})" if s_ in PROTOCOL_SCRIPTS and not h.startswith(PROTOCOL_SCRIPTS[s_]) else ""))
    if diff:
        print("STALE: engine files differ from the legacy table:", diff, "-> outputs carrying lpge_sha256 ff4a692d are not covered")
        led.setdefault("checks", []).append(dict(time=now, host=os.uname().nodename, ok=False, differ=diff))
        json.dump(led, open(p, "w"), indent=1)
        return 1
    if ent is None:
        ent = dict(legacy_lpge_sha256=files["pub_pipeline/lpge.py"], tree_sha=ts, files=files, scripts={}, checked=[])
        led["legacy"].append(ent)
    hist = ent.setdefault("script_history", [])
    for s_ in changed:
        hist.append(dict(time=now, host=os.uname().nodename, script=s_, old=prev[s_], new=scripts[s_]))
    ent["scripts"] = scripts
    ent["checked"].append(dict(time=now, host=os.uname().nodename, scripts_changed=changed))
    json.dump(led, open(p, "w"), indent=1)
    print("legacy mapping confirmed:", files["pub_pipeline/lpge.py"][:16], "->", ts[:16], f"({len(ent['checked'])} checks)")
    if changed:
        print("NOTE: scripts changed since the last check:", changed, "- outputs written before the change used the old script (see script_history)")
    return 0
