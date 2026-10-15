"""Emit LaTeX rows for the paper's generalization tables from the Phase 12 per-test analyses (no hand-copied numbers).

Usage: python scripts/make_p12_tables.py  -> results/summary/p12_tables.tex
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("panel", "scripts/analyze_p12_panel.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
S = Path("results/summary")
LABEL = {"cartpole-rflip": "CartPole, reward flip", "acrobot-rflip": "Acrobot, reward flip",
         "si-actnoise": "SpaceInvaders, actuator noise", "freeway-rflip": "Freeway, reward flip",
         "si-obsnoise": "SpaceInvaders, observation noise", "si-extinction": "SpaceInvaders, reward extinction"}
BASE = {"acrobot-rflip": -500.0}


def ci(t, d=1, star=True):
    e, lo, hi = t
    s = "\\sig" if star and (lo > 0 or hi < 0) else ""
    return f"${e:+.{d}f}\\ci{{{lo:+.{d}f}}}{{{hi:+.{d}f}}}{s}$"


def pt(t, d=1):
    e, lo, hi = t
    return f"${e:+.{d}f}" + ("\\sig$" if (lo > 0 or hi < 0) else "$")


def pval(p):
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"${p:.4f}$" if p < .001 else f"${p:.3f}$"


def main():
    gen, carriers, gap = [], [], []
    audit_path=S/"p12_audit-posthoc.json"
    audited=json.loads(audit_path.read_text()).get("tests",{}) if audit_path.exists() else {}
    for gid, name, env, deficit in P.TESTS:
        md = S / f"p12_{name.replace('-', '_')}-analysis.md"
        done = P.ready(name)
        if not md.exists() or (not done and "--include-partial" not in sys.argv):
            continue
        r = P.parse(md)
        if gid in audited and audited[gid].get("complete"):
            exact=audited[gid]
            for key in ("total","gap","delay_auc","delay_end"):
                r[key]=exact[key]
            for c in ("actor","critic","opt"):
                r[c]=exact["effects"][c]
            r["actor_holm"]=exact["within_setting_holm"]["actor"]
            r["ccc_auc"]=exact["clean_auc"][0]
        d = 0 if abs(r["total"][0]) >= 20 else 1
        dmg = r["total"][2] < 0
        share = (f"{r['actor_share']}\\%" if dmg and r.get("actor_share") is not None else "--")
        verdict = ("met" if r.get("verdict") == "REPLICATES" else ("no gap detected" if not dmg else "not met")) if done else "descriptive only"
        gen.append(f"{LABEL[name]} ({gid}) & {r['n']} & {ci(r['total'], d)} & "
                   f"{ci(r['delay_auc'], d) if r.get('delay_auc') else '--'} & {share} & {verdict}" + r" \\")
        hp=float(r["actor_holm"])
        hpf=pval(hp)
        margin=.1*abs(r["ccc_auc"])
        carriers.append(f"{LABEL[name]} ({gid}) & {r['n']} & {ci(r['actor'],d)} & {ci(r['critic'],d)} & "
                        f"{ci(r['opt'],d)} & {hpf} & $\\pm{margin:.1f}$" + r" \\")
        scale = r["c_w1"] - BASE.get(name, 0.0)
        relgap = -r["gap"][0] / scale
        pred = "no gap" if relgap < 0.25 else ("gap" if relgap >= 0.60 else "no prediction")
        obs = "gap" if dmg else "no gap"
        hit = obs if pred == "no prediction" else obs + (" (hit)" if (pred == "gap") == dmg else " (miss)")
        if not done:
            pred += " (partial)"
            hit = "descriptive only"
        gap.append(f"{LABEL[name]} ({gid}) & {r['n']} & {100 * relgap:.0f}\\% & {ci(r['total'], d)} & "
                   f"${100 * r['total'][0] / scale:+.0f}\\%$ & {ci(r['delay_auc'], d) if r.get('delay_auc') else '--'} & "
                   f"{pred}; {hit} \\\\")
    # The independent H1 is a replication of G1, not a distinct setting in cross-setting counts.
    h1j=S/"p12_h1_cartpole_fresh-analysis.json"
    if h1j.exists():
        h1=json.loads(h1j.read_text())
        if h1.get("complete"):
            rr=P.parse(S/"p12_h1_cartpole_fresh-analysis.md")
            hp=h1["holm"]["actor"]
            hpf=pval(hp)
            carriers.append(f"CartPole, fresh replication (H1) & 8 & {ci(rr['actor'],0)} & {ci(rr['critic'],0)} & "
                            f"{ci(rr['opt'],0)} & {hpf} & $\\pm{h1['margin']:.1f}$" + r" \\")
    out = (["% tab:general rows (Phase 12)"] + gen + ["", "% tab:carriers rows (Phase 12)"] + carriers
           + ["", "% tab:gap rows (Phase 12; prediction; hit)"] + gap)
    (S / "p12_tables.tex").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
